"""
art_rag.py
==========
Outil unifié pour le RAG spécialisé peinture / techniques picturales / histoire de l'art.

Ce fichier fusionne ce qui était auparavant trois scripts séparés :
    - ingest_documents.py  (extraction, chunking, auto-tagging, embeddings, stockage)
    - query_rag.py         (recherche sémantique + génération de réponse via LLM)
    - gui_ingest.py        (interface graphique CustomTkinter)

Tout est maintenant dans un seul module afin de ne plus avoir à gérer les imports
croisés entre fichiers (gui_ingest important core + query_rag, etc.).

Pipeline d'ingestion :
    fichier -> extraction texte -> chunking sémantique -> auto-tagging (LLM)
            -> embedding (LLM) -> stockage ChromaDB + SQLite

Pipeline de recherche :
    question -> embedding de la question
             -> recherche des chunks les plus proches sémantiquement (ChromaDB)
             -> injection de ces chunks dans le prompt du LLM de chat
             -> réponse du LLM, basée sur les extraits fournis, avec sources citées

Backend LLM :
    Ollama par défaut (LLM_BACKEND = "ollama"), LM Studio en alternative
    (LLM_BACKEND = "lmstudio", voir réglage LMSTUDIO_HOST) — réglable dans l'onglet
    Réglages de la GUI ou directement dans art_rag_config.json. Tous les appels LLM
    (chat, embeddings, vision, auto-tagging) passent par la couche d'abstraction
    llm_chat() / llm_embed() / llm_list_models() qui absorbe les différences entre
    les deux (voir la section "COUCHE D'ABSTRACTION BACKEND LLM" plus bas). En mode
    LM Studio, les noms de modèles (OLLAMA_*_MODEL) doivent correspondre à des
    identifiants LM Studio (ex: "qwen2.5-7b-instruct") et le serveur local LM Studio
    doit être démarré (Developer tab -> Local Server).

Usage CLI :
    python art_rag.py                                   # lance l'interface graphique (par défaut)
    python art_rag.py gui                                # idem, explicite

    python art_rag.py ingest --file "mon_livre.pdf"
    python art_rag.py ingest --dir "./mes_documents"
    python art_rag.py ingest --file "article.txt" --courant "Impressionnisme" --no-autotag
    python art_rag.py ingest --image "tableau.jpg" --artiste "Le Caravage" --oeuvre "Judith"

    python art_rag.py query "Quelles techniques utilisait Le Caravage pour le clair-obscur ?"
    python art_rag.py query "Décris le style pointilliste" --top-k 5
    python art_rag.py query "..." --courant "Impressionnisme"

Dépendances :
    pip install chromadb pypdf requests customtkinter --break-system-packages
    (customtkinter n'est nécessaire que pour le mode GUI)

    Backend Ollama (par défaut) : Ollama doit tourner localement (ollama serve) avec
    les modèles :
        - nomic-embed-text  (embeddings)
        - gemma4:31b ou équivalent (chat / auto-tagging, adapte OLLAMA_LLM_MODEL)
        - qwen3.5:35b ou équivalent (analyse vision d'œuvres, adapte OLLAMA_VISION_MODEL)

    Backend LM Studio (optionnel, LLM_BACKEND = "lmstudio") : le serveur local LM
    Studio doit tourner (Developer tab -> Local Server, port 1234 par défaut, voir
    LMSTUDIO_HOST) avec les modèles équivalents chargés/servis sous leurs identifiants
    LM Studio.
"""

import argparse
import contextlib
import hashlib
import html as html_lib
import io
import json
import queue
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock, Semaphore
from typing import Optional

import requests

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

try:
    from PIL import Image
    PIL_IMPORT_ERROR = None
except ImportError as _pil_err:
    Image = None
    PIL_IMPORT_ERROR = str(_pil_err)


# ============================================================
# CONFIGURATION
# ============================================================

# LLM_BACKEND : quel serveur d'inférence utiliser pour TOUS les appels LLM (chat, embeddings,
# vision, auto-tagging, etc.) : "ollama" (défaut) ou "lmstudio". Les deux exposent des API assez
# différentes (routes, format de payload/réponse, streaming) — c'est la couche d'abstraction
# llm_chat() / llm_embed() / llm_list_models() (voir plus bas, juste après check_ollama_models)
# qui absorbe cette différence. Le reste du pipeline (chunking, ChromaDB, SQLite, prompts) ne
# change pas d'un octet selon le backend choisi.
# Basculer vers LM Studio suppose que ses modèles soient déjà chargés/servis (Developer tab ->
# Local Server) et que les noms de modèles ci-dessous (OLLAMA_*_MODEL) correspondent à des
# identifiants LM Studio (ex: "qwen2.5-7b-instruct") plutôt qu'à des noms Ollama.
LLM_BACKEND = "ollama"          # "ollama" ou "lmstudio"
LMSTUDIO_HOST = "http://localhost:1234"   # utilisé uniquement si LLM_BACKEND == "lmstudio"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "bge-m3"
OLLAMA_LLM_MODEL = "gemma4:31b_rag_art"          # adapte selon ce que tu as en local (modèle "de référence", utilisé pour l'analyse d'œuvre vision et par défaut pour répondre)
OLLAMA_VISION_MODEL = "qwen3.5:35b"      # modèle vision pour l'analyse d'œuvres (adapte si besoin)

# OLLAMA_AUTOTAG_MODEL : modèle utilisé spécifiquement pour l'auto-tagging (extraction JSON
# courant/artiste/periode/technique à partir d'un chunk ou d'un échantillon de document).
# C'est une tâche d'extraction/classification simple, pas un exercice de raisonnement fin :
# un modèle nettement plus petit que le 31B (ex: "qwen2.5:7b-instruct-q4_K_M", "gemma3:12b",
# ou même "gemma3:4b" pour aller au plus vite) suffit largement et tourne 3 à 6x plus vite.
# Par défaut identique à OLLAMA_LLM_MODEL pour ne rien changer si tu ne configures rien, mais
# le vrai gain de vitesse sur l'auto-tagging par chunk (le principal goulot en mode "chunk")
# vient de le surcharger avec un modèle plus léger.
OLLAMA_AUTOTAG_MODEL = OLLAMA_LLM_MODEL

# OLLAMA_ANSWER_MODEL : modèle utilisé spécifiquement pour répondre aux questions RAG
# (ask_llm / query_rag). Par défaut identique à OLLAMA_LLM_MODEL, mais tu peux le
# surcharger avec un modèle plus petit et plus rapide en génération (ex:
# "qwen2.5:14b-instruct-q4_K_M" ou "gemma3:12b"). En RAG, l'essentiel du savoir factuel
# vient des extraits injectés dans le prompt, pas des paramètres du modèle — un modèle
# plus petit suffit souvent à bien synthétiser une réponse à partir d'un contexte déjà
# fourni, avec un gain de vitesse en génération qu'une quantization ne peut plus donner
# une fois qu'on est déjà en Q4_K_M sur le gros modèle.
OLLAMA_ANSWER_MODEL = "gemma4:12b"
# OLLAMA_EMBED_TIMEOUT à 60s suffit largement quand l'embedding tourne seul. Mais en
# granularité "chunk", des appels de chat 31B tournent en continu à côté (LLM_CONCURRENCY),
# et même si la VRAM tient, l'embedding doit alors attendre son tour de calcul GPU derrière
# ces générations lourdes : 60s peut ne pas suffire, pas parce que ça échoue, mais parce que
# ça fait la queue. On monte la marge pour absorber cette contention plutôt que d'échouer
# à tort un embedding qui aurait fini en réessayant une seconde de plus.
OLLAMA_EMBED_TIMEOUT = 180
OLLAMA_CHAT_TIMEOUT = 300                # un gros LLM (ex: 31B) peut prendre du temps, surtout en file d'attente
OLLAMA_VISION_TIMEOUT = 300              # l'analyse d'image peut être lente sur un gros modèle vision

# OLLAMA_KEEP_ALIVE : durée pendant laquelle Ollama garde le modèle chargé en VRAM après
# une requête. Sans ce paramètre, Ollama décharge le modèle après son délai par défaut
# (souvent 5 min), ce qui oblige à recharger un modèle 31B depuis le disque à chaque
# nouvelle question si tu enchaînes des recherches RAG dans le GUI. Avec 32 Go de VRAM
# (RTX 5090) il n'y a aucune raison de ne pas le garder chargé plus longtemps.
OLLAMA_KEEP_ALIVE = "30m"

# OLLAMA_NUM_CTX : taille de la fenêtre de contexte demandée à Ollama (en tokens). Sans
# valeur explicite, Ollama peut retomber sur une valeur par défaut trop petite (les
# extraits envoyés se font tronquer silencieusement) ou inutilement grande (temps de
# calcul gaspillé).
#
# ATTENTION : cette fenêtre est PARTAGÉE entre le prompt (system + historique de
# conversation + extraits RAG) et la réponse générée. En mode chat multi-tours, l'historique
# grossit à chaque tour ; si le prompt devient trop gros, il ne reste plus assez de place pour
# que le modèle termine sa réponse, qui se retrouve alors coupée en plein milieu (pas parce que
# le modèle a fini, mais parce que le contexte est saturé). 16384 laisse une marge confortable
# pour un historique de conversation + top_k=8 extraits RAG (~1200 caractères chacun) tout en
# restant bon marché en VRAM sur une RTX 5090 avec OLLAMA_FLASH_ATTENTION=1 et
# OLLAMA_KV_CACHE_TYPE=q8_0. Si les réponses se coupent encore, monte à 24576 ou 32768 (le
# 31B en Q4_K_M avec kv cache q8_0 doit encore passer confortablement dans 32 Go de VRAM),
# ou réduis MAX_CHAT_HISTORY_MESSAGES / TOP_K_DEFAULT pour laisser plus de place à la réponse.
OLLAMA_NUM_CTX = 16384

# OLLAMA_NUM_PREDICT : nombre maximum de tokens que le modèle est autorisé à générer pour une
# réponse. Sans valeur explicite, Ollama retombe sur son propre défaut (selon les versions,
# parfois une valeur assez basse), ce qui peut couper une réponse en plein milieu sans lien avec
# OLLAMA_NUM_CTX. 2048 tokens de génération correspond à une réponse assez longue et détaillée
# (plusieurs paragraphes) — largement au-dessus de ce qu'un JSON de prompt image ou une réponse
# de chat normale nécessite. -1 (illimité, borné uniquement par num_ctx) reste une option si tu
# veux vraiment ne jamais couper, au prix d'un risque de générations qui partent en boucle.
OLLAMA_NUM_PREDICT = 2048

# MAX_CHAT_HISTORY_MESSAGES : nombre maximum de messages (user+assistant confondus) de
# l'historique de conversation effectivement envoyés au LLM à chaque tour. Sans cette limite,
# l'historique grossit indéfiniment au fil d'une longue conversation et finit, combiné aux
# extraits RAG du tour, par saturer OLLAMA_NUM_CTX — c'est la cause la plus fréquente de
# réponses tronquées en cours de conversation. L'historique complet reste affiché à
# l'utilisateur (GUI/CLI) ; seule la fenêtre envoyée au modèle est tronquée aux tours les plus
# récents. 20 messages (~10 échanges) est un bon compromis pour garder la cohérence
# conversationnelle sans jamais approcher la limite de contexte.
MAX_CHAT_HISTORY_MESSAGES = 20

DB_PATH = "art_rag.db"
CHROMA_DIR = "./chroma_store"
CHROMA_COLLECTION = "art_knowledge"
SCHEMA_VERSION = 2

# --- Recherche web (alimentation du RAG) ---
# WEB_SEARCH_TIMEOUT : appels de recherche (listes de résultats, généralement rapides).
# WEB_FETCH_TIMEOUT : récupération d'une page complète + extraction (plus lent, surtout
# pour une page HTML volumineuse passée à trafilatura).
WEB_SEARCH_TIMEOUT = 15
WEB_FETCH_TIMEOUT = 20
WEB_USER_AGENT = "Mozilla/5.0 (compatible; ArtRAGBot/1.0; usage personnel de recherche documentaire)"
WIKIPEDIA_LANG = "fr"  # langue Wikipedia utilisée pour la recherche et l'extraction de texte

# WEB_SEARCH_LANGUAGES : langues interrogées pour la recherche web généraliste (ddgs). Une
# requête par langue (région DDG associée), résultats fusionnés et dédupliqués par URL.
WEB_SEARCH_LANGUAGES = ("fr", "en")
DDGS_REGION_BY_LANG = {"fr": "fr-fr", "en": "us-en"}

CHUNK_TARGET_CHARS = 1200     # taille cible d'un chunk
CHUNK_MAX_CHARS = 2000        # taille max avant découpage forcé
CHUNK_OVERLAP_CHARS = 150     # chevauchement entre chunks pour garder le contexte

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# TOP_K_DEFAULT : nombre d'extraits envoyés au LLM pour répondre. Plus ce nombre est
# élevé, plus le prefill (traitement du contexte avant génération) est long. 8 suffit
# pour la grande majorité des questions ; utilise --top-k pour une recherche plus
# exhaustive quand tu en as besoin.
TOP_K_DEFAULT = 8

# EMBED_CONCURRENCY : le modèle d'embedding est petit et rapide, 4-8 en simultané passe bien.
# LLM_CONCURRENCY   : sur un gros LLM (ex: 31B), chaque requête sature une bonne partie du GPU.
# En granularité "document" (1 seul appel LLM pour tout le fichier), LLM_CONCURRENCY=2 ne pose
# aucun souci puisqu'il n'y a qu'un appel au total. En granularité "chunk" en revanche, 2 appels
# de chat 31B tournent EN CONTINU pendant toute l'ingestion, et peuvent affamer en calcul GPU les
# requêtes d'embedding qui tournent à côté (même quand la VRAM tient). Si tu vois des timeouts
# d'embedding uniquement en mode "chunk", baisse LLM_CONCURRENCY à 1 pour laisser plus de cycles
# GPU aux embeddings (au prix d'un auto-tagging par chunk un peu plus lent).
EMBED_CONCURRENCY = 3
LLM_CONCURRENCY = 1

# AUTOTAG_EMBED_BATCH_SIZE : ne s'applique qu'en granularité "chunk" (1 appel LLM par chunk).
# Dans ce cas, le pipeline alterne par LOTS de cette taille entre une passe d'auto-tagging LLM
# et une passe d'embedding, plutôt que d'entrelacer les deux modèles chunk par chunk (qui force
# un swap VRAM à chaque chunk si Ollama ne garde qu'un seul modèle chargé) ou de tout taguer
# avant de tout embedder (qui retarde la première sauvegarde jusqu'à la fin du document entier).
# Avec 50, un livre de 500 chunks ne fait "que" 10 allers-retours LLM<->embedding au lieu de 500,
# et une interruption ne perd au pire que l'auto-tagging du lot en cours (jamais encore stocké),
# pas celui de tout le document. Baisse cette valeur si tu veux des sauvegardes plus fréquentes,
# augmente-la si tu préfères minimiser encore le nombre de swaps VRAM.
AUTOTAG_EMBED_BATCH_SIZE = 50

# AUTOTAG_GRANULARITY : c'est le vrai levier de vitesse.
#   "document" (par défaut) : UN SEUL appel LLM par fichier (sur un échantillon du texte),
#       le résultat (courant/artiste/periode/technique) est appliqué à tous les chunks.
#       Idéal quand un fichier traite d'un seul sujet cohérent (un livre, un article) —
#       ce qui divise le nombre d'appels LLM par ~nb_chunks (500 chunks -> 1 appel au lieu de 500).
#   "chunk" : un appel LLM par chunk (comportement d'origine, plus lent mais plus fin si un
#       même fichier mélange plusieurs artistes/courants différents).
AUTOTAG_GRANULARITY = "document"
DOCUMENT_AUTOTAG_SAMPLE_CHARS = 6000  # taille de l'échantillon utilisé pour le tag au niveau document

# ============================================================
# PARAMÈTRES PERSISTANTS (config.json)
# ============================================================
# Toutes les constantes ci-dessus sont des valeurs par défaut "codées en dur". Le fichier
# art_rag_config.json (créé à côté de ce script) permet de les surcharger sans toucher au
# code, et l'onglet "Paramètres" du GUI lit/écrit ce même fichier. SETTINGS_SCHEMA décrit
# quels noms de variables globales sont éditables, leur type, leur section d'affichage et
# si un changement nécessite un redémarrage de l'appli pour prendre effet.
CONFIG_PATH = Path(__file__).resolve().parent / "art_rag_config.json"

# (clé globale, type, section, libellé, description, redémarrage requis ?)
SETTINGS_SCHEMA = [
    # --- Backend LLM ---
    ("LLM_BACKEND", str, "Backend", "LLM backend", "'ollama' or 'lmstudio' — applies to chat, embeddings, vision, auto-tagging", False),
    ("LMSTUDIO_HOST", str, "Backend", "LM Studio server URL", "Ex: http://localhost:1234 — used only if backend = lmstudio", False),
    # --- Connexion Ollama ---
    ("OLLAMA_HOST", str, "Ollama", "Ollama server URL", "Ex: http://localhost:11434", False),
    ("OLLAMA_EMBED_MODEL", str, "Ollama", "Embedding model", "Used to index and query the database", False),
    ("OLLAMA_LLM_MODEL", str, "Ollama", "Reference LLM model", "Vision analysis + default value if the others aren't set", False),
    ("OLLAMA_VISION_MODEL", str, "Ollama", "Vision model", "Used for artwork analysis (Artwork Analysis tab)", False),
    ("OLLAMA_ANSWER_MODEL", str, "Ollama", "Answer model (RAG chat)", "Used to answer questions in Chat", False),
    ("OLLAMA_AUTOTAG_MODEL", str, "Ollama", "Auto-tagging model", "Used to extract movement/artist/period/technique", False),
    ("OLLAMA_KEEP_ALIVE", str, "Ollama", "VRAM keep-alive duration", "E.g.: 30m, 1h, -1 (indefinitely)", False),
    # --- Performance ---
    ("OLLAMA_NUM_CTX", int, "Performance", "Context size (tokens)", "Shared prompt + response window", False),
    ("OLLAMA_NUM_PREDICT", int, "Performance", "Max response length (tokens)", "-1 for unlimited (bounded by context)", False),
    ("OLLAMA_EMBED_TIMEOUT", int, "Performance", "Embedding timeout (s)", "", False),
    ("OLLAMA_CHAT_TIMEOUT", int, "Performance", "Chat timeout (s)", "", False),
    ("OLLAMA_VISION_TIMEOUT", int, "Performance", "Vision timeout (s)", "", False),
    ("EMBED_CONCURRENCY", int, "Performance", "Embedding concurrency", "Requires a restart", True),
    ("LLM_CONCURRENCY", int, "Performance", "LLM concurrency", "Lower to 1 if embedding timeouts occur during ingestion", True),
    ("AUTOTAG_EMBED_BATCH_SIZE", int, "Performance", "Autotag/embedding batch size", "'chunk' granularity only", False),
    ("TOP_K_DEFAULT", int, "Performance", "Number of RAG excerpts (top-k)", "Sent to the LLM to answer", False),
    ("MAX_CHAT_HISTORY_MESSAGES", int, "Performance", "Chat history sent to the LLM", "In number of messages (user+assistant)", False),
    # --- Base de connaissances ---
    ("DB_PATH", str, "Knowledge base", "SQLite database path", "Requires a restart", True),
    ("CHROMA_DIR", str, "Knowledge base", "ChromaDB folder", "Requires a restart", True),
    ("CHROMA_COLLECTION", str, "Knowledge base", "Collection name", "Requires a restart", True),
    ("AUTOTAG_GRANULARITY", str, "Knowledge base", "Auto-tag granularity", "'document' (fast) or 'chunk' (fine-grained)", False),
    ("DOCUMENT_AUTOTAG_SAMPLE_CHARS", int, "Knowledge base", "Auto-tag sample (chars)", "'document' granularity only", False),
    # --- Recherche web ---
    ("WIKIPEDIA_LANG", str, "Web Search", "Wikipedia language", "Ex: fr, en", False),
    ("WEB_SEARCH_TIMEOUT", int, "Web Search", "Search timeout (s)", "", False),
    ("WEB_FETCH_TIMEOUT", int, "Web Search", "Page fetch timeout (s)", "", False),
]


def load_config():
    """Charge art_rag_config.json (s'il existe) et surcharge les constantes globales
    correspondantes. Silencieux si le fichier n'existe pas encore (première utilisation) :
    les valeurs par défaut codées ci-dessus restent actives."""
    if not CONFIG_PATH.exists():
        return
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[!] Impossible de lire {CONFIG_PATH.name} ({e}) — valeurs par défaut utilisées.")
        return
    for key, cast, *_ in SETTINGS_SCHEMA:
        if key in data:
            try:
                globals()[key] = cast(data[key])
            except (TypeError, ValueError):
                print(f"[!] Valeur invalide pour '{key}' dans {CONFIG_PATH.name}, ignorée.")


def save_config():
    """Sauvegarde la valeur courante de chaque paramètre de SETTINGS_SCHEMA dans
    art_rag_config.json."""
    data = {key: globals()[key] for key, *_ in SETTINGS_SCHEMA}
    CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


load_config()

_db_lock = RLock()
_embed_semaphore = Semaphore(EMBED_CONCURRENCY)
_llm_semaphore = Semaphore(LLM_CONCURRENCY)

ANSWER_SYSTEM_PROMPT = """Tu es un professionnel chevronné des arts graphiques : historien de l'art, praticien
de la peinture et directeur artistique, avec une connaissance fine des courants, des techniques picturales,
des matières et de la culture visuelle au sens large (peinture, illustration, photographie, cinéma, design).
Tu discutes avec l'utilisateur comme un collègue passionné le ferait — de façon naturelle, précise, jamais
scolaire ni robotique. Ce n'est pas un simple Q&R : c'est une vraie conversation qui peut dévier, revenir en
arrière, ou enchaîner plusieurs sujets.

Quand la conversation porte sur l'art (histoire de l'art, courants, techniques, artistes, œuvres, composition,
matière, lumière, esthétique visuelle...), des extraits pertinents de la base de connaissances te sont fournis
ci-dessous : appuie-toi systématiquement dessus pour ancrer ta réponse dans des faits vérifiés plutôt que dans
de vagues souvenirs, et cite les éléments factuels précis (dates, noms, techniques) qu'ils contiennent. Si les
extraits ne couvrent pas ce qui est demandé, dis-le honnêtement et complète avec ton expertise générale, sans
jamais inventer une source ni prétendre qu'un extrait dit quelque chose qu'il ne dit pas.

Si aucun extrait ne t'est fourni pour ce tour, c'est que le message sortait du champ artistique ou qu'aucune
ressource pertinente n'a été trouvée : réponds alors simplement en professionnel, sans forcer une référence à
la base de connaissances.

Réponds en français, sauf demande explicite contraire (par exemple lorsqu'on te demande un prompt en anglais
pour une IA générative d'image)."""


# ============================================================
# STRUCTURES DE DONNÉES
# ============================================================

@dataclass
class ChunkMetadata:
    courant: str = ""
    artiste: str = ""
    periode: str = ""
    technique: str = ""
    oeuvre: str = ""              # titre de l'œuvre analysée (si applicable)
    mots_cles: list = field(default_factory=list)
    confiance_autotag: str = ""   # "auto", "manuel" ou "vision"
    type_contenu: str = "document"  # "document" (texte ingéré) ou "analyse_oeuvre" (analyse d'une œuvre)


@dataclass
class Chunk:
    id: str
    source_file: str
    chunk_index: int
    text: str
    metadata: ChunkMetadata


# ============================================================
# IDENTITE DES SOURCES ET MIGRATION
# ============================================================

def make_local_source_id(path: Path) -> str:
    """Identifiant stable d'un fichier local, distinct de son simple nom."""
    return str(path.resolve())


def content_fingerprint(text: str) -> str:
    """Empreinte du texte nettoye, utilisee pour detecter les modifications."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _migration_backup(db_path: str):
    """Sauvegarde SQLite et Chroma avant la premiere migration de schema."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    db = Path(db_path)
    if db.exists():
        backup = db.with_name(f"{db.stem}.pre_migration_{stamp}{db.suffix}")
        # sqlite3.backup() inclut correctement les donnees encore presentes dans le WAL.
        source_conn = sqlite3.connect(db)
        backup_conn = sqlite3.connect(backup)
        try:
            source_conn.backup(backup_conn)
        finally:
            backup_conn.close()
            source_conn.close()
        print(f"[MIGRATION] Sauvegarde SQLite creee : {backup}")

    chroma = Path(CHROMA_DIR)
    if chroma.exists() and chroma.is_dir():
        backup_dir = chroma.with_name(f"{chroma.name}.pre_migration_{stamp}")
        shutil.copytree(chroma, backup_dir)
        print(f"[MIGRATION] Sauvegarde Chroma creee : {backup_dir}")


# ============================================================
# EXTRACTION DE TEXTE
# ============================================================

def extract_text_from_pdf(path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf n'est pas installé : pip install pypdf --break-system-packages")
    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as e:
            print(f"  [!] Erreur extraction page dans {path.name}: {e}")
    return "\n\n".join(pages_text)


def extract_text_from_txt(path: Path) -> str:
    # tente plusieurs encodages, les vieux documents traînent souvent en latin-1
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Impossible de décoder {path.name}")


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(path)
    else:
        raise ValueError(f"Extension non supportée : {ext}")


# ============================================================
# NETTOYAGE
# ============================================================

def clean_text(text: str) -> str:
    # normalise les sauts de ligne multiples, supprime les espaces en fin de ligne
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # colle les mots coupés par une césure en fin de ligne PDF ("techni-\nque")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()


# ============================================================
# CHUNKING SÉMANTIQUE (par paragraphe, pas par taille brute)
# ============================================================

def split_into_paragraphs(text: str) -> list:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def semantic_chunk(text: str) -> list:
    """
    Regroupe les paragraphes jusqu'à atteindre CHUNK_TARGET_CHARS,
    sans jamais couper un paragraphe en plein milieu (sauf s'il dépasse
    CHUNK_MAX_CHARS à lui seul, auquel cas on découpe par phrases).
    Ajoute un léger overlap entre chunks consécutifs pour préserver le contexte.
    """
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > CHUNK_MAX_CHARS:
            # paragraphe trop long (ex: table des matières mal parsée) -> découpe par phrases
            if current:
                chunks.append(current)
                current = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buffer = ""
            for sentence in sentences:
                if len(buffer) + len(sentence) > CHUNK_TARGET_CHARS and buffer:
                    chunks.append(buffer)
                    buffer = sentence
                else:
                    buffer = f"{buffer} {sentence}".strip()
            if buffer:
                chunks.append(buffer)
            continue

        if len(current) + len(para) > CHUNK_TARGET_CHARS and current:
            chunks.append(current)
            # overlap : on repart avec la fin du chunk précédent
            current = current[-CHUNK_OVERLAP_CHARS:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}".strip()

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# VÉRIFICATION PRÉVENTIVE DES MODÈLES OLLAMA
# ============================================================

def check_ollama_models() -> bool:
    """
    Vérifie que le backend actif (LLM_BACKEND : Ollama ou LM Studio) répond et que les modèles
    requis sont bien disponibles. Un 404 sur /api/chat ou /api/embeddings (Ollama) vient quasi
    toujours d'un modèle absent, pas d'un problème réseau : on le détecte ici plutôt que de
    laisser échouer chunk par chunk sur un document de 300 pages.
    Retourne True si tout est OK, False sinon (avec message explicite).
    """
    backend_label = "LM Studio" if LLM_BACKEND == "lmstudio" else "Ollama"
    host = LMSTUDIO_HOST if LLM_BACKEND == "lmstudio" else OLLAMA_HOST
    try:
        installed = set(llm_list_models(timeout=10))
        # Ollama liste parfois avec le tag (":latest") en plus du nom court
        installed_base = {name.split(":")[0] for name in installed}
    except requests.RequestException as e:
        print(f"[ERREUR] Impossible de contacter {backend_label} sur {host} : {e}")
        if LLM_BACKEND == "lmstudio":
            print("         Vérifie que le serveur local LM Studio est démarré (Developer tab -> Local Server)")
        else:
            print("         Vérifie qu'Ollama tourne (commande : ollama serve)")
        return False

    missing = []
    required_models = {OLLAMA_EMBED_MODEL, OLLAMA_LLM_MODEL, OLLAMA_ANSWER_MODEL, OLLAMA_AUTOTAG_MODEL}
    for required in required_models:
        base_name = required.split(":")[0]
        if required not in installed and base_name not in installed_base:
            missing.append(required)

    if missing:
        print(f"[ERREUR] Modèle(s) {backend_label} manquant(s) ou non chargé(s) :")
        for m in missing:
            if LLM_BACKEND == "lmstudio":
                print(f"           - {m}  ->  charge-le dans LM Studio (Developer tab / My Models)")
            else:
                print(f"           - {m}  ->  ollama pull {m}")
        print(f"[INFO] Modèles actuellement disponibles : {sorted(installed) or '(aucun)'}")
        print("[INFO] Si tes modèles portent un autre nom, adapte OLLAMA_LLM_MODEL / OLLAMA_ANSWER_MODEL /")
        print("       OLLAMA_AUTOTAG_MODEL / OLLAMA_EMBED_MODEL en haut de art_rag.py (ces mêmes noms")
        print("       servent pour LM Studio quand LLM_BACKEND = \"lmstudio\" : mets-y les identifiants LM Studio).")
        return False

    print(f"[OK] Modèles {backend_label} détectés : {', '.join(sorted(required_models))}")
    return True


# ============================================================
# COUCHE D'ABSTRACTION BACKEND LLM (Ollama / LM Studio)
# ============================================================
#
# Point d'entrée unique pour TOUT appel LLM du reste du fichier : llm_chat(), llm_embed(),
# llm_list_models(). Selon LLM_BACKEND, ces fonctions dispatchent vers l'implémentation Ollama
# (native, /api/chat, /api/embeddings, /api/tags) ou LM Studio (compatible OpenAI, /v1/chat/
# completions, /v1/embeddings, /v1/models). Les fonctions appelantes (autotag_chunk, get_embedding,
# ask_llm, chat_llm, analyze_artwork_vision, etc.) ne connaissent plus les détails de transport :
# elles passent model/messages/stream/etc. et récupèrent du texte, comme avant.
#
# Erreurs : les deux implémentations appellent resp.raise_for_status(), donc le code appelant
# existant (except requests.HTTPError as e: if e.response.status_code == 404: ...) continue de
# fonctionner sans changement.
#
# Différences volontairement absorbées ici :
#   - keep_alive (VRAM) : concept propre à Ollama, ignoré côté LM Studio (qui gère le JIT loading
#     / idle TTL / auto-evict via ses propres réglages serveur, pas requête par requête).
#   - stats de perf : Ollama renvoie prompt_eval_count/eval_count/durations en nanosecondes ;
#     LM Studio (API compatible OpenAI) ne renvoie qu'un usage.prompt_tokens/completion_tokens,
#     sans détail de durée. format_llm_stats() gère les deux (durées à 0 si absentes).
#   - vision : Ollama attend "images": [base64...] sur le message ; LM Studio (format OpenAI)
#     attend un content en liste de blocs {"type": "image_url", "image_url": {"url": "data:..."}}.


def _ollama_chat(model, messages, *, stream, json_mode, timeout, num_predict, images, on_token, stats):
    if images:
        messages = messages[:-1] + [{**messages[-1], "images": images}]

    options = {"num_ctx": OLLAMA_NUM_CTX}
    if num_predict is not None:
        options["num_predict"] = num_predict

    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": options,
    }
    if json_mode:
        payload["format"] = "json"

    if not stream:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        _collect_stats(stats, data)
        return data["message"]["content"]

    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=timeout, stream=True)
    resp.raise_for_status()
    full_text = []
    for line in resp.iter_lines():
        if not line:
            continue
        piece = json.loads(line)
        token = piece.get("message", {}).get("content", "")
        if token:
            full_text.append(token)
            if on_token:
                on_token(token)
        if piece.get("done"):
            _collect_stats(stats, piece)
            break
    return "".join(full_text)


def _collect_stats_lmstudio(stats: Optional[dict], data: dict):
    """Équivalent de _collect_stats pour une réponse (compatible OpenAI) de LM Studio : pas de
    détail de durée (chargement/prefill/génération), seulement les tokens via `usage`."""
    if stats is None:
        return
    usage = (data or {}).get("usage") or {}
    stats.update({
        "total_duration": 0, "load_duration": 0,
        "prompt_eval_count": usage.get("prompt_tokens", 0), "prompt_eval_duration": 0,
        "eval_count": usage.get("completion_tokens", 0), "eval_duration": 0,
    })
    choices = (data or {}).get("choices") or []
    finish_reason = choices[0].get("finish_reason") if choices else None
    # "length" côté OpenAI == réponse coupée par max_tokens, même sémantique que done_reason
    # "length" côté Ollama : format_llm_stats() s'appuie dessus pour son avertissement.
    stats["done_reason"] = finish_reason or ""


def _lmstudio_chat(model, messages, *, stream, json_mode, timeout, num_predict, images, on_token, stats):
    if images:
        last = messages[-1]
        content_blocks = [{"type": "text", "text": last["content"]}]
        for b64 in images:
            content_blocks.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        messages = messages[:-1] + [{**last, "content": content_blocks}]

    payload = {"model": model, "messages": messages, "stream": stream}
    if num_predict is not None and num_predict > 0:
        payload["max_tokens"] = num_predict
    # NB : contrairement à Ollama et à l'API OpenAI officielle, le serveur LM Studio (llama.cpp)
    # rejette response_format={"type": "json_object"} avec une 400 ("must be 'json_schema' or
    # 'text'") — voir lmstudio-ai/lmstudio-bug-tracker#189. On ne l'envoie donc pas ici : les
    # prompts système demandent déjà explicitement du JSON, et le code appelant passe le résultat
    # par _extract_json_object() pour tolérer un éventuel texte parasite autour du JSON.

    if not stream:
        resp = requests.post(f"{LMSTUDIO_HOST}/v1/chat/completions", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        _collect_stats_lmstudio(stats, data)
        return data["choices"][0]["message"]["content"]

    resp = requests.post(f"{LMSTUDIO_HOST}/v1/chat/completions", json=payload, timeout=timeout, stream=True)
    resp.raise_for_status()
    full_text = []
    last_chunk = None
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        if raw == "[DONE]":
            break
        piece = json.loads(raw)
        last_chunk = piece
        choices = piece.get("choices") or []
        if choices:
            token = choices[0].get("delta", {}).get("content", "")
            if token:
                full_text.append(token)
                if on_token:
                    on_token(token)
    if last_chunk:
        _collect_stats_lmstudio(stats, last_chunk)
    return "".join(full_text)


def llm_chat(
    model: str,
    messages: list,
    *,
    stream: bool = False,
    json_mode: bool = False,
    timeout: int = OLLAMA_CHAT_TIMEOUT,
    num_predict: Optional[int] = None,
    images: Optional[list] = None,
    on_token=None,
    stats: Optional[dict] = None,
) -> str:
    """
    Point d'entrée unique pour un appel de chat (avec ou sans image, avec ou sans streaming,
    avec ou sans sortie JSON forcée), quel que soit LLM_BACKEND. Retourne le texte de la réponse
    complète. `images` : liste de chaînes base64 (0 ou 1 image généralement, plusieurs possibles
    selon le modèle vision utilisé).
    """
    kwargs = dict(
        model=model, messages=messages, stream=stream, json_mode=json_mode,
        timeout=timeout, num_predict=num_predict, images=images,
        on_token=on_token, stats=stats,
    )
    if LLM_BACKEND == "lmstudio":
        return _lmstudio_chat(**kwargs)
    return _ollama_chat(**kwargs)


def llm_embed(model: str, text: str, timeout: int = OLLAMA_EMBED_TIMEOUT) -> list:
    """Point d'entrée unique pour un embedding, quel que soit LLM_BACKEND."""
    if LLM_BACKEND == "lmstudio":
        resp = requests.post(f"{LMSTUDIO_HOST}/v1/embeddings", json={"model": model, "input": text}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": model, "prompt": text, "keep_alive": OLLAMA_KEEP_ALIVE},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def llm_list_models(timeout: int = 10) -> list:
    """Liste les modèles disponibles/chargés sur le backend actif (LLM_BACKEND)."""
    if LLM_BACKEND == "lmstudio":
        resp = requests.get(f"{LMSTUDIO_HOST}/v1/models", timeout=timeout)
        resp.raise_for_status()
        return sorted(m["id"] for m in resp.json().get("data", []))
    resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=timeout)
    resp.raise_for_status()
    return sorted(m["name"] for m in resp.json().get("models", []))


# ============================================================
# AUTO-TAGGING VIA OLLAMA (extraction JSON structurée)
# ============================================================

AUTOTAG_SYSTEM_PROMPT = """Tu es un expert en histoire de l'art et techniques picturales.
Analyse le texte fourni et extrais UNIQUEMENT les informations explicitement présentes ou clairement déductibles.
Réponds STRICTEMENT en JSON valide, sans aucun texte avant ou après, selon ce schéma exact :

{
  "courant": "nom du courant artistique si identifiable, sinon chaîne vide",
  "artiste": "nom de l'artiste principal évoqué, sinon chaîne vide",
  "periode": "période ou dates si mentionnées, sinon chaîne vide",
  "technique": "technique picturale évoquée (ex: sfumato, glacis, empâtement), sinon chaîne vide",
  "oeuvre": "titre de l'œuvre précise évoquée, sinon chaîne vide",
  "mots_cles": ["liste", "de", "mots-cles", "visuels", "pertinents", "pour", "generation", "image"]
}

Ne fabrique aucune information absente du texte. Si tu ne sais pas, laisse une chaîne vide."""


def autotag_chunk(text: str) -> ChunkMetadata:
    """Appelle le backend LLM actif (llm_chat) pour extraire les métadonnées structurées d'un chunk."""
    messages = [
        {"role": "system", "content": AUTOTAG_SYSTEM_PROMPT},
        {"role": "user", "content": text[:3000]},  # on limite pour la vitesse
    ]

    with _llm_semaphore:
        try:
            content = llm_chat(OLLAMA_AUTOTAG_MODEL, messages, json_mode=True, timeout=OLLAMA_CHAT_TIMEOUT)
            data = json.loads(_extract_json_object(content))
            return ChunkMetadata(
                courant=data.get("courant", ""),
                artiste=data.get("artiste", ""),
                periode=data.get("periode", ""),
                technique=data.get("technique", ""),
                oeuvre=data.get("oeuvre", ""),
                mots_cles=data.get("mots_cles", []) or [],
                confiance_autotag="auto",
            )
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                print(f"  [!] Auto-tagging: modèle '{OLLAMA_AUTOTAG_MODEL}' introuvable (404). "
                      f"Vérifie 'ollama pull {OLLAMA_AUTOTAG_MODEL}' ou corrige OLLAMA_AUTOTAG_MODEL.")
            else:
                print(f"  [!] Auto-tagging échoué ({e}), métadonnées vides pour ce chunk")
            return ChunkMetadata(confiance_autotag="echec")
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"  [!] Auto-tagging échoué ({e}), métadonnées vides pour ce chunk")
            return ChunkMetadata(confiance_autotag="echec")


def autotag_document(full_text: str, sample_chars: Optional[int] = None) -> ChunkMetadata:
    """
    Version "document" de l'auto-tagging : UN SEUL appel LLM sur un échantillon du texte
    (début du document, là où se trouve généralement le sujet principal), dont le résultat
    est ensuite appliqué à tous les chunks. C'est ce qui permet de passer de ~500 appels LLM
    à 1 seul pour un gros fichier — le principal goulot d'étranglement du pipeline.

    sample_chars : permet de surcharger DOCUMENT_AUTOTAG_SAMPLE_CHARS pour ce livre précis
    (ex: un gros ouvrage qui mélange plusieurs techniques/courants selon les chapitres peut
    bénéficier d'un échantillon plus large pour que le LLM ait une chance de voir des détails
    de technique picturale au-delà de la simple introduction).
    """
    effective_sample_chars = sample_chars if sample_chars is not None else DOCUMENT_AUTOTAG_SAMPLE_CHARS
    sample = full_text[:effective_sample_chars]
    metadata = autotag_chunk(sample)
    # mots_cles perd en granularité par chunk mais reste utile comme résumé global du document
    return metadata


# ============================================================
# GESTION DE LA REPRISE (chunks déjà traités pour un fichier)
# ============================================================

def get_processed_chunk_indices(conn: sqlite3.Connection, source_file: str) -> set:
    """
    Retourne l'ensemble des chunk_index déjà présents en base pour ce fichier.
    Utilisé pour reprendre une ingestion interrompue sans refaire les appels Ollama
    déjà effectués (le gros du temps de traitement).
    """
    cur = conn.execute("SELECT DISTINCT chunk_index FROM chunks WHERE source_file = ?", (source_file,))
    return {row[0] for row in cur.fetchall()}


def delete_source_file(conn: sqlite3.Connection, chroma_collection, source_file: str):
    """Supprime toutes les traces d'un fichier (SQLite + ChromaDB) avant une ré-ingestion forcée propre."""
    with _db_lock:
        conn.execute("DELETE FROM chunks WHERE source_file = ?", (source_file,))
        conn.execute("DELETE FROM ingested_files WHERE source_file = ?", (source_file,))
        conn.commit()
    try:
        chroma_collection.delete(where={"source_file": source_file})
    except Exception as e:
        print(f"  [!] Nettoyage ChromaDB partiel pour {source_file} : {e}")


# ============================================================
# EMBEDDINGS VIA OLLAMA
# ============================================================

def get_embedding(text: str, retries: int = 3) -> Optional[list]:
    with _embed_semaphore:
        for attempt in range(retries):
            try:
                return llm_embed(OLLAMA_EMBED_MODEL, text, timeout=OLLAMA_EMBED_TIMEOUT)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    # Modèle absent : inutile de retenter, ça échouera toujours pareil
                    print(f"  [!] Embedding: modèle '{OLLAMA_EMBED_MODEL}' introuvable (404). "
                          f"Lance 'ollama pull {OLLAMA_EMBED_MODEL}'. Abandon des tentatives pour ce chunk.")
                    return None
                print(f"  [!] Embedding tentative {attempt + 1}/{retries} échouée : {e}")
                time.sleep(2)
            except (requests.RequestException, KeyError) as e:
                print(f"  [!] Embedding tentative {attempt + 1}/{retries} échouée : {e}")
                time.sleep(2)
    return None


# ============================================================
# STOCKAGE SQLite (métadonnées + texte, consultable/éditable)
# ============================================================

def init_sqlite(db_path: str) -> sqlite3.Connection:
    db_preexisting = Path(db_path).exists()
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            courant TEXT,
            artiste TEXT,
            periode TEXT,
            technique TEXT,
            mots_cles TEXT,
            confiance_autotag TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            source_file TEXT PRIMARY KEY,
            nb_chunks INTEGER,
            ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Migration : ajoute les colonnes "oeuvre" et "type_contenu" si la base existe déjà
    # depuis une version antérieure (ALTER TABLE ADD COLUMN est idempotent via ce garde-fou,
    # SQLite ne supportant pas "IF NOT EXISTS" sur ADD COLUMN).
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    ingested_cols = {row[1] for row in conn.execute("PRAGMA table_info(ingested_files)").fetchall()}
    needs_backup = db_preexisting and (
        "oeuvre" not in existing_cols or "type_contenu" not in existing_cols
        or "content_hash" not in ingested_cols or "source_path" not in ingested_cols
    )
    if needs_backup:
        # Une sauvegarde est prise avant toute evolution automatique du schema.
        conn.close()
        _migration_backup(db_path)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
        ingested_cols = {row[1] for row in conn.execute("PRAGMA table_info(ingested_files)").fetchall()}
    if "oeuvre" not in existing_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN oeuvre TEXT DEFAULT ''")
    if "type_contenu" not in existing_cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN type_contenu TEXT DEFAULT 'document'")
    if "content_hash" not in ingested_cols:
        conn.execute("ALTER TABLE ingested_files ADD COLUMN content_hash TEXT")
    if "source_path" not in ingested_cols:
        conn.execute("ALTER TABLE ingested_files ADD COLUMN source_path TEXT")
    if "updated_at" not in ingested_cols:
        conn.execute("ALTER TABLE ingested_files ADD COLUMN updated_at TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_file ON chunks(source_file)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        );
    """)

    conn.commit()
    return conn


def save_chunk_sqlite(conn: sqlite3.Connection, chunk: Chunk):
    with _db_lock:
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (id, source_file, chunk_index, text, courant, artiste, periode, technique,
                mots_cles, confiance_autotag, oeuvre, type_contenu)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                chunk.id,
                chunk.source_file,
                chunk.chunk_index,
                chunk.text,
                chunk.metadata.courant,
                chunk.metadata.artiste,
                chunk.metadata.periode,
                chunk.metadata.technique,
                json.dumps(chunk.metadata.mots_cles, ensure_ascii=False),
                chunk.metadata.confiance_autotag,
                chunk.metadata.oeuvre,
                chunk.metadata.type_contenu,
            ),
        )
        conn.commit()


def mark_file_ingested(conn: sqlite3.Connection, source_file: str, nb_chunks: int,
                       content_hash: str = "", source_path: str = ""):
    with _db_lock:
        conn.execute(
            """INSERT INTO ingested_files (source_file, nb_chunks, content_hash, source_path, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(source_file) DO UPDATE SET
                 nb_chunks = excluded.nb_chunks,
                 content_hash = excluded.content_hash,
                 source_path = excluded.source_path,
                 updated_at = CURRENT_TIMESTAMP""",
            (source_file, nb_chunks, content_hash, source_path),
        )
        conn.commit()


def get_ingestion_state(conn: sqlite3.Connection, source_file: str) -> Optional[dict]:
    cur = conn.execute(
        "SELECT nb_chunks, content_hash, source_path FROM ingested_files WHERE source_file = ?",
        (source_file,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {"nb_chunks": row[0], "content_hash": row[1] or "", "source_path": row[2] or ""}


def is_already_ingested(conn: sqlite3.Connection, source_file: str, content_hash: str = "") -> bool:
    state = get_ingestion_state(conn, source_file)
    if state is None:
        return False
    # Les anciennes entrees, sans hash, restent intactes et continuent a etre utilisables.
    return not content_hash or not state["content_hash"] or state["content_hash"] == content_hash


# ============================================================
# STOCKAGE DES CONVERSATIONS (onglet "Discussion")
# ============================================================
#
# Chaque conversation multi-tours est identifiée par un UUID (généré côté GUI au premier
# message) et associée à un titre (dérivé du premier message utilisateur, voir
# make_conversation_title()) ainsi qu'à un horodatage de dernière activité (updated_at),
# utilisé pour trier l'historique du plus récent au plus ancien.

def make_conversation_title(text: str, max_len: int = 60) -> str:
    """Dérive un titre court et lisible à partir du premier message d'une conversation."""
    flat = " ".join(text.split())
    if len(flat) <= max_len:
        return flat or "Conversation sans titre"
    return flat[:max_len].rstrip() + "…"


def create_conversation(conn: sqlite3.Connection, conversation_id: str, title: str):
    with _db_lock:
        conn.execute(
            "INSERT OR REPLACE INTO conversations (id, title) VALUES (?, ?)",
            (conversation_id, title),
        )
        conn.commit()


def add_conversation_message(conn: sqlite3.Connection, conversation_id: str, role: str, content: str):
    with _db_lock:
        conn.execute(
            "INSERT INTO conversation_messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conversation_id,),
        )
        conn.commit()


def list_conversations(conn: sqlite3.Connection) -> list:
    """Retourne les conversations enregistrées, triées de la plus récente à la plus ancienne."""
    cur = conn.execute(
        "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC"
    )
    return cur.fetchall()


def load_conversation(conn: sqlite3.Connection, conversation_id: str) -> list:
    """Recharge l'historique d'une conversation sous la forme attendue par chat_llm()/process_chat_turn()."""
    cur = conn.execute(
        "SELECT role, content FROM conversation_messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,),
    )
    return [{"role": role, "content": content} for role, content in cur.fetchall()]


def delete_conversation(conn: sqlite3.Connection, conversation_id: str):
    with _db_lock:
        conn.execute("DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


def update_conversation_title(conn: sqlite3.Connection, conversation_id: str, title: str):
    with _db_lock:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id),
        )
        conn.commit()


# ============================================================
# STOCKAGE VECTORIEL (ChromaDB)
# ============================================================

def init_chroma():
    if chromadb is None:
        raise RuntimeError("chromadb n'est pas installé : pip install chromadb --break-system-packages")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return collection


def save_chunk_chroma(collection, chunk: Chunk, embedding: list):
    collection.upsert(
        ids=[chunk.id],
        embeddings=[embedding],
        documents=[chunk.text],
        metadatas=[{
            "source_file": chunk.source_file,
            "chunk_index": chunk.chunk_index,
            "courant": chunk.metadata.courant,
            "artiste": chunk.metadata.artiste,
            "periode": chunk.metadata.periode,
            "technique": chunk.metadata.technique,
            "oeuvre": chunk.metadata.oeuvre,
            "type_contenu": chunk.metadata.type_contenu,
            "mots_cles": ", ".join(chunk.metadata.mots_cles),
        }],
    )


def repair_source_vector_index(conn: sqlite3.Connection, collection, source_file: str) -> int:
    """Recree seulement les vecteurs absents de Chroma a partir de SQLite.

    SQLite est ecrit avant Chroma : apres une interruption, cette verification rend
    l'operation idempotente et evite qu'un chunk deja present en SQLite soit perdu du RAG.
    """
    rows = conn.execute(
        """SELECT id, source_file, chunk_index, text, courant, artiste, periode, technique,
                  mots_cles, confiance_autotag, oeuvre, type_contenu
           FROM chunks WHERE source_file = ? ORDER BY chunk_index""",
        (source_file,),
    ).fetchall()
    if not rows:
        return 0
    try:
        existing_ids = set(collection.get(where={"source_file": source_file}, include=[]).get("ids", []))
    except Exception as e:
        print(f"  [!] Verification Chroma impossible pour {source_file} : {e}")
        return 0

    repaired = 0
    for row in rows:
        if row[0] in existing_ids:
            continue
        try:
            keywords = json.loads(row[8]) if row[8] else []
        except json.JSONDecodeError:
            keywords = []
        chunk = Chunk(
            id=row[0], source_file=row[1], chunk_index=row[2], text=row[3],
            metadata=ChunkMetadata(
                courant=row[4] or "", artiste=row[5] or "", periode=row[6] or "",
                technique=row[7] or "", mots_cles=keywords,
                confiance_autotag=row[9] or "", oeuvre=row[10] or "",
                type_contenu=row[11] or "document",
            ),
        )
        embedding = get_embedding(chunk.text)
        if embedding is None:
            continue
        save_chunk_chroma(collection, chunk, embedding)
        repaired += 1
    if repaired:
        print(f"  [REPARATION] {repaired} vecteur(s) Chroma restaure(s) pour {source_file}")
    return repaired


def migrate_matching_legacy_source(conn: sqlite3.Connection, collection, legacy_source: str,
                                   source_file: str, chunk_texts: list, content_hash: str) -> bool:
    """Migre un ancien identifiant base sur le nom seul, uniquement si le texte correspond exactement."""
    if legacy_source == source_file or get_ingestion_state(conn, source_file) is not None:
        return False
    legacy_rows = conn.execute(
        "SELECT id, chunk_index, text FROM chunks WHERE source_file = ? ORDER BY chunk_index",
        (legacy_source,),
    ).fetchall()
    if len(legacy_rows) != len(chunk_texts) or any(row[1] != i or row[2] != chunk_texts[i]
                                                    for i, row in enumerate(legacy_rows)):
        return False
    with _db_lock:
        conn.execute("UPDATE chunks SET source_file = ? WHERE source_file = ?", (source_file, legacy_source))
        conn.execute("DELETE FROM ingested_files WHERE source_file = ?", (legacy_source,))
        conn.execute(
            """INSERT INTO ingested_files (source_file, nb_chunks, content_hash, source_path, updated_at)
               VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (source_file, len(chunk_texts), content_hash, source_file),
        )
        conn.commit()
    try:
        old_ids = collection.get(where={"source_file": legacy_source}, include=[]).get("ids", [])
        if old_ids:
            collection.update(ids=old_ids, metadatas=[{"source_file": source_file}] * len(old_ids))
    except Exception as e:
        print(f"  [!] Migration Chroma partielle ({e}) ; une reparation va etre tentee.")
    print(f"[MIGRATION] Source existante migree : {legacy_source} -> {source_file}")
    repair_source_vector_index(conn, collection, source_file)
    return True


def _format_duration(seconds: float) -> str:
    """Formate une durée en secondes vers un format lisible (ex: 2h 14min, 45s)."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}min"
    if minutes:
        return f"{minutes}min {secs:02d}s"
    return f"{secs}s"


# ============================================================
# PIPELINE PRINCIPAL D'INGESTION
# ============================================================

def ingest_text_block(
    text: str,
    source_name: str,
    sqlite_conn: sqlite3.Connection,
    chroma_collection,
    manual_metadata: Optional[dict] = None,
    autotag: bool = True,
    type_contenu: str = "document",
    skip_indices: Optional[set] = None,
    autotag_granularity: Optional[str] = None,
    document_sample_chars: Optional[int] = None,
) -> tuple:
    """
    Cœur du pipeline : découpe un bloc de texte en chunks, les tague (auto ou manuel),
    les embed et les stocke (SQLite + ChromaDB). Réutilisé par ingest_file() pour les
    documents et par ingest_artwork_analysis() pour les analyses d'œuvres générées
    (vision) ou saisies manuellement — les deux passent par exactement le même pipeline
    de chunking/embedding/stockage, seul le texte source et les métadonnées diffèrent.

    skip_indices : indices de chunks déjà traités lors d'une exécution précédente
    (reprise après interruption) — ces chunks ne déclenchent aucun appel Ollama.

    Retourne (nb_success, nb_chunks) : le nombre de chunks stockés avec succès (y compris
    ceux repris tels quels) et le nombre total de chunks attendus pour ce texte. L'appelant
    doit comparer les deux avant de considérer l'ingestion comme complète : si nb_success <
    nb_chunks, certains chunks ont échoué définitivement (embedding en échec après retries)
    sans que l'utilisateur ait interrompu le programme, et le fichier ne doit PAS être marqué
    comme entièrement ingéré (voir ingest_file / ingest_artwork_analysis) pour permettre de
    retenter uniquement les chunks manquants au prochain lancement.
    Peut lever KeyboardInterrupt (proprement, sans threads zombies) si l'utilisateur
    interrompt en cours de route ; les chunks déjà sauvegardés avant l'interruption
    restent en base pour permettre une reprise ultérieure.
    """
    skip_indices = skip_indices or set()
    chunk_texts = semantic_chunk(text)
    nb_chunks = len(chunk_texts)

    # Granularité effective pour CE fichier : la valeur passée en paramètre (choisie dans le
    # GUI ou en CLI pour ce livre précis) prime sur la globale, qui reste le comportement par défaut.
    effective_granularity = autotag_granularity if autotag_granularity is not None else AUTOTAG_GRANULARITY

    # --- Auto-tagging au niveau document : un seul appel LLM pour tout le fichier ---
    document_metadata = None
    if manual_metadata is None and autotag and effective_granularity == "document":
        sample_chars = document_sample_chars if document_sample_chars is not None else DOCUMENT_AUTOTAG_SAMPLE_CHARS
        print(f"  -> Auto-tagging document (1 appel LLM, échantillon {sample_chars} caractères) "
              f"sur {nb_chunks} chunks...")
        document_metadata = autotag_document(text, sample_chars=sample_chars)
        print(f"  -> courant='{document_metadata.courant}' artiste='{document_metadata.artiste}' "
              f"periode='{document_metadata.periode}' technique='{document_metadata.technique}' "
              f"oeuvre='{document_metadata.oeuvre}'")

    todo_indices = [i for i in range(nb_chunks) if i not in skip_indices]
    if skip_indices:
        print(f"  -> Reprise : {len(skip_indices)}/{nb_chunks} chunks déjà en base, "
              f"{len(todo_indices)} restants à traiter")

    # Auto-tagging par chunk (1 appel LLM par chunk) uniquement quand ni métadonnées manuelles,
    # ni auto-tagging "document" ne s'appliquent. Dans ce cas précis, on sépare le travail en
    # deux passes SÉQUENTIELLES (tout l'autotag LLM, puis tout l'embedding) au lieu d'entrelacer
    # les deux types d'appels chunk par chunk : si le serveur Ollama ne garde qu'un seul modèle
    # chargé en VRAM à la fois, l'entrelacement forcerait un chargement/déchargement du modèle
    # LLM et du modèle d'embedding à CHAQUE chunk. Regrouper les appels par modèle évite ce swap.
    needs_per_chunk_autotag = manual_metadata is None and document_metadata is None and autotag

    def build_static_metadata(index: int) -> ChunkMetadata:
        """Métadonnées ne nécessitant aucun appel réseau (manuel ou document déjà résolu)."""
        if manual_metadata:
            return ChunkMetadata(
                courant=manual_metadata.get("courant", ""),
                artiste=manual_metadata.get("artiste", ""),
                periode=manual_metadata.get("periode", ""),
                technique=manual_metadata.get("technique", ""),
                oeuvre=manual_metadata.get("oeuvre", ""),
                mots_cles=manual_metadata.get("mots_cles", []),
                confiance_autotag=manual_metadata.get("confiance_autotag", "manuel"),
                type_contenu=type_contenu,
            )
        elif document_metadata is not None:
            return ChunkMetadata(
                courant=document_metadata.courant,
                artiste=document_metadata.artiste,
                periode=document_metadata.periode,
                technique=document_metadata.technique,
                oeuvre=document_metadata.oeuvre,
                mots_cles=document_metadata.mots_cles,
                confiance_autotag="auto_document",
                type_contenu=type_contenu,
            )
        return ChunkMetadata(type_contenu=type_contenu)

    print(f"  -> {len(todo_indices)} chunks à traiter (embedding: {EMBED_CONCURRENCY} simultanés, "
          f"autotag LLM: {'1 appel document' if document_metadata is not None else (f'{LLM_CONCURRENCY} simultané(s) par chunk, par lots de {AUTOTAG_EMBED_BATCH_SIZE}' if needs_per_chunk_autotag else 'désactivé')})")

    def process_autotag(index: int):
        metadata = autotag_chunk(chunk_texts[index])
        metadata.type_contenu = type_contenu
        return index, metadata

    # chunk_metadata_by_index est réinitialisé/rempli à chaque lot (voir plus bas) quand
    # needs_per_chunk_autotag est actif ; process_one_chunk lit toujours sa valeur courante.
    chunk_metadata_by_index = {}

    def process_one_chunk(index: int, chunk_text: str):
        """Embedding d'un chunk (seul appel réseau à ce stade) + assemblage du Chunk final."""
        if needs_per_chunk_autotag:
            metadata = chunk_metadata_by_index.get(
                index, ChunkMetadata(type_contenu=type_contenu, confiance_autotag="echec")
            )
        else:
            metadata = build_static_metadata(index)

        embedding = get_embedding(chunk_text)

        chunk = Chunk(
            id=f"{source_name}::{index}",  # déterministe : une reprise réécrit le même id au lieu de dupliquer
            source_file=source_name,
            chunk_index=index,
            text=chunk_text,
            metadata=metadata,
        )
        return index, chunk, embedding

    # --- Découpage en lots ---
    # En autotag "chunk", on alterne par lots de AUTOTAG_EMBED_BATCH_SIZE entre une passe
    # d'auto-tagging LLM et une passe d'embedding+sauvegarde pour ce même lot, avant de passer
    # au suivant. Ça limite le nombre de swaps VRAM (un aller-retour par lot, pas par chunk) tout
    # en sauvegardant progressivement (pas d'attente jusqu'à la fin du document entier).
    # Sinon (métadonnées manuelles ou auto-tagging "document", un seul modèle sollicité en boucle),
    # un seul "lot" = tous les chunks, traités en une seule passe d'embedding comme avant.
    if needs_per_chunk_autotag and todo_indices:
        batches = [
            todo_indices[i:i + AUTOTAG_EMBED_BATCH_SIZE]
            for i in range(0, len(todo_indices), AUTOTAG_EMBED_BATCH_SIZE)
        ]
    else:
        batches = [todo_indices] if todo_indices else []

    nb_success = len(skip_indices)  # les chunks repris comptent déjà comme "réussis"
    nb_done = 0
    nb_todo = len(todo_indices)
    start_time = time.monotonic()

    # Compteurs de progression pour la passe d'auto-tagging (cumulés sur tous les lots, sinon
    # l'estimation de temps restant repartirait de zéro à chaque lot et serait peu fiable).
    nb_tagged = 0
    start_time_tag = time.monotonic()

    for batch_num, batch_indices in enumerate(batches, start=1):
        batch_label = f"lot {batch_num}/{len(batches)}"

        # --- Passe 1/2 du lot : auto-tagging LLM (uniquement si autotag par chunk) ---
        if needs_per_chunk_autotag:
            print(f"  -> {batch_label} - passe 1/2 : auto-tagging de {len(batch_indices)} chunk(s)...")
            chunk_metadata_by_index.clear()
            executor_tag = ThreadPoolExecutor(max_workers=LLM_CONCURRENCY + 1)
            try:
                futures_tag = {executor_tag.submit(process_autotag, i): i for i in batch_indices}
                for future in as_completed(futures_tag):
                    nb_tagged += 1
                    index = futures_tag[future]
                    try:
                        index, metadata = future.result()
                    except Exception as e:
                        print(f"  [X] Autotag chunk {index} : erreur inattendue ({e}), métadonnées vides")
                        metadata = ChunkMetadata(type_contenu=type_contenu, confiance_autotag="echec")
                    chunk_metadata_by_index[index] = metadata

                    # Progression affichée tous les 10 chunks tagués (le LLM 31B est lent,
                    # un intervalle plus court que la passe embedding évite un silence trop long).
                    if nb_tagged % 10 == 0 or nb_tagged == nb_todo:
                        elapsed_tag = time.monotonic() - start_time_tag
                        avg_tag = elapsed_tag / nb_tagged
                        remaining_tag = avg_tag * (nb_todo - nb_tagged)
                        print(f"  --- [autotag {nb_tagged}/{nb_todo}] écoulé: {_format_duration(elapsed_tag)} "
                              f"| moyenne: {avg_tag:.1f}s/chunk | restant estimé (tagging): "
                              f"{_format_duration(remaining_tag)} ---")
            except KeyboardInterrupt:
                print(f"\n  [!] Interruption pendant l'auto-tagging du {batch_label} "
                      f"({nb_success}/{nb_chunks} chunks déjà sauvegardés au total ; "
                      f"ce lot n'est pas encore stocké, il sera retenté au prochain lancement)...")
                executor_tag.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor_tag.shutdown(wait=True)
            print(f"  -> {batch_label} - passe 2/2 : embedding de {len(batch_indices)} chunk(s)...")

        # --- Passe 2/2 du lot (ou passe unique) : embedding + stockage ---
        pool_size = EMBED_CONCURRENCY + 2
        executor = ThreadPoolExecutor(max_workers=pool_size)
        try:
            futures = {
                executor.submit(process_one_chunk, i, chunk_texts[i]): i
                for i in batch_indices
            }

            for future in as_completed(futures):
                nb_done += 1
                try:
                    index, chunk, embedding = future.result()
                except Exception as e:
                    print(f"  [X] Chunk {futures[future]} : erreur inattendue ({e})")
                    continue

                if embedding is None:
                    print(f"  [X] Chunk {index} ignoré (échec embedding après retries) [{nb_done}/{nb_todo}]")
                else:
                    save_chunk_sqlite(sqlite_conn, chunk)
                    save_chunk_chroma(chroma_collection, chunk, embedding)
                    nb_success += 1
                    print(f"  [{nb_done}/{nb_todo}] OK  chunk={index}  courant='{chunk.metadata.courant}' "
                          f"artiste='{chunk.metadata.artiste}' periode='{chunk.metadata.periode}' "
                          f"technique='{chunk.metadata.technique}' oeuvre='{chunk.metadata.oeuvre}'")

                # Estimation de temps restant, affichée tous les 20 chunks (pas à chaque ligne, illisible sinon)
                if nb_done % 20 == 0 or nb_done == nb_todo:
                    elapsed = time.monotonic() - start_time
                    avg_per_chunk = elapsed / nb_done
                    remaining = avg_per_chunk * (nb_todo - nb_done)
                    print(f"  --- [{nb_done}/{nb_todo}] écoulé: {_format_duration(elapsed)} "
                          f"| moyenne: {avg_per_chunk:.1f}s/chunk | restant estimé: {_format_duration(remaining)} ---")
        except KeyboardInterrupt:
            print(f"\n  [!] Interruption demandée : arrêt propre en cours "
                  f"({nb_success}/{nb_chunks} chunks déjà sauvegardés)...")
            # cancel_futures (Python 3.9+) évite de laisser tourner les tâches pas encore démarrées ;
            # celles déjà en vol se terminent normalement (leurs résultats seront simplement ignorés).
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    return nb_success, nb_chunks


def ingest_file(
    path: Path,
    sqlite_conn: sqlite3.Connection,
    chroma_collection,
    manual_metadata: Optional[dict] = None,
    autotag: bool = True,
    force: bool = False,
    autotag_granularity: Optional[str] = None,
    document_sample_chars: Optional[int] = None,
):
    source_name = make_local_source_id(path)
    display_name = path.name

    if force:
        delete_source_file(sqlite_conn, chroma_collection, source_name)
    elif False:  # La verification est faite apres extraction, avec l'empreinte du contenu.
        print(f"[SKIP] {source_name} déjà ingéré (utilise --force pour ré-ingérer)")
        return

    print(f"[...] Traitement de {source_name}")

    try:
        raw_text = extract_text(path)
    except Exception as e:
        print(f"  [X] Échec extraction : {e}")
        return

    text = clean_text(raw_text)
    if not text:
        print(f"  [X] Aucun texte extrait de {source_name}")
        return

    current_hash = content_fingerprint(text)
    chunk_texts = semantic_chunk(text)
    migrate_matching_legacy_source(
        sqlite_conn, chroma_collection, display_name, source_name, chunk_texts, current_hash
    )
    state = get_ingestion_state(sqlite_conn, source_name)
    if not force and state and state["content_hash"] == current_hash:
        repair_source_vector_index(sqlite_conn, chroma_collection, source_name)
        print(f"[SKIP] {display_name} inchangé ; index vérifié (utilise --force pour ré-ingérer)")
        return
    if not force and state:
        print(f"[MISE À JOUR] {display_name} a changé : ancienne version remplacée.")
        delete_source_file(sqlite_conn, chroma_collection, source_name)

    skip_indices = set() if force else get_processed_chunk_indices(sqlite_conn, source_name)

    try:
        nb_success, nb_chunks = ingest_text_block(
            text, source_name, sqlite_conn, chroma_collection,
            manual_metadata=manual_metadata, autotag=autotag, type_contenu="document",
            skip_indices=skip_indices,
            autotag_granularity=autotag_granularity,
            document_sample_chars=document_sample_chars,
        )
    except KeyboardInterrupt:
        print(f"[!] {source_name} : traitement interrompu, {len(skip_indices)}+ chunks conservés en base.\n"
              f"    Relance exactement la même commande pour reprendre là où tu t'es arrêté.")
        raise

    if nb_success == nb_chunks:
        # Tous les chunks sont en base : on peut marquer le fichier comme entièrement traité,
        # is_already_ingested() le fera sauter au prochain lancement (sauf --force).
        repair_source_vector_index(sqlite_conn, chroma_collection, source_name)
        mark_file_ingested(sqlite_conn, source_name, nb_success, current_hash, source_name)
        print(f"[OK] {source_name} : {nb_success}/{nb_chunks} chunks stockés\n")
    else:
        # Échec(s) définitif(s) sur au moins un chunk (embedding en échec après retries),
        # PAS une interruption utilisateur. On ne marque volontairement PAS le fichier comme
        # ingéré : au prochain lancement (même sans --force), get_processed_chunk_indices()
        # retrouvera les chunks déjà en base et seuls les chunks manquants seront retentés.
        print(f"[!] {source_name} : {nb_success}/{nb_chunks} chunks stockés — "
              f"{nb_chunks - nb_success} chunk(s) en échec définitif.\n"
              f"    Fichier NON marqué comme entièrement ingéré : relance la même commande "
              f"(sans --force) pour ne retenter que les chunks manquants.\n")


# ============================================================
# ANALYSE D'ŒUVRE (vision Ollama ou saisie manuelle)
# ============================================================

ARTWORK_ANALYSIS_SYSTEM_PROMPT = """Tu es un expert en histoire de l'art et en analyse technique d'œuvres picturales.
On te montre l'image d'une œuvre. Rédige une analyse détaillée et structurée en français, couvrant :
- Composition et cadrage (lignes de force, équilibre, points focaux)
- Palette chromatique et traitement de la lumière (clair-obscur, contrastes, tons dominants)
- Technique picturale apparente (touche, matière, glacis, empâtement, etc. si identifiable)
- Style et rattachement probable à un courant artistique
- Sujet, symbolique et atmosphère générale
Ne fabrique pas d'informations biographiques ou historiques que l'image ne permet pas de déduire ;
concentre-toi sur ce qui est visuellement observable. Réponds en paragraphes clairs, sans balises JSON."""


def encode_image_base64(path: Path) -> str:
    import base64
    return base64.b64encode(path.read_bytes()).decode("utf-8")


_FILENAME_FORBIDDEN_CHARS = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(text: str, max_len: int = 80) -> str:
    """Nettoie une chaîne pour en faire un nom de fichier valide (Windows/Linux/Mac)."""
    text = (text or "").strip()
    text = _FILENAME_FORBIDDEN_CHARS.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text[:max_len] if text else ""


def build_rename_stem(pattern: str, metadata: "ChunkMetadata", fallback_stem: str) -> str:
    """
    Construit le nouveau nom de fichier (sans extension) à partir d'un pattern du type
    "{artiste} - {oeuvre}" et des métadonnées extraites (vision + auto-tag).
    Retombe sur fallback_stem si les champs utilisés sont tous vides.
    """
    values = {
        "artiste": sanitize_filename(metadata.artiste),
        "oeuvre": sanitize_filename(metadata.oeuvre),
        "courant": sanitize_filename(metadata.courant),
        "periode": sanitize_filename(metadata.periode),
        "technique": sanitize_filename(metadata.technique),
    }
    try:
        stem = pattern.format(**values)
    except (KeyError, IndexError):
        stem = ""
    # nettoyage des séparateurs laissés vides (ex: " - " quand un champ est vide)
    stem = re.sub(r"\s*-\s*-\s*", " - ", stem)
    stem = stem.strip(" -_")
    stem = sanitize_filename(stem, max_len=120)
    return stem or fallback_stem


def rename_artwork_file(image_path: Path, pattern: str, metadata: "ChunkMetadata") -> Path:
    """
    Renomme le fichier image sur disque selon `pattern` et les métadonnées extraites.
    Gère les collisions en ajoutant un suffixe numérique. Retourne le nouveau chemin
    (== image_path inchangé si le renommage échoue ou si le nom est déjà correct).
    """
    stem = build_rename_stem(pattern, metadata, fallback_stem=image_path.stem)
    new_path = image_path.with_name(f"{stem}{image_path.suffix.lower()}")

    if new_path == image_path:
        return image_path

    counter = 2
    while new_path.exists():
        new_path = image_path.with_name(f"{stem} ({counter}){image_path.suffix.lower()}")
        counter += 1

    try:
        image_path.rename(new_path)
        return new_path
    except OSError as e:
        print(f"  [!] Renommage impossible pour {image_path.name} : {e}")
        return image_path


def analyze_artwork_vision(image_path: Path, artiste: str = "", oeuvre: str = "", courant: str = "") -> Optional[str]:
    """
    Envoie une image d'œuvre au modèle de vision Ollama (OLLAMA_VISION_MODEL) et
    récupère une analyse technique/stylistique en texte libre (pas de JSON ici :
    c'est ce texte qui sera lui-même chunké et embeddé, comme un document classique).
    """
    try:
        image_b64 = encode_image_base64(image_path)
    except Exception as e:
        print(f"  [X] Impossible de lire l'image {image_path.name} : {e}")
        return None

    context_hint = ""
    if artiste or oeuvre or courant:
        parts = []
        if artiste:
            parts.append(f"artiste présumé : {artiste}")
        if oeuvre:
            parts.append(f"titre présumé : {oeuvre}")
        if courant:
            parts.append(f"courant présumé : {courant}")
        context_hint = f"\n\nContexte fourni par l'utilisateur ({', '.join(parts)}) — à confirmer ou nuancer si l'image le permet."

    messages = [
        {"role": "system", "content": ARTWORK_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyse cette œuvre.{context_hint}"},
    ]

    try:
        return llm_chat(OLLAMA_VISION_MODEL, messages, images=[image_b64], timeout=OLLAMA_VISION_TIMEOUT)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"  [X] Modèle vision '{OLLAMA_VISION_MODEL}' introuvable (404). "
                  f"Vérifie 'ollama pull {OLLAMA_VISION_MODEL}' ou corrige OLLAMA_VISION_MODEL.")
        else:
            print(f"  [X] Échec de l'analyse vision : {e}")
        return None
    except (requests.RequestException, KeyError) as e:
        print(f"  [X] Échec de l'analyse vision : {e}")
        return None


def ingest_artwork_analysis(
    analysis_text: str,
    sqlite_conn: sqlite3.Connection,
    chroma_collection,
    artiste: str = "",
    oeuvre: str = "",
    courant: str = "",
    periode: str = "",
    technique: str = "",
    autotag: bool = False,
    source_confidence: str = "manuel",
    force: bool = False,
) -> int:
    """
    Ingère un texte d'analyse d'œuvre (généré par vision ou écrit à la main) dans le
    même pipeline que les documents classiques (chunking, embedding, stockage), avec
    type_contenu="analyse_oeuvre" pour pouvoir les distinguer/filtrer lors des recherches.
    autotag=False par défaut car les métadonnées (artiste/œuvre/courant) sont
    généralement déjà connues explicitement pour une analyse d'œuvre ; passe autotag=True
    si tu veux laisser Ollama en déduire d'autres mots-clés en plus des champs fournis.
    """
    text = clean_text(analysis_text)
    if not text:
        print("  [X] Analyse vide, rien à ingérer.")
        return 0

    source_name = f"analyse::{artiste or 'inconnu'}::{oeuvre or uuid.uuid4().hex[:8]}"

    if force:
        delete_source_file(sqlite_conn, chroma_collection, source_name)
    elif is_already_ingested(sqlite_conn, source_name):
        print(f"[SKIP] {source_name} déjà ingéré (utilise force=True pour ré-ingérer)")
        return 0

    manual_metadata = {
        "courant": courant,
        "artiste": artiste,
        "periode": periode,
        "technique": technique,
        "oeuvre": oeuvre,
        "mots_cles": [],
        "confiance_autotag": source_confidence,
    }

    skip_indices = set() if force else get_processed_chunk_indices(sqlite_conn, source_name)

    print(f"[...] Ingestion de l'analyse d'œuvre : {source_name}")
    nb_success, nb_chunks = ingest_text_block(
        text, source_name, sqlite_conn, chroma_collection,
        manual_metadata=manual_metadata, autotag=autotag, type_contenu="analyse_oeuvre",
        skip_indices=skip_indices,
    )

    if nb_success == nb_chunks:
        mark_file_ingested(sqlite_conn, source_name, nb_success)
        print(f"[OK] {source_name} : {nb_success}/{nb_chunks} chunks stockés\n")
    else:
        # Échec définitif sur au moins un chunk : on ne marque pas l'analyse comme ingérée,
        # pour permettre de retenter uniquement les chunks manquants au prochain lancement.
        print(f"[!] {source_name} : {nb_success}/{nb_chunks} chunks stockés — "
              f"{nb_chunks - nb_success} chunk(s) en échec définitif.\n"
              f"    Analyse NON marquée comme entièrement ingérée : relance avec les mêmes "
              f"paramètres pour ne retenter que les chunks manquants.\n")

    return nb_success


# ============================================================
# RECHERCHE WEB (alimentation du RAG)
# ============================================================
#
# Objectif : chercher des sources fiables sur un sujet d'art graphique (courant, artiste,
# technique...) et les faire passer par EXACTEMENT le même pipeline d'ingestion que les
# fichiers locaux (ingest_text_block), avec une étape de revue manuelle entre recherche et
# ingestion (voir l'onglet GUI "Recherche web" : rien n'est ajouté au RAG sans validation
# explicite de l'utilisateur).
#
# Trois sources, complémentaires :
#   - Wikipedia   : fiable, gratuit, sans clé API, très bonne couverture des courants/artistes/
#     techniques. La recherche renvoie des titres ; le texte complet est récupéré via l'API
#     "extracts" de MediaWiki (plaintext déjà propre, aucun scraping HTML nécessaire).
#   - Met Museum  : Open Access API du Metropolitan Museum of Art, gratuite, sans clé. Fournit
#     des métadonnées structurées (artiste, date, technique, médium, provenance...) qu'on
#     transforme directement en texte descriptif, sans avoir besoin d'extraire une page HTML.
#   - ddgs (DuckDuckGo) : recherche web généraliste via la bibliothèque `ddgs`, en local, sans
#     Docker ni clé API. Limitée au français et à l'anglais (une requête par langue, résultats
#     fusionnés). Aucun traitement LLM à la recherche : uniquement les résultats bruts, pour
#     rester rapide. Le texte de chaque page est ensuite extrait via trafilatura, qui isole le
#     contenu utile (retire menus/pubs/boilerplate).
#
# Chaque résultat de recherche est un dict léger :
#   {"source": "wikipedia"/"met_museum"/"ddgs", "title": str, "url": str, "snippet": str}
# Le texte complet n'est récupéré (fetch_web_content) que pour les résultats que
# l'utilisateur choisit de prévisualiser ou d'ingérer — jamais pour tous les résultats
# de recherche d'un coup, pour ne pas multiplier les appels réseau inutilement.

def search_wikipedia(query: str, lang: str = WIKIPEDIA_LANG, max_results: int = 5) -> list:
    """Recherche Wikipedia (API officielle MediaWiki, gratuite, sans clé)."""
    try:
        resp = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "format": "json", "srlimit": max_results,
            },
            headers={"User-Agent": WEB_USER_AGENT},
            timeout=WEB_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  [X] Recherche Wikipedia échouée : {e}")
        return []

    results = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "")
        snippet = html_lib.unescape(re.sub(r"<.*?>", "", item.get("snippet", "")))
        url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        results.append({"source": "wikipedia", "title": title, "url": url, "snippet": snippet, "_lang": lang})
    return results


def search_met_museum(query: str, max_results: int = 5) -> list:
    """
    Recherche dans l'Open Access API du Metropolitan Museum of Art (gratuite, sans clé).
    L'endpoint de recherche ne renvoie que des IDs : on récupère donc le détail de chaque
    objet (limité à max_results) pour obtenir un titre affichable, ce qui a l'avantage de
    fournir déjà toutes les métadonnées nécessaires à l'ingestion sans second appel réseau.
    """
    try:
        resp = requests.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/search",
            params={"q": query, "hasImages": "true"},
            timeout=WEB_SEARCH_TIMEOUT,
        )
        resp.raise_for_status()
        object_ids = (resp.json().get("objectIDs") or [])[:max_results]
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  [X] Recherche Met Museum échouée : {e}")
        return []

    results = []
    for object_id in object_ids:
        try:
            detail = requests.get(
                f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}",
                timeout=WEB_SEARCH_TIMEOUT,
            )
            detail.raise_for_status()
            obj = detail.json()
        except (requests.RequestException, json.JSONDecodeError):
            continue

        title = obj.get("title") or "(sans titre)"
        artist = obj.get("artistDisplayName") or ""
        date = obj.get("objectDate") or ""
        snippet_parts = [p for p in (artist, date, obj.get("medium", "")) if p]
        results.append({
            "source": "met_museum",
            "title": f"{title} — {artist}" if artist else title,
            "url": obj.get("objectURL") or f"https://www.metmuseum.org/art/collection/search/{object_id}",
            "snippet": ", ".join(snippet_parts),
            "_object_data": obj,  # évite un second appel réseau lors de la récupération du contenu
        })
    return results


def download_image_bytes(url: str, timeout: int = WEB_SEARCH_TIMEOUT) -> Optional[bytes]:
    """Télécharge une image depuis une URL et retourne ses octets bruts, ou None en cas d'échec."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        print(f"  [X] Téléchargement d'image échoué ({url}) : {e}")
        return None


def save_image_bytes_to_temp(data: bytes, suffix: str = ".jpg") -> Path:
    """Écrit des octets d'image dans un fichier temporaire et retourne son chemin."""
    tmp_dir = Path(tempfile.gettempdir()) / "art_rag_temp_images"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"
    tmp_path.write_bytes(data)
    return tmp_path


def search_web_ddgs(query: str, max_results: int = 5, languages: tuple = WEB_SEARCH_LANGUAGES) -> list:
    """
    Recherche web généraliste via la bibliothèque `ddgs` (moteur DuckDuckGo), en local, sans
    Docker ni clé API. Contrairement à un scraping HTML artisanal (fragile face aux pages de
    blocage anti-bot), `ddgs` est une bibliothèque maintenue qui gère elle-même les en-têtes
    et les particularités du backend — nettement plus fiable.

    Une requête est lancée par langue demandée (région DDG associée : fr-fr / us-en),
    résultats fusionnés et dédupliqués par URL. Aucun traitement LLM à ce stade : uniquement
    les résultats bruts de recherche (titre, url, extrait), pour rester rapide — le texte
    complet n'est récupéré que pour les résultats prévisualisés ou ingérés (fetch_web_content).
    """
    if DDGS is None:
        print("  [X] Le module 'ddgs' n'est pas installé : pip install ddgs --break-system-packages")
        return []

    seen_urls = set()
    merged = []
    try:
        with DDGS(timeout=WEB_SEARCH_TIMEOUT) as ddgs:
            for lang in languages:
                region = DDGS_REGION_BY_LANG.get(lang)
                if not region:
                    continue
                try:
                    raw_results = list(ddgs.text(query, region=region, max_results=max_results))
                except Exception as e:
                    print(f"  [X] Recherche web ({lang}) échouée : {e}")
                    continue
                for item in raw_results:
                    url = item.get("href") or item.get("url") or ""
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    merged.append({
                        "source": "ddgs",
                        "title": item.get("title") or url,
                        "url": url,
                        "snippet": item.get("body", "") or "",
                    })
    except Exception as e:
        print(f"  [X] Recherche web échouée : {e}")

    return merged


def web_search_all(query: str, sources: tuple = ("wikipedia", "met_museum", "ddgs"),
                    max_results_per_source: int = 5) -> list:
    """
    Lance la recherche sur les sources demandées et fusionne les résultats en dédupliquant
    par URL (une même page peut ressortir de plusieurs sources). L'ordre de `sources`
    détermine l'ordre d'affichage des résultats.
    """
    dispatch = {
        "wikipedia": lambda: search_wikipedia(query, max_results=max_results_per_source),
        "met_museum": lambda: search_met_museum(query, max_results=max_results_per_source),
        "ddgs": lambda: search_web_ddgs(query, max_results=max_results_per_source),
    }
    seen_urls = set()
    merged = []
    for source in sources:
        fn = dispatch.get(source)
        if fn is None:
            continue
        for result in fn():
            if result["url"] in seen_urls:
                continue
            seen_urls.add(result["url"])
            merged.append(result)
    return merged


def _met_object_to_text(obj: dict) -> str:
    """Transforme les métadonnées structurées d'une œuvre Met Museum en texte descriptif exploitable pour le RAG."""
    lines = [obj.get("title") or "(sans titre)"]
    if obj.get("artistDisplayName"):
        bio = f" ({obj['artistDisplayBio']})" if obj.get("artistDisplayBio") else ""
        lines.append(f"Artiste : {obj['artistDisplayName']}{bio}")
    for label, key in [
        ("Date", "objectDate"), ("Culture", "culture"), ("Période", "period"),
        ("Médium / technique", "medium"), ("Dimensions", "dimensions"),
        ("Département", "department"), ("Classification", "classification"),
        ("Provenance / crédit", "creditLine"),
    ]:
        if obj.get(key):
            lines.append(f"{label} : {obj[key]}")
    tag_terms = [t.get("term", "") for t in obj.get("tags") or [] if t.get("term")]
    if tag_terms:
        lines.append(f"Mots-clés : {', '.join(tag_terms)}")
    if obj.get("repository"):
        lines.append(f"Conservé à : {obj['repository']}")
    return "\n".join(lines)


def fetch_web_content(result: dict, timeout: int = WEB_FETCH_TIMEOUT) -> dict:
    """
    Récupère le texte complet d'un résultat de recherche web, selon sa source :
      - wikipedia  : API "extracts" MediaWiki (plaintext propre, pas de scraping HTML)
      - met_museum : métadonnées déjà récupérées lors de la recherche (_object_data),
                     transformées en texte descriptif — aucun appel réseau supplémentaire
      - ddgs (ou toute autre URL générique) : récupération de la page + extraction
                     via trafilatura

    Retourne {"title": str, "text": str, "url": str} ou {"_error": str}.
    """
    source = result.get("source")

    if source == "met_museum":
        obj = result.get("_object_data")
        if not obj:
            return {"_error": "Métadonnées Met Museum manquantes."}
        return {"title": result["title"], "text": _met_object_to_text(obj), "url": result["url"]}

    if source == "wikipedia":
        lang = result.get("_lang", WIKIPEDIA_LANG)
        title = result["title"]
        try:
            resp = requests.get(
                f"https://{lang}.wikipedia.org/w/api.php",
                params={
                    "action": "query", "prop": "extracts", "explaintext": 1,
                    "titles": title, "format": "json",
                },
                headers={"User-Agent": WEB_USER_AGENT},
                timeout=timeout,
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            page = next(iter(pages.values()), {})
            text = page.get("extract", "")
        except (requests.RequestException, json.JSONDecodeError) as e:
            return {"_error": f"Échec de récupération Wikipedia : {e}"}
        if not text:
            return {"_error": "Aucun texte extrait de cette page Wikipedia (page vide ou redirection)."}
        return {"title": title, "text": text, "url": result["url"]}

    # ddgs, ou toute autre URL générique
    if trafilatura is None:
        return {"_error": "Le module 'trafilatura' n'est pas installé : "
                           "pip install trafilatura --break-system-packages"}
    try:
        resp = requests.get(result["url"], headers={"User-Agent": WEB_USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"_error": f"Échec de récupération de la page : {e}"}

    extracted = trafilatura.extract(
        resp.text, url=result["url"], include_comments=False, include_tables=False, favor_recall=True,
    )
    if not extracted:
        return {"_error": "Aucun texte exploitable extrait de cette page (page trop dynamique, ou bloquée)."}

    title = result.get("title") or result["url"]
    try:
        meta = trafilatura.extract_metadata(resp.text, default_url=result["url"])
        if meta and meta.title:
            title = meta.title
    except Exception:
        pass

    return {"title": title, "text": extracted, "url": result["url"]}


def ingest_web_result(result: dict, sqlite_conn: sqlite3.Connection, chroma_collection,
                       autotag_granularity: Optional[str] = None, force: bool = False) -> dict:
    """
    Ingère un résultat de recherche web validé manuellement, en réutilisant EXACTEMENT le
    même pipeline (ingest_text_block) que pour un fichier local. L'URL sert d'identifiant de
    source (source_file) : elle permet à la fois le dédoublonnage (is_already_ingested) et de
    tracer la provenance de chaque extrait dans les réponses du RAG (voir build_context_block,
    qui affiche déjà "source: <source_file>" — donc l'URL directement pour ces chunks).

    Retourne {"status": "ok"/"skip"/"partial"/"error", "message": str, "title": str}.
    """
    url = result["url"]
    title_fallback = result.get("title", url)

    if not force and is_already_ingested(sqlite_conn, url):
        return {"status": "skip", "message": "Déjà ingéré précédemment (coche 'Forcer' pour ré-ingérer).",
                "title": title_fallback}

    content = fetch_web_content(result)
    if "_error" in content:
        return {"status": "error", "message": content["_error"], "title": title_fallback}

    text = clean_text(content["text"])
    if not text:
        return {"status": "error", "message": "Aucun texte exploitable après nettoyage.",
                "title": content.get("title", title_fallback)}

    if force:
        delete_source_file(sqlite_conn, chroma_collection, url)
        skip_indices = set()
    else:
        skip_indices = get_processed_chunk_indices(sqlite_conn, url)

    nb_success, nb_chunks = ingest_text_block(
        text, url, sqlite_conn, chroma_collection,
        autotag=True, type_contenu="web", skip_indices=skip_indices,
        autotag_granularity=autotag_granularity,
    )

    title = content.get("title", title_fallback)
    if nb_success == nb_chunks:
        mark_file_ingested(sqlite_conn, url, nb_success)
        return {"status": "ok", "message": f"{nb_success}/{nb_chunks} chunks stockés", "title": title}
    return {
        "status": "partial",
        "message": f"{nb_success}/{nb_chunks} chunks stockés — {nb_chunks - nb_success} échec(s)",
        "title": title,
    }


def collect_files(args) -> list:
    files = []
    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"Fichier introuvable : {p}")
            sys.exit(1)
        files.append(p)
    if args.dir:
        d = Path(args.dir)
        if not d.exists():
            print(f"Dossier introuvable : {d}")
            sys.exit(1)
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(sorted(d.rglob(f"*{ext}")))
    return files


# ============================================================
# RECHERCHE SÉMANTIQUE (RAG)
# ============================================================

def search_chunks(question: str, top_k: int = TOP_K_DEFAULT, where_filter: dict = None) -> list:
    """
    Cherche les chunks les plus proches sémantiquement de la question.
    where_filter permet de filtrer par métadonnée ChromaDB (ex: {"courant": "Impressionnisme"}).
    Retourne une liste de dicts : {text, metadata, distance}.
    """
    question_embedding = get_embedding(question)
    if question_embedding is None:
        print("[ERREUR] Impossible de générer l'embedding de la question (voir erreurs Ollama ci-dessus)")
        return []

    collection = init_chroma()

    query_kwargs = {
        "query_embeddings": [question_embedding],
        "n_results": top_k,
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})

    return chunks


def build_context_block(chunks: list) -> str:
    """Formate les chunks trouvés en un bloc de contexte lisible pour le LLM."""
    blocks = []
    for i, c in enumerate(chunks):
        meta = c["metadata"]
        source_info = f"(source: {meta.get('source_file', '?')}"
        if meta.get("courant"):
            source_info += f", courant: {meta['courant']}"
        if meta.get("artiste"):
            source_info += f", artiste: {meta['artiste']}"
        if meta.get("oeuvre"):
            source_info += f", œuvre: {meta['oeuvre']}"
        if meta.get("type_contenu") == "analyse_oeuvre":
            source_info += ", analyse d'œuvre"
        source_info += ")"
        blocks.append(f"--- Extrait {i + 1} {source_info} ---\n{c['text']}")
    return "\n\n".join(blocks)


_STATS_FIELDS = (
    "total_duration", "load_duration",
    "prompt_eval_count", "prompt_eval_duration",
    "eval_count", "eval_duration",
)


def _collect_stats(stats: Optional[dict], data: dict):
    """
    Copie les métriques de perf (_STATS_FIELDS) ET done_reason depuis une réponse Ollama (JSON
    complet en mode non-streamé, ou dernier fragment en mode streamé) vers le dict `stats` fourni
    par l'appelant. done_reason vaut "stop" pour une fin normale (le modèle a choisi de s'arrêter)
    ou "length" quand la génération a été coupée parce qu'elle a atteint num_predict ou la limite
    de contexte restante — c'est ce qui permet de détecter une réponse tronquée.
    """
    if stats is None:
        return
    stats.update({k: data.get(k, 0) for k in _STATS_FIELDS})
    stats["done_reason"] = data.get("done_reason", "")


def format_llm_stats(stats: dict) -> str:
    """
    Formate les métriques de performance renvoyées par Ollama (durées en nanosecondes)
    en une ligne lisible : temps de chargement du modèle, débit du prefill (prompt),
    débit de génération. Sert à diagnostiquer où passe le temps :
    - load_duration élevé (> quelques centaines de ms) à chaque requête -> le modèle
      est rechargé depuis le disque à chaque fois (keep_alive pas respecté, VRAM
      insuffisante pour garder plusieurs modèles chargés, ou options qui changent
      entre deux appels et forcent un rechargement du "runner" Ollama).
    - prompt_eval lent -> le contexte (nombre d'extraits, top_k) est trop gros pour
      le prefill ; réduire top_k ou num_ctx aide directement.
    - eval (génération) lent -> limité par la bande passante mémoire du GPU pour ce
      modèle ; à ce stade, seul un modèle plus petit ou une quantization plus agressive
      changera la donne, la quantization actuelle jouant déjà son rôle.

    Ajoute un avertissement explicite si done_reason == "length" : la réponse s'est arrêtée
    parce qu'elle a atteint OLLAMA_NUM_PREDICT ou la place restante dans OLLAMA_NUM_CTX, pas
    parce que le modèle avait fini — c'est la signature d'une réponse tronquée.
    """
    if not stats:
        return ""
    load_s = stats.get("load_duration", 0) / 1e9
    prompt_n = stats.get("prompt_eval_count", 0)
    prompt_s = stats.get("prompt_eval_duration", 0) / 1e9
    eval_n = stats.get("eval_count", 0)
    eval_s = stats.get("eval_duration", 0) / 1e9
    total_s = stats.get("total_duration", 0) / 1e9

    prompt_rate = f"{prompt_n / prompt_s:.0f} tok/s" if prompt_s > 0 else "n/a"
    eval_rate = f"{eval_n / eval_s:.1f} tok/s" if eval_s > 0 else "n/a"

    line = (
        f"[perf] chargement modèle: {load_s:.2f}s | "
        f"prompt: {prompt_n} tokens @ {prompt_rate} | "
        f"génération: {eval_n} tokens @ {eval_rate} | "
        f"total: {total_s:.2f}s"
    )

    if stats.get("done_reason") == "length":
        line += (
            f"\n[!] Réponse tronquée : la génération a atteint la limite de tokens "
            f"(OLLAMA_NUM_PREDICT={OLLAMA_NUM_PREDICT} ou contexte OLLAMA_NUM_CTX={OLLAMA_NUM_CTX} saturé), "
            f"pas une fin naturelle. Renvoie 'continue' pour la suite, ou augmente OLLAMA_NUM_CTX / "
            f"réduis MAX_CHAT_HISTORY_MESSAGES / TOP_K_DEFAULT."
        )

    return line


def ask_llm(question: str, chunks: list, on_token=None, stats: Optional[dict] = None) -> str:
    """
    Envoie la question + les extraits au LLM de chat et retourne la réponse complète.

    on_token : callback optionnel appelé avec chaque fragment de texte au fur et à mesure
    de la génération (stream=True côté Ollama). Utile pour afficher la réponse en direct
    (CLI ou GUI) au lieu d'attendre la génération complète — ça ne réduit pas le temps de
    calcul total mais réduit fortement la latence perçue. Si on_token est None, la requête
    reste non-streamée (comportement simple, un seul bloc de texte retourné).

    stats : dict optionnel, rempli en place avec les métriques de performance renvoyées
    par Ollama (voir format_llm_stats) si fourni. Utile pour diagnostiquer un goulot
    d'étranglement (chargement du modèle / prefill / génération).
    """
    if not chunks:
        return "Aucun extrait pertinent trouvé dans la base de connaissances pour cette question."

    context_block = build_context_block(chunks)
    user_prompt = f"Extraits de documents :\n\n{context_block}\n\nQuestion : {question}"

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        return llm_chat(
            OLLAMA_ANSWER_MODEL, messages,
            stream=on_token is not None, timeout=OLLAMA_CHAT_TIMEOUT,
            num_predict=OLLAMA_NUM_PREDICT, on_token=on_token, stats=stats,
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return (f"[ERREUR] Modèle '{OLLAMA_ANSWER_MODEL}' introuvable (404). "
                    f"Vérifie 'ollama pull {OLLAMA_ANSWER_MODEL}' ou corrige OLLAMA_ANSWER_MODEL dans art_rag.py")
        return f"[ERREUR] Échec de la génération : {e}"
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        return f"[ERREUR] Échec de la génération : {e}"


def build_where_filter(courant: str = "", artiste: str = "", oeuvre: str = "", analyses_only: bool = False):
    """Construit un filtre ChromaDB $and à partir de conditions optionnelles, utilisé par le CLI et le GUI."""
    conditions = {}
    if courant:
        conditions["courant"] = courant
    if artiste:
        conditions["artiste"] = artiste
    if oeuvre:
        conditions["oeuvre"] = oeuvre
    if analyses_only:
        conditions["type_contenu"] = "analyse_oeuvre"
    if len(conditions) == 1:
        return conditions
    elif len(conditions) > 1:
        return {"$and": [{k: v} for k, v in conditions.items()]}
    return None


# ============================================================
# CONVERSATION : routeur RAG + chat multi-tours + génération de prompt image
# ============================================================
#
# Le principe : la "recherche RAG" n'est plus une action ponctuelle isolée (question -> extraits ->
# réponse en un coup), mais une brique au service d'une vraie conversation avec un professionnel des
# arts graphiques (ANSWER_SYSTEM_PROMPT ci-dessus). À chaque tour :
#   1. route_message()        décide si le sujet relève de l'art (et donc si le RAG doit être
#                              systématiquement interrogé) et si l'utilisateur veut un prompt image.
#   2. search_chunks()        (déjà existant plus haut) va chercher les extraits pertinents si besoin.
#   3. chat_llm()              ou generate_image_prompt() produit la réponse, avec ou sans les extraits.
#
# process_chat_turn() est le point d'entrée unique qui enchaîne ces 3 étapes ; CLI (`chat`) et GUI
# (onglet "Discussion") appellent tous les deux cette même fonction.

# Schéma imposé pour tout prompt destiné à une IA générative d'image : exactement ces 10 clés,
# dans cet ordre, chaque valeur étant une prose anglaise dense et sensorielle (pas une liste de mots-clés).
IMAGE_PROMPT_JSON_KEYS = [
    "SUBJECT", "POSE", "ENVIRONMENT", "STYLE", "COLOUR PALETTE",
    "LIGHT", "MATERIAL", "ATMOSPHERE", "CAMERA", "RENDER",
]

ROUTER_SYSTEM_PROMPT = """Tu es un classificateur rapide qui analyse le dernier message d'une conversation
avec un professionnel des arts graphiques, à la lumière du contexte précédent fourni.
Réponds STRICTEMENT en JSON valide, sans aucun texte avant ou après, selon ce schéma exact :

{
  "art_related": true ou false,
  "wants_image_prompt": true ou false,
  "search_query": "reformulation concise (une phrase, en français) de ce qu'il faut chercher dans une base documentaire sur l'art, chaîne vide si art_related est false"
}

art_related = true si le dernier message (compte tenu du contexte) concerne l'art, l'histoire de l'art, un
courant, une technique picturale, un artiste, une œuvre, la composition, la lumière, la matière, l'esthétique
visuelle en général, ou la création d'un visuel / d'un prompt d'image.
wants_image_prompt = true si l'utilisateur demande explicitement (ou de façon évidente) un prompt ou une
description destinée à une IA générative d'image, ou quelque chose comme "crée-moi une image", "génère un
visuel", "fais-moi un prompt pour...".
Ne mets jamais wants_image_prompt à true si art_related est false.
Ne mets JAMAIS de texte, d'explication ou de balise markdown autour du JSON."""

IMAGE_PROMPT_SYSTEM_PROMPT = """You are a professional visual prompt architect working for a generative
image AI pipeline. You translate an artistic concept - grounded in real art history, painting technique and
visual culture - into a single structured JSON prompt.

Use the reference material (excerpts from an art-history knowledge base) provided below, together with the
conversation with the user, to settle on a coherent visual concept: subject, artistic influences, material
qualities, light, atmosphere, staging, camera framing and rendering style. Do not merely copy the reference
text - synthesize it into an original, vivid visual description that reflects the user's intent.

You must return ONLY a single valid JSON object, with no markdown code fences, no preamble, no commentary
before or after it.

The JSON must contain exactly the following ten keys, in this order: "SUBJECT", "POSE", "ENVIRONMENT", "STYLE",
"COLOUR PALETTE", "LIGHT", "MATERIAL", "ATMOSPHERE", "CAMERA", "RENDER". Every value is a string written in dense,
vivid, sensory English prose - not a technical checklist, not a bare list of keywords, but a crafted phrase or
short set of phrases that reads like a visual novelist describing what the lens sees. Values should stay concise
(roughly one to two sentences each). The "COLOUR PALETTE" value should name the dominant hues and their
relationships (e.g. contrast, harmony, temperature) as a vivid description, not a bare list of colour names."""


def _strip_json_fences(text: str) -> str:
    """Retire d'éventuelles balises ```json ... ``` autour d'une réponse LLM."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str:
    """
    Isole le premier objet JSON complet dans un texte, au cas où le LLM aurait malgré tout
    ajouté du texte parasite avant/après (même avec format="json", certains modèles dérivent).
    """
    text = _strip_json_fences(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def build_conversation_summary(history: list, max_turns: int = 12) -> str:
    """Formate les derniers tours de conversation (liste de {"role", "content"}) en texte brut."""
    recent = history[-max_turns:] if max_turns else history
    lines = []
    for turn in recent:
        speaker = "Utilisateur" if turn.get("role") == "user" else "Expert"
        lines.append(f"{speaker}: {turn.get('content', '')}")
    return "\n".join(lines)


# Lignes de liste à puces (-, *, •) ou numérotées (1. / 1)) en début de ligne.
_LIST_ITEM_PATTERN = re.compile(r'^[ \t]*(?:[-*•]|\d+[.)])[ \t]+(.+)$', re.MULTILINE)
# Formatage markdown gras/italique à retirer pour obtenir une requête de recherche propre.
_MD_EMPHASIS_PATTERN = re.compile(r'\*\*(.+?)\*\*|\*(.+?)\*|__(.+?)__|_(.+?)_')


def extract_list_items(text: str, max_items: int = 25) -> list:
    """
    Extrait les éléments d'une liste à puces/numérotée présente dans une réponse du chat (ex :
    une liste de tableaux, d'artistes, de techniques...), pour les proposer ensuite en un clic
    comme requêtes de recherche web (onglet "Discussion" -> "Éléments détectés").

    Extraction purement par expression régulière, SANS appel LLM supplémentaire : repère les
    lignes commençant par -, *, • ou un numéro suivi de '.' ou ')', puis retire le formatage
    markdown (gras/italique) pour obtenir un texte de requête propre. Ne renvoie rien si moins
    de deux lignes de ce type sont trouvées (une seule ligne isolée n'est pas vraiment une
    liste, et éviter le bruit d'une fausse détection sur une réponse purement conversationnelle).
    """
    items = []
    seen = set()
    for match in _LIST_ITEM_PATTERN.finditer(text):
        raw = match.group(1).strip()
        clean = _MD_EMPHASIS_PATTERN.sub(
            lambda m: next(g for g in m.groups() if g is not None), raw
        )
        clean = clean.strip(" :-–—").strip()
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        items.append(clean)
        if len(items) >= max_items:
            break
    return items if len(items) >= 2 else []


TITLE_SYSTEM_PROMPT = """Tu es un générateur de titres d'onglets pour une messagerie. Tu reçois soit un message
isolé, soit un historique de conversation (lignes "Utilisateur: ..." / "Expert: ..."), portant sur l'art,
l'histoire de l'art ou des sujets techniques de génération d'images.

Ta tâche : identifier le THÈME central réellement discuté (une technique, un courant artistique, une œuvre,
un projet en cours, un bug technique...) et le NOMMER en 3 à 6 mots en français. C'est une synthèse, pas une
citation : il est INTERDIT de recopier ou de reformuler légèrement les mots du message d'entrée. Mets-toi à la
place de quelqu'un qui n'a pas lu la conversation et doit la reconnaître d'un coup d'œil dans une liste.

Mauvais titre (copie déguisée) : le message dit "Parle-moi du clair-obscur chez le Caravage" -> mauvais titre
"Parle-moi du clair-obscur chez Caravage". Bon titre pour ce même message : "Clair-obscur caravagesque".

Autres bons exemples : "Courants picturaux du baroque", "Prompt image ambiance biomécanique", "Débogage
OLLAMA_NUM_CTX art_rag".

Réponds STRICTEMENT avec le titre lui-même : pas de guillemets, pas de point final, pas de préambule
("Voici un titre :"...), pas de markdown, une seule ligne."""


def generate_conversation_title(text: str, fallback: Optional[str] = None) -> str:
    """
    Génère un titre synthétique (pas une copie) pour une conversation, via OLLAMA_AUTOTAG_MODEL
    (modèle léger, même choix que route_message : une tâche de résumé simple ne justifie pas le
    gros modèle de conversation). `text` peut être un message isolé ou un résumé multi-tours
    produit par build_conversation_summary().

    En cas d'échec (Ollama down, réponse vide, ou réponse qui se contente visiblement de recopier
    le texte reçu), retombe sur `fallback` si fourni, sinon sur une troncature brute du texte
    (make_conversation_title), pour ne jamais laisser une conversation sans titre.
    """
    fallback_title = fallback if fallback is not None else make_conversation_title(text)

    if not text.strip():
        return fallback_title

    messages = [
        {"role": "system", "content": TITLE_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    try:
        content = llm_chat(OLLAMA_AUTOTAG_MODEL, messages, timeout=OLLAMA_CHAT_TIMEOUT, num_predict=32).strip()
        content = content.strip("\"'« »").strip().rstrip(".")
        content = content.splitlines()[0].strip() if content else ""
        if not content:
            return fallback_title
        # Filet de sécurité : si le modèle a quand même recopié le texte d'entrée quasi tel quel
        # (cas observé avec certains petits modèles sur un message isolé et court), on retombe sur
        # le fallback plutôt que d'afficher une "synthèse" qui n'en est pas une.
        normalize = lambda s: re.sub(r"\s+", " ", s.strip().lower())
        if normalize(content) == normalize(text):
            return fallback_title
        return make_conversation_title(content, max_len=60)
    except (requests.RequestException, KeyError, IndexError):
        return fallback_title


def generate_conversation_title_from_history(history: list, fallback: Optional[str] = None) -> str:
    """
    Génère un titre à partir de l'ENSEMBLE de la conversation (et non du seul premier message),
    en s'appuyant sur build_conversation_summary(). Utilisé par le bouton "Régénérer le titre"
    de la fenêtre Historique, une fois que la conversation a suffisamment avancé pour que le
    premier message ne reflète plus forcément son sujet réel.
    """
    summary = build_conversation_summary(history, max_turns=20)
    return generate_conversation_title(summary, fallback=fallback)


def route_message(history: list) -> dict:
    """
    Classifie le dernier message de la conversation : sujet artistique (-> RAG systématique) et
    demande explicite de prompt image. Utilise OLLAMA_AUTOTAG_MODEL (modèle léger) car c'est une
    tâche de classification simple, pas un exercice de raisonnement fin — pas besoin du gros modèle
    de conversation pour trancher ça à chaque tour.

    En cas d'échec (Ollama down, JSON invalide...), on retombe par prudence sur art_related=True :
    une recherche RAG inutile coûte moins cher qu'une réponse non ancrée sur un vrai sujet d'art.
    """
    last_message = history[-1]["content"] if history else ""
    default = {"art_related": True, "wants_image_prompt": False, "search_query": last_message}

    if not last_message.strip():
        return default

    conversation_summary = build_conversation_summary(history, max_turns=6)

    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": conversation_summary},
    ]

    try:
        content = llm_chat(OLLAMA_AUTOTAG_MODEL, messages, json_mode=True, timeout=OLLAMA_CHAT_TIMEOUT)
        data = json.loads(_extract_json_object(content))
        art_related = bool(data.get("art_related", True))
        return {
            "art_related": art_related,
            "wants_image_prompt": bool(data.get("wants_image_prompt", False)) and art_related,
            "search_query": (data.get("search_query") or "").strip() or last_message,
        }
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return default


def _trim_history_for_llm(history: list, max_messages: int = MAX_CHAT_HISTORY_MESSAGES) -> list:
    """
    Restreint l'historique effectivement envoyé au LLM aux `max_messages` derniers tours, pour
    éviter qu'une conversation qui s'allonge finisse par saturer OLLAMA_NUM_CTX (cause la plus
    fréquente de réponses tronquées après plusieurs échanges). L'historique complet continue
    d'être affiché à l'utilisateur (GUI/CLI) ; seule la fenêtre envoyée au modèle est réduite.
    """
    if max_messages and len(history) > max_messages:
        return history[-max_messages:]
    return history


def chat_llm(history: list, context_block: str = "", on_token=None, stats: Optional[dict] = None) -> str:
    """
    Envoie l'historique de la conversation (tronqué à MAX_CHAT_HISTORY_MESSAGES tours les plus
    récents, voir _trim_history_for_llm) au LLM de chat (persona professionnel des arts graphiques
    défini par ANSWER_SYSTEM_PROMPT) et retourne la réponse complète.

    context_block : extraits RAG pour le tour en cours (peut être vide). Injecté juste avant le
    dernier message utilisateur, pour ne pas polluer l'historique tel qu'affiché à l'utilisateur
    (les tours précédents de `history` restent inchangés, sans leurs propres extraits).

    on_token / stats : mêmes conventions que ask_llm (streaming optionnel, métriques de perf ;
    stats["done_reason"] == "length" signale une réponse tronquée par manque de place).
    """
    if not history:
        return "Aucun message à traiter."

    trimmed = _trim_history_for_llm(history)

    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    if len(trimmed) > 1:
        messages.extend(trimmed[:-1])

    last_user_content = trimmed[-1]["content"]
    if context_block:
        last_user_content = (
            f"[Extraits pertinents de la base de connaissances pour ce tour]\n{context_block}\n\n---\n\n"
            f"{last_user_content}"
        )
    messages.append({"role": "user", "content": last_user_content})

    try:
        return llm_chat(
            OLLAMA_ANSWER_MODEL, messages,
            stream=on_token is not None, timeout=OLLAMA_CHAT_TIMEOUT,
            num_predict=OLLAMA_NUM_PREDICT, on_token=on_token, stats=stats,
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return (f"[ERREUR] Modèle '{OLLAMA_ANSWER_MODEL}' introuvable (404). "
                    f"Vérifie 'ollama pull {OLLAMA_ANSWER_MODEL}' ou corrige OLLAMA_ANSWER_MODEL dans art_rag.py")
        return f"[ERREUR] Échec de la génération : {e}"
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        return f"[ERREUR] Échec de la génération : {e}"


def generate_image_prompt(history: list, chunks: list, stats: Optional[dict] = None) -> dict:
    """
    Génère un prompt JSON structuré (les 10 clés d'IMAGE_PROMPT_JSON_KEYS) pour IA générative
    d'image, en s'appuyant sur :
      - la conversation en cours (intention/sujet exprimés par l'utilisateur)
      - les extraits RAG fournis (pour ancrer style/matière/technique dans une réalité artistique
        plutôt que dans des généralités)

    Retourne un dict avec exactement les 10 clés (valeurs = chaînes), ou {"_error": "..."} en cas
    d'échec réseau/modèle. Une clé "_warning" optionnelle signale des valeurs restées vides malgré
    format="json" (le LLM a bien répondu mais a laissé un ou plusieurs champs incomplets).
    """
    conversation_summary = build_conversation_summary(history)
    context_block = (
        build_context_block(chunks) if chunks
        else "(no relevant excerpt found in the knowledge base for this request)"
    )

    user_content = (
        f"Conversation with the user so far:\n{conversation_summary}\n\n"
        f"Reference material from the art knowledge base:\n{context_block}\n\n"
        f"Now produce the JSON image-generation prompt as instructed, reflecting what the user is "
        f"asking for in the conversation above."
    )

    messages = [
        {"role": "system", "content": IMAGE_PROMPT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = llm_chat(
            OLLAMA_ANSWER_MODEL, messages, json_mode=True, timeout=OLLAMA_CHAT_TIMEOUT,
            num_predict=OLLAMA_NUM_PREDICT, stats=stats,
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return {"_error": f"Modèle '{OLLAMA_ANSWER_MODEL}' introuvable (404). "
                               f"Vérifie 'ollama pull {OLLAMA_ANSWER_MODEL}'."}
        return {"_error": f"Échec de la génération : {e}"}
    except (requests.RequestException, KeyError) as e:
        return {"_error": f"Échec de la génération : {e}"}

    try:
        parsed = json.loads(_extract_json_object(raw))
    except json.JSONDecodeError:
        hint = ""
        if stats is not None and stats.get("done_reason") == "length":
            hint = (f" (probablement tronqué : augmente OLLAMA_NUM_PREDICT={OLLAMA_NUM_PREDICT} "
                    f"ou OLLAMA_NUM_CTX={OLLAMA_NUM_CTX})")
        return {"_error": f"Réponse du LLM non-JSON malgré format='json'{hint} : {raw[:300]}"}

    if not isinstance(parsed, dict):
        return {"_error": f"Réponse JSON inattendue (pas un objet) : {raw[:300]}"}

    result = {}
    for key in IMAGE_PROMPT_JSON_KEYS:
        value = parsed.get(key, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        result[key] = str(value).strip()

    missing = [k for k in IMAGE_PROMPT_JSON_KEYS if not result[k]]
    if missing:
        result["_warning"] = f"Clé(s) vide(s) renvoyée(s) par le LLM : {', '.join(missing)}"

    return result


def format_image_prompt_json(prompt_dict: dict) -> str:
    """Formate le dict de prompt (10 clés, sans les clés internes _error/_warning) en JSON lisible et ordonné."""
    clean = {k: prompt_dict.get(k, "") for k in IMAGE_PROMPT_JSON_KEYS}
    return json.dumps(clean, ensure_ascii=False, indent=2)


def process_chat_turn(history: list, top_k: int = TOP_K_DEFAULT, force_image_prompt: bool = False,
                       on_token=None, stats: Optional[dict] = None) -> dict:
    """
    Point d'entrée unique pour traiter un tour de conversation (history[-1] est le message
    utilisateur courant). Utilisé à la fois par le CLI (`chat`) et par le GUI (onglet "Discussion").

    Enchaîne : route_message() -> search_chunks() si besoin -> chat_llm() ou generate_image_prompt().

    force_image_prompt : permet de forcer la génération d'un prompt JSON même si le routeur ne l'a
    pas détecté (utile pour un bouton dédié dans le GUI plutôt que de compter uniquement sur la
    détection automatique de l'intention).

    Retourne :
        {"kind": "text", "content": str, "chunks": list, "route": dict}
        ou
        {"kind": "image_prompt", "content": dict (10 clés + éventuel _error/_warning), "chunks": list, "route": dict}
    """
    route = route_message(history)
    if force_image_prompt:
        route["wants_image_prompt"] = True
        route["art_related"] = True

    chunks = []
    if route["art_related"]:
        chunks = search_chunks(route["search_query"] or history[-1]["content"], top_k=top_k)

    if route["wants_image_prompt"]:
        result = generate_image_prompt(history, chunks, stats=stats)
        return {"kind": "image_prompt", "content": result, "chunks": chunks, "route": route}

    context_block = build_context_block(chunks) if chunks else ""
    answer = chat_llm(history, context_block=context_block, on_token=on_token, stats=stats)
    return {"kind": "text", "content": answer, "chunks": chunks, "route": route}


# ============================================================
# INTERFACE GRAPHIQUE (CustomTkinter) — chargée à la demande
# ============================================================
#
# customtkinter n'est requis que pour le mode GUI ; les commandes CLI
# `ingest` et `query` fonctionnent sans cette dépendance installée.

def launch_gui():
    try:
        import customtkinter as ctk
        import tkinter.filedialog as filedialog
    except ImportError:
        print("[ERREUR] customtkinter n'est pas installé : pip install customtkinter --break-system-packages")
        sys.exit(1)

    if Image is None:
        print(
            f"[ATTENTION] Pillow (PIL) n'a pas pu être importé — détail : {PIL_IMPORT_ERROR}\n"
            "            L'aperçu visuel des œuvres dans l'onglet 'Analyse d'œuvre' sera indisponible.\n"
            "            Installe-le dans le MÊME environnement Python que celui utilisé pour lancer ce "
            "script : pip install pillow --break-system-packages"
        )

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    WINDOW_TITLE = "Art RAG — Ingestion & Search"
    WINDOW_SIZE = "900x720"

    FONT_TITLE = ("Segoe UI", 20, "bold")
    FONT_LABEL = ("Segoe UI", 16)
    FONT_LOG = ("Consolas", 13)

    class QueueWriter(io.TextIOBase):
        """Redirige les print() du pipeline d'ingestion vers une queue thread-safe."""

        def __init__(self, log_queue: queue.Queue):
            self.log_queue = log_queue

        def write(self, text):
            if text.strip():
                self.log_queue.put(text)
            return len(text)

        def flush(self):
            pass

    class ArtRagIngestApp(ctk.CTk):
        def __init__(self):
            super().__init__()

            self.title(WINDOW_TITLE)
            self.geometry(WINDOW_SIZE)
            self.minsize(800, 600)

            self.selected_files: list = []
            self.selected_image = None
            self.log_queue: queue.Queue = queue.Queue()
            self.result_queue: queue.Queue = queue.Queue()  # fragments de réponse RAG streamés depuis le thread worker
            self.worker_thread: Optional[threading.Thread] = None
            self.is_running = False
            self.active_job: Optional[str] = None
            self._last_analysis_text: Optional[str] = None
            self.chat_history: list = []  # liste de {"role": "user"/"assistant", "content": str}
            self.chat_list_items: list = []  # éléments de liste détectés dans la dernière réponse du chat
            self.history_conn = init_sqlite(DB_PATH)  # connexion dédiée à l'historique des conversations
            self.current_conversation_id: Optional[str] = None  # None tant qu'aucun message n'a été envoyé
            self.web_results: list = []          # derniers résultats de recherche web (dicts)
            self.web_result_vars: list = []      # BooleanVar parallèle à web_results (sélection pour ingestion)

            # --- Analyse d'œuvre en lot (répertoire d'images) ---
            self.batch_image_files: list = []                     # images trouvées dans le répertoire choisi
            self.batch_auto_ingest_var = ctk.BooleanVar(value=False)
            self.batch_auto_rename_var = ctk.BooleanVar(value=False)
            self.batch_review_items: list = []   # analyses générées en attente de validation manuelle (auto-ingest=off)

            self._build_ui()
            self._poll_log_queue()

        # --------------------------------------------------------
        # CONSTRUCTION DE L'INTERFACE
        # --------------------------------------------------------

        def _build_ui(self):
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=1)

            # --- En-tête ---
            header = ctk.CTkLabel(self, text="Art RAG — Ingestion & Search", font=FONT_TITLE)
            header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

            # --- Onglets : Documents / Analyse d'œuvre / Recherche RAG ---
            self.tabview = ctk.CTkTabview(self)
            self.tabview.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
            tab_documents = self.tabview.add("Documents")
            tab_analyse = self.tabview.add("Artwork Analysis")
            tab_recherche = self.tabview.add("Chat")
            tab_websearch = self.tabview.add("Web Search")
            tab_parametres = self.tabview.add("Settings")
            tab_documents.grid_columnconfigure(0, weight=1)
            tab_analyse.grid_columnconfigure(0, weight=1)
            tab_recherche.grid_columnconfigure(0, weight=1)
            tab_parametres.grid_columnconfigure(0, weight=1)

            self._build_documents_tab(tab_documents)
            self._build_analysis_tab(tab_analyse)
            self._build_recherche_tab(tab_recherche)
            self._build_websearch_tab(tab_websearch)
            self._build_settings_tab(tab_parametres)

            # --- Zone de log (partagée entre les onglets) ---
            log_frame = ctk.CTkFrame(self)
            log_frame.grid(row=2, column=0, padx=20, pady=(2, 5), sticky="we")
            log_frame.grid_columnconfigure(0, weight=1)
            log_frame.grid_rowconfigure(1, weight=1)

            ctk.CTkLabel(log_frame, text="Execution log", font=FONT_LABEL).grid(
                row=0, column=0, padx=15, pady=(10, 5), sticky="w"
            )

            self.log_box = ctk.CTkTextbox(log_frame, font=FONT_LOG, height=120)
            self.log_box.grid(row=1, column=0, padx=15, pady=(0, 5), sticky="nsew")
            self.log_box.configure(state="disabled")

            # --- Barre de progression partagée ---
            action_frame = ctk.CTkFrame(self, fg_color="transparent")
            action_frame.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")
            action_frame.grid_columnconfigure(0, weight=1)

            self.progress_bar = ctk.CTkProgressBar(action_frame)
            self.progress_bar.grid(row=0, column=0, sticky="ew")
            self.progress_bar.set(0)
            self.progress_bar.configure(mode="indeterminate")

            self.status_label = ctk.CTkLabel(action_frame, text="Ready.", font=FONT_LABEL, text_color="gray70")
            self.status_label.grid(row=1, column=0, pady=(8, 0), sticky="w")

        def _build_documents_tab(self, parent):
            parent.grid_rowconfigure(2, weight=1)

            # --- Section sélection de fichiers ---
            selection_frame = ctk.CTkFrame(parent)
            selection_frame.grid(row=0, column=0, padx=0, pady=(10, 10), sticky="ew")
            selection_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(selection_frame, text="Files / folder to ingest", font=FONT_LABEL).grid(
                row=0, column=0, columnspan=3, padx=15, pady=(15, 5), sticky="w"
            )

            self.files_display = ctk.CTkTextbox(selection_frame, height=80, font=FONT_LOG)
            self.files_display.grid(row=1, column=0, columnspan=3, padx=15, pady=5, sticky="ew")
            self.files_display.configure(state="disabled")

            btn_add_files = ctk.CTkButton(selection_frame, text="Add files", command=self._pick_files)
            btn_add_files.grid(row=2, column=0, padx=(15, 5), pady=(5, 15), sticky="ew")

            btn_add_dir = ctk.CTkButton(selection_frame, text="Add a folder", command=self._pick_directory)
            btn_add_dir.grid(row=2, column=1, padx=5, pady=(5, 15), sticky="ew")

            btn_clear = ctk.CTkButton(
                selection_frame, text="Clear list", fg_color="#8B2635", hover_color="#6b1d29",
                command=self._clear_files
            )
            btn_clear.grid(row=2, column=2, padx=(5, 15), pady=(5, 15), sticky="ew")

            # --- Section métadonnées manuelles (optionnel) ---
            meta_frame = ctk.CTkFrame(parent)
            meta_frame.grid(row=1, column=0, padx=0, pady=10, sticky="ew")
            for i in range(4):
                meta_frame.grid_columnconfigure(i, weight=1)

            ctk.CTkLabel(
                meta_frame,
                text="Manual metadata (optional — leave blank to let auto-tagging decide)",
                font=FONT_LABEL,
            ).grid(row=0, column=0, columnspan=4, padx=15, pady=(15, 10), sticky="w")

            self.entry_courant = self._labeled_entry(meta_frame, "Art movement", row=1, col=0)
            self.entry_artiste = self._labeled_entry(meta_frame, "Artist", row=1, col=1)
            self.entry_periode = self._labeled_entry(meta_frame, "Period", row=1, col=2)
            self.entry_technique = self._labeled_entry(meta_frame, "Technique", row=1, col=3)

            # --- Réglages d'auto-tagging pour CE lancement (adaptés au livre/texte en cours) ---
            autotag_settings_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
            autotag_settings_row.grid(row=2, column=0, columnspan=4, padx=15, pady=(5, 0), sticky="ew")

            granularity_container = ctk.CTkFrame(autotag_settings_row, fg_color="transparent")
            granularity_container.pack(side="left", padx=(0, 20))
            ctk.CTkLabel(
                granularity_container, text="Auto-tagging granularity",
                font=("Segoe UI", 10), text_color="gray70",
            ).pack(anchor="w")
            self.GRANULARITY_LABELS = {
                "Document (1 LLM call, fast)": "document",
                "Per chunk (1 call/chunk, precise)": "chunk",
            }
            self.option_granularity = ctk.CTkOptionMenu(
                granularity_container,
                values=list(self.GRANULARITY_LABELS.keys()),
                command=self._on_granularity_changed,
                width=230,
            )
            self.option_granularity.set("Document (1 LLM call, fast)")
            self.option_granularity.pack(anchor="w")

            sample_container = ctk.CTkFrame(autotag_settings_row, fg_color="transparent")
            sample_container.pack(side="left")
            ctk.CTkLabel(
                sample_container, text="Document sample (characters)",
                font=("Segoe UI", 10), text_color="gray70",
            ).pack(anchor="w")
            self.entry_sample_chars = ctk.CTkEntry(sample_container, width=100)
            self.entry_sample_chars.insert(0, str(DOCUMENT_AUTOTAG_SAMPLE_CHARS))
            self.entry_sample_chars.pack(anchor="w")

            self.label_granularity_hint = ctk.CTkLabel(
                autotag_settings_row,
                text="A single LLM call on the start of the text, applied to all chunks — "
                     "fast, ideal for a homogeneous book/text. Increase the sample if "
                     "the beginning doesn't reflect the content well (technique/movement varying across chapters).",
                font=("Segoe UI", 9), text_color="gray60", justify="left", wraplength=500,
            )
            self.label_granularity_hint.pack(side="left", padx=(20, 0))

            options_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
            options_row.grid(row=3, column=0, columnspan=4, padx=15, pady=(5, 15), sticky="w")

            self.var_autotag = ctk.BooleanVar(value=True)
            self.chk_autotag = ctk.CTkCheckBox(
                options_row, text="Enable auto-tagging via Ollama", variable=self.var_autotag
            )
            self.chk_autotag.pack(side="left", padx=(0, 20))

            self.var_force = ctk.BooleanVar(value=False)
            self.chk_force = ctk.CTkCheckBox(
                options_row, text="Force re-ingestion (ignores already processed files)", variable=self.var_force
            )
            self.chk_force.pack(side="left")

            # --- Bouton de lancement ---
            self.btn_run = ctk.CTkButton(
                parent, text="Start ingestion", font=FONT_LABEL, height=40,
                command=self._start_ingestion
            )
            self.btn_run.grid(row=2, column=0, pady=(5, 10), sticky="e")

        # --------------------------------------------------------
        # ONGLET ANALYSE D'ŒUVRE
        # --------------------------------------------------------

        def _build_analysis_tab(self, parent):
            # L'onglet "Artwork Analysis" est scindé en deux sous-onglets pour séparer clairement
            # le flux "une image à la fois" du flux "traitement d'un dossier entier", qui n'ont
            # pas grand-chose en commun visuellement et alourdissaient un unique long scroll.
            parent.grid_rowconfigure(0, weight=1)
            self.analysis_subtabview = ctk.CTkTabview(parent, fg_color="transparent")
            self.analysis_subtabview.grid(row=0, column=0, sticky="nsew")
            tab_single = self.analysis_subtabview.add("Single Image")
            tab_batch = self.analysis_subtabview.add("Batch Folder")
            tab_single.grid_columnconfigure(0, weight=1)
            tab_batch.grid_columnconfigure(0, weight=1)

            self._build_analysis_single_tab(tab_single)
            self._build_analysis_batch_tab(tab_batch)

        def _build_analysis_single_tab(self, parent):
            """Sous-onglet : analyse d'une image unique (sélection -> métadonnées -> texte -> sauvegarde)."""
            parent.grid_rowconfigure(0, weight=1)
            scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
            scroll.grid(row=0, column=0, sticky="nsew")
            scroll.grid_columnconfigure(0, weight=1)

            # --- 1. Sélection de l'image ---
            image_frame = ctk.CTkFrame(scroll)
            image_frame.grid(row=0, column=0, padx=0, pady=(10, 10), sticky="ew")
            image_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(image_frame, text="1. Source image", font=FONT_LABEL).grid(
                row=0, column=0, columnspan=2, padx=15, pady=(15, 5), sticky="w"
            )

            self.image_path_label = ctk.CTkLabel(
                image_frame, text="(no image selected)", font=FONT_LOG, text_color="gray70", anchor="w"
            )
            self.image_path_label.grid(row=1, column=0, padx=15, pady=5, sticky="ew")

            btn_pick_image = ctk.CTkButton(image_frame, text="Browse...", width=140, command=self._pick_image)
            btn_pick_image.grid(row=1, column=1, padx=(5, 15), pady=5, sticky="e")

            self.image_preview_label = ctk.CTkLabel(
                image_frame, text="(no preview)", text_color="gray60", font=FONT_LOG,
                fg_color="gray17", corner_radius=6, width=260, height=260
            )
            self.image_preview_label.grid(row=2, column=0, columnspan=2, padx=15, pady=(0, 15))

            # --- 2. Métadonnées de l'œuvre ---
            meta_frame = ctk.CTkFrame(scroll)
            meta_frame.grid(row=1, column=0, padx=0, pady=10, sticky="ew")
            for i in range(4):
                meta_frame.grid_columnconfigure(i, weight=1)

            ctk.CTkLabel(
                meta_frame,
                text="2. Artwork metadata  (Artist or Artwork title required, the rest is optional)",
                font=FONT_LABEL,
            ).grid(row=0, column=0, columnspan=4, padx=15, pady=(15, 10), sticky="w")

            self.entry_artiste_a = self._labeled_entry(meta_frame, "Artist", row=1, col=0)
            self.entry_oeuvre_a = self._labeled_entry(meta_frame, "Artwork (title)", row=1, col=1)
            self.entry_courant_a = self._labeled_entry(meta_frame, "Art movement", row=1, col=2)
            self.entry_periode_a = self._labeled_entry(meta_frame, "Period", row=1, col=3)
            self.entry_technique_a = self._labeled_entry(meta_frame, "Technique", row=2, col=0)

            # --- 3. Analyse : mode de génération + bouton "générer" côte à côte, puis le texte ---
            text_frame = ctk.CTkFrame(scroll)
            text_frame.grid(row=2, column=0, padx=0, pady=10, sticky="ew")
            text_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                text_frame, text="3. Analysis text", font=FONT_LABEL,
            ).grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 5), sticky="w")

            mode_row = ctk.CTkFrame(text_frame, fg_color="transparent")
            mode_row.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="w")
            ctk.CTkLabel(
                mode_row, text="Source:", font=("Segoe UI", 11), text_color="gray70"
            ).pack(side="left", padx=(0, 10))
            self.analysis_mode = ctk.StringVar(value="vision")
            ctk.CTkRadioButton(
                mode_row, text="Generate via Ollama (vision)", variable=self.analysis_mode, value="vision"
            ).pack(side="left", padx=(0, 20))
            ctk.CTkRadioButton(
                mode_row, text="Type manually", variable=self.analysis_mode, value="manuel"
            ).pack(side="left")

            self.btn_generate_analysis = ctk.CTkButton(
                text_frame, text="Generate analysis (vision)", font=FONT_LABEL, width=220,
                command=self._start_analysis_generation
            )
            self.btn_generate_analysis.grid(row=1, column=1, padx=(0, 15), pady=(0, 10), sticky="e")

            self.analysis_text_box = ctk.CTkTextbox(text_frame, font=FONT_LOG, height=260)
            self.analysis_text_box.grid(row=2, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="ew")

            # --- 4. Sauvegarde : action finale, mise en évidence et séparée du reste ---
            save_frame = ctk.CTkFrame(scroll)
            save_frame.grid(row=3, column=0, padx=0, pady=(0, 15), sticky="ew")
            save_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                save_frame, text="4. Save",
                font=FONT_LABEL, text_color="gray70",
            ).grid(row=0, column=0, padx=15, pady=(12, 0), sticky="w")

            self.btn_ingest_analysis = ctk.CTkButton(
                save_frame, text="Save analysis to the RAG database", font=FONT_LABEL, height=42,
                command=self._start_analysis_ingestion
            )
            self.btn_ingest_analysis.grid(row=1, column=0, padx=15, pady=(8, 15), sticky="ew")

        def _build_analysis_batch_tab(self, parent):
            """Sous-onglet : analyse en lot de toutes les images d'un dossier, puis révision manuelle."""
            parent.grid_rowconfigure(0, weight=1)
            scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
            scroll.grid(row=0, column=0, sticky="nsew")
            scroll.grid_columnconfigure(0, weight=1)

            # --- 1. Choix du dossier ---
            folder_frame = ctk.CTkFrame(scroll)
            folder_frame.grid(row=0, column=0, padx=0, pady=(10, 10), sticky="ew")
            folder_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                folder_frame, text="1. Folder to process", font=FONT_LABEL
            ).grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 5), sticky="w")

            self.batch_dir_label = ctk.CTkLabel(
                folder_frame, text="(no folder selected)", font=FONT_LOG, text_color="gray70", anchor="w"
            )
            self.batch_dir_label.grid(row=1, column=0, padx=15, pady=5, sticky="ew")

            btn_pick_batch_dir = ctk.CTkButton(
                folder_frame, text="Browse...", width=140, command=self._pick_batch_directory
            )
            btn_pick_batch_dir.grid(row=1, column=1, padx=(5, 15), pady=5, sticky="e")

            self.batch_files_display = ctk.CTkTextbox(folder_frame, font=FONT_LOG, height=70)
            self.batch_files_display.grid(row=2, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="ew")
            self.batch_files_display.insert("end", "(no image found)")
            self.batch_files_display.configure(state="disabled")

            # --- 2. Options du traitement en lot ---
            options_frame = ctk.CTkFrame(scroll)
            options_frame.grid(row=1, column=0, padx=0, pady=10, sticky="ew")
            options_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                options_frame, text="2. Batch options", font=FONT_LABEL
            ).grid(row=0, column=0, padx=15, pady=(15, 10), sticky="w")

            batch_options_row = ctk.CTkFrame(options_frame, fg_color="transparent")
            batch_options_row.grid(row=1, column=0, padx=15, pady=(0, 5), sticky="ew")

            ctk.CTkCheckBox(
                batch_options_row, text="Automatically integrate into the RAG database",
                variable=self.batch_auto_ingest_var,
            ).pack(side="left", padx=(0, 20))

            ctk.CTkCheckBox(
                batch_options_row, text="Automatically rename image files",
                variable=self.batch_auto_rename_var, command=self._on_batch_rename_toggled,
            ).pack(side="left")

            rename_row = ctk.CTkFrame(options_frame, fg_color="transparent")
            rename_row.grid(row=2, column=0, padx=15, pady=(8, 15), sticky="ew")
            ctk.CTkLabel(
                rename_row, text="Rename pattern:", font=("Segoe UI", 10), text_color="gray70"
            ).pack(side="left", padx=(0, 8))
            self.batch_rename_pattern_entry = ctk.CTkEntry(rename_row, width=260)
            self.batch_rename_pattern_entry.insert(0, "{artiste} - {oeuvre}")
            self.batch_rename_pattern_entry.pack(side="left", fill="x", expand=True)
            self.batch_rename_pattern_entry.configure(state="disabled")
            ctk.CTkLabel(
                rename_row, text="  (available: {artiste} {oeuvre} {courant} {periode} {technique})",
                font=("Segoe UI", 9), text_color="gray60",
            ).pack(side="left", padx=(8, 0))

            # --- 3. Lancement ---
            start_frame = ctk.CTkFrame(scroll, fg_color="transparent")
            start_frame.grid(row=2, column=0, padx=0, pady=(0, 10), sticky="ew")
            start_frame.grid_columnconfigure(0, weight=1)

            self.btn_start_batch = ctk.CTkButton(
                start_frame, text="3. Analyze the folder", font=FONT_LABEL, height=42,
                command=self._start_batch_analysis
            )
            self.btn_start_batch.grid(row=0, column=0, sticky="ew")

            # --- 4. Révision manuelle des analyses générées en lot (auto-ingest désactivé) ---
            review_frame = ctk.CTkFrame(scroll)
            review_frame.grid(row=3, column=0, padx=0, pady=(0, 15), sticky="ew")
            review_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                review_frame, text="4. Review generated analyses", font=FONT_LABEL,
            ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

            ctk.CTkLabel(
                review_frame,
                text="Analyses generated in batch appear below — click 'Review / edit' to load "
                     "the full analysis text and metadata in the 'Single Image' tab, then use "
                     "'Save analysis to the RAG database' there as usual.",
                font=("Segoe UI", 9), text_color="gray60", wraplength=760, justify="left",
            ).grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")

            # Frame simple (pas de scroll imbriqué) : c'est l'onglet entier qui défile.
            self.batch_review_frame = ctk.CTkFrame(review_frame, fg_color="transparent")
            self.batch_review_frame.grid(row=2, column=0, padx=15, pady=(0, 15), sticky="ew")
            self.batch_review_frame.grid_columnconfigure(0, weight=1)

        # --------------------------------------------------------
        # ONGLET RECHERCHE RAG
        # --------------------------------------------------------

        def _build_recherche_tab(self, parent):
            parent.grid_rowconfigure(0, weight=1)

            # --- Historique de la conversation ---
            chat_frame = ctk.CTkFrame(parent)
            chat_frame.grid(row=0, column=0, padx=0, pady=(10, 10), sticky="nsew")
            chat_frame.grid_columnconfigure(0, weight=1)
            chat_frame.grid_rowconfigure(1, weight=1)

            ctk.CTkLabel(
                chat_frame, text="Chat with the graphic arts expert", font=FONT_LABEL
            ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

            self.chat_box = ctk.CTkTextbox(chat_frame, font=FONT_LOG, wrap="word")
            self.chat_box.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
            self.chat_box.configure(state="disabled")

            # --- Éléments détectés dans la dernière réponse (liste -> requêtes de recherche web) ---
            self.chat_items_frame = ctk.CTkFrame(parent)
            self.chat_items_frame.grid_columnconfigure(0, weight=1)

            chat_items_header = ctk.CTkFrame(self.chat_items_frame, fg_color="transparent")
            chat_items_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(12, 5))
            ctk.CTkLabel(
                chat_items_header,
                text="Items detected in the response (click to search the web)",
                font=FONT_LABEL,
            ).pack(side="left")
            ctk.CTkButton(
                chat_items_header, text="Search all", width=130, height=26,
                command=self._search_all_chat_items,
            ).pack(side="right")

            self.chat_items_wrap = ctk.CTkScrollableFrame(self.chat_items_frame, fg_color="transparent", height=110)
            self.chat_items_wrap.grid(row=1, column=0, sticky="ew", padx=15, pady=(0, 12))
            self.chat_items_wrap.grid_columnconfigure(0, weight=1)

            self.chat_items_frame.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")
            self.chat_items_frame.grid_remove()  # masqué tant qu'aucune liste n'a été détectée

            # --- Zone de saisie ---
            input_frame = ctk.CTkFrame(parent)
            input_frame.grid(row=2, column=0, padx=0, pady=(0, 10), sticky="ew")
            input_frame.grid_columnconfigure(0, weight=1)

            self.entry_question = ctk.CTkTextbox(input_frame, height=70, font=FONT_LOG)
            self.entry_question.grid(row=0, column=0, columnspan=3, padx=15, pady=(15, 10), sticky="ew")

            self.var_force_image_prompt = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(
                input_frame,
                text="Force image prompt generation (JSON, 10 keys)",
                variable=self.var_force_image_prompt,
                font=("Segoe UI", 10),
            ).grid(row=1, column=0, padx=15, pady=(0, 15), sticky="w")

            self.btn_history = ctk.CTkButton(
                input_frame, text="History", font=FONT_LABEL, height=35, width=120,
                fg_color="gray30", hover_color="gray20", command=self._open_history_window,
            )
            self.btn_history.grid(row=1, column=1, padx=(5, 5), pady=(0, 15), sticky="e")

            self.btn_new_chat = ctk.CTkButton(
                input_frame, text="New conversation", font=FONT_LABEL, height=35, width=170,
                fg_color="gray30", hover_color="gray20", command=self._reset_chat,
            )
            self.btn_new_chat.grid(row=1, column=2, padx=(5, 5), pady=(0, 15), sticky="e")

            self.btn_search = ctk.CTkButton(
                input_frame, text="Send", font=FONT_LABEL, height=35, width=120,
                command=self._start_rag_query,
            )
            self.btn_search.grid(row=1, column=3, padx=(5, 15), pady=(0, 15), sticky="e")

        # --------------------------------------------------------
        # DISCUSSION (routeur RAG + chat multi-tours, dans un thread séparé)
        # --------------------------------------------------------

        def _start_rag_query(self):
            if self.is_running:
                return
            question = self.entry_question.get("1.0", "end").strip()
            if not question:
                self._append_log("[!] The message is empty.")
                return

            force_image_prompt = self.var_force_image_prompt.get()

            is_new_conversation = False
            if self.current_conversation_id is None:
                self.current_conversation_id = str(uuid.uuid4())
                is_new_conversation = True
                # Titre provisoire (troncature brute), remplacé par le titre LLM en tout début du
                # worker ci-dessous, avant même la recherche RAG - séquentiellement, dans le même
                # thread, pour ne jamais solliciter Ollama en parallèle d'un autre appel.
                create_conversation(
                    self.history_conn, self.current_conversation_id, make_conversation_title(question)
                )

            self.chat_history.append({"role": "user", "content": question})
            add_conversation_message(self.history_conn, self.current_conversation_id, "user", question)
            self._append_chat("You", question)
            self.entry_question.delete("1.0", "end")

            self.active_job = "rag_query"
            self.is_running = True
            self.btn_search.configure(state="disabled", text="Thinking...")
            self.progress_bar.start()
            self.status_label.configure(text="Analyzing the message and searching the knowledge base...")

            history_snapshot = list(self.chat_history)
            self.worker_thread = threading.Thread(
                target=self._run_rag_query_worker,
                args=(history_snapshot, force_image_prompt, is_new_conversation, question, self.current_conversation_id),
                daemon=True,
            )
            self.worker_thread.start()

        def _run_rag_query_worker(
            self, history_snapshot, force_image_prompt,
            is_new_conversation=False, first_question="", conversation_id=None,
        ):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    if is_new_conversation and conversation_id:
                        print("[...] Generating the conversation title...")
                        title = generate_conversation_title(first_question)
                        update_conversation_title(self.history_conn, conversation_id, title)
                        print(f"[OK] Title: {title}")

                    if not check_ollama_models():
                        print("[ABORTED] Fix the missing models above before retrying.")
                        self.result_queue.put(("assistant_start", None))
                        self.result_queue.put(("token", "[ERROR] Missing Ollama models, see the log below."))
                        self.result_queue.put(("assistant_end", None))
                        return

                    print("[...] Analyzing the message topic...")
                    route = route_message(history_snapshot)
                    if force_image_prompt:
                        route["wants_image_prompt"] = True
                        route["art_related"] = True

                    chunks = []
                    if route["art_related"]:
                        print(f"[...] Art-related topic detected -> RAG search: \"{route['search_query']}\"")
                        chunks = search_chunks(
                            route["search_query"] or history_snapshot[-1]["content"], top_k=TOP_K_DEFAULT
                        )
                        print(f"[OK] {len(chunks)} excerpt(s) found" if chunks
                              else "[!] No relevant excerpt found, answering from general expertise")
                    else:
                        print("[...] Message outside the art domain, no RAG search for this turn")

                    stats = {}
                    self.result_queue.put(("assistant_start", None))

                    if route["wants_image_prompt"]:
                        print("[...] Generating structured image prompt (JSON, 10 keys)...")
                        result = generate_image_prompt(history_snapshot, chunks, stats=stats)
                        if "_error" in result:
                            text = f"[ERROR] {result['_error']}"
                        else:
                            text = format_image_prompt_json(result)
                            if "_warning" in result:
                                text += f"\n[!] {result['_warning']}"
                        self.result_queue.put(("token", text))
                        self.chat_history.append({"role": "assistant", "content": text})
                        add_conversation_message(self.history_conn, self.current_conversation_id, "assistant", text)
                        self.result_queue.put(("chat_list_items", []))
                    else:
                        context_block = build_context_block(chunks) if chunks else ""
                        full_answer = chat_llm(
                            history_snapshot, context_block=context_block,
                            on_token=lambda tok: self.result_queue.put(("token", tok)),
                            stats=stats,
                        )
                        self.chat_history.append({"role": "assistant", "content": full_answer})
                        add_conversation_message(
                            self.history_conn, self.current_conversation_id, "assistant", full_answer
                        )
                        list_items = extract_list_items(full_answer)
                        if list_items:
                            print(f"[i] {len(list_items)} list item(s) detected in the response")
                        self.result_queue.put(("chat_list_items", list_items))

                    self.result_queue.put(("assistant_end", None))
                    print(format_llm_stats(stats))
            except Exception as e:
                self.result_queue.put(("token", f"[ERROR] {e}"))
                self.result_queue.put(("assistant_end", None))

        def _append_chat(self, speaker: str, text: str):
            """Ajoute un tour complet (avec préfixe locuteur) à l'historique affiché."""
            self._append_chat_raw(f"{speaker} : {text}\n\n")

        def _append_chat_raw(self, text: str):
            """Ajoute un fragment brut (sans préfixe) à la fin de la zone de discussion, sans rien effacer."""
            self.chat_box.configure(state="normal")
            self.chat_box.insert("end", text)
            self.chat_box.see("end")
            self.chat_box.configure(state="disabled")

        def _reset_chat(self):
            if self.is_running:
                return
            self.chat_history = []
            self.current_conversation_id = None
            self.chat_box.configure(state="normal")
            self.chat_box.delete("1.0", "end")
            self.chat_box.configure(state="disabled")
            self._show_chat_list_items([])
            self._append_log("[i] New conversation started.")

        def _open_history_window(self):
            """Ouvre une fenêtre listant les conversations enregistrées (titre + date), avec
            possibilité de reprendre ou de supprimer chacune d'entre elles."""
            if self.is_running:
                return

            win = ctk.CTkToplevel(self)
            win.title("Conversation history")
            win.geometry("560x480")
            win.transient(self)
            win.grab_set()

            ctk.CTkLabel(
                win, text="Previous conversations", font=FONT_LABEL
            ).pack(padx=15, pady=(15, 10), anchor="w")

            rows = list_conversations(self.history_conn)

            if not rows:
                ctk.CTkLabel(
                    win, text="No conversations saved yet.", text_color="gray60"
                ).pack(padx=15, pady=20, anchor="w")
                return

            scroll = ctk.CTkScrollableFrame(win)
            scroll.pack(fill="both", expand=True, padx=15, pady=(0, 15))
            scroll.grid_columnconfigure(0, weight=1)

            for conversation_id, title, updated_at in rows:
                row = ctk.CTkFrame(scroll)
                row.grid(sticky="ew", pady=4)
                row.grid_columnconfigure(0, weight=1)

                ctk.CTkLabel(
                    row, text=f"{title}\n{updated_at}", font=FONT_LOG, justify="left", anchor="w"
                ).grid(row=0, column=0, padx=10, pady=8, sticky="w")

                ctk.CTkButton(
                    row, text="Resume", width=90, height=28,
                    command=lambda cid=conversation_id, w=win: self._load_conversation(cid, w),
                ).grid(row=0, column=1, padx=(5, 5), pady=8)

                ctk.CTkButton(
                    row, text="Regenerate title", width=120, height=28,
                    fg_color="gray30", hover_color="gray20",
                    command=lambda cid=conversation_id, r=row: self._regenerate_conversation_title(cid, r),
                ).grid(row=0, column=2, padx=(0, 5), pady=8)

                ctk.CTkButton(
                    row, text="Delete", width=90, height=28,
                    fg_color="#8B3A3A", hover_color="#6E2E2E",
                    command=lambda cid=conversation_id, w=win: self._delete_conversation(cid, w),
                ).grid(row=0, column=3, padx=(0, 10), pady=8)

        def _load_conversation(self, conversation_id: str, win=None):
            """Recharge une conversation enregistrée pour la reprendre là où elle s'était arrêtée."""
            if self.is_running:
                return

            messages = load_conversation(self.history_conn, conversation_id)
            self.chat_history = messages
            self.current_conversation_id = conversation_id

            self.chat_box.configure(state="normal")
            self.chat_box.delete("1.0", "end")
            self.chat_box.configure(state="disabled")
            for msg in messages:
                speaker = "You" if msg["role"] == "user" else "Expert"
                self._append_chat(speaker, msg["content"])

            self._show_chat_list_items([])
            self._append_log(f"[i] Conversation loaded ({len(messages)} message(s)).")

            if win is not None:
                win.destroy()

        def _delete_conversation(self, conversation_id: str, win):
            """Supprime définitivement une conversation enregistrée, puis rafraîchit la fenêtre."""
            delete_conversation(self.history_conn, conversation_id)
            if self.current_conversation_id == conversation_id:
                self._reset_chat()
            win.destroy()
            self._open_history_window()

        def _regenerate_conversation_title(self, conversation_id: str, row):
            """
            Relance la génération du titre à partir de l'INTÉGRALITÉ de la conversation (et non
            plus seulement du premier message) : utile quand le sujet a dérivé au fil des tours et
            que le titre d'origine ne correspond plus vraiment à ce qui a été discuté.
            """
            label, *buttons = row.winfo_children()
            current_title, _, date_part = label.cget("text").partition("\n")
            label.configure(text=f"{current_title}\n(regenerating title...)")
            for b in buttons:
                b.configure(state="disabled")

            threading.Thread(
                target=self._regenerate_title_worker,
                args=(conversation_id, current_title, date_part, row, label, buttons),
                daemon=True,
            ).start()

        def _regenerate_title_worker(self, conversation_id, current_title, date_part, row, label, buttons):
            new_title = current_title
            error_msg = None
            try:
                history = load_conversation(self.history_conn, conversation_id)
                if history:
                    new_title = generate_conversation_title_from_history(history, fallback=current_title)
                    update_conversation_title(self.history_conn, conversation_id, new_title)
                else:
                    error_msg = "empty conversation"
            except Exception as e:
                # Filet de sécurité : sans ce try/except, une exception ici tue le thread en
                # silence et self.after() n'est jamais appelé -> l'UI reste bloquée sur "en
                # cours..." sans jamais refléter ni le succès ni l'échec de l'appel LLM.
                error_msg = f"{type(e).__name__}: {e}"
                print(f"[!] Title regeneration failed: {error_msg}")

            self.after(
                0, lambda: self._on_title_regenerated(row, label, buttons, new_title, date_part, error_msg)
            )

        def _on_title_regenerated(self, row, label, buttons, new_title, date_part, error_msg=None):
            if not row.winfo_exists():
                return
            if error_msg:
                label.configure(text=f"{new_title}\n{date_part}  [regeneration failed: {error_msg}]")
                self._append_log(f"[!] Title regeneration failed: {error_msg}")
            else:
                label.configure(text=f"{new_title}\n{date_part}")
                self._append_log(f"[i] Title regenerated: {new_title}")
            for b in buttons:
                b.configure(state="normal")

        def _show_chat_list_items(self, items: list):
            """Affiche (ou masque si vide) la liste d'éléments détectés dans la dernière réponse du chat."""
            self.chat_list_items = items
            for widget in self.chat_items_wrap.winfo_children():
                widget.destroy()

            if not items:
                self.chat_items_frame.grid_remove()
                return

            for i, item in enumerate(items):
                row = ctk.CTkFrame(self.chat_items_wrap, fg_color="transparent")
                row.grid(row=i, column=0, sticky="ew", pady=2)
                row.grid_columnconfigure(0, weight=1)
                ctk.CTkLabel(
                    row, text=item, font=("Segoe UI", 11), anchor="w", justify="left", wraplength=650
                ).grid(row=0, column=0, sticky="ew", padx=(5, 10))
                ctk.CTkButton(
                    row, text="Search", width=100, height=26,
                    fg_color="gray30", hover_color="gray20",
                    command=lambda it=item: self._search_chat_item(it),
                ).grid(row=0, column=1, padx=5)

            self.chat_items_frame.grid()

        def _selected_web_sources(self) -> list:
            sources = []
            if self.var_source_wikipedia.get():
                sources.append("wikipedia")
            if self.var_source_met.get():
                sources.append("met_museum")
            if self.var_source_ddgs.get():
                sources.append("ddgs")
            return sources or ["wikipedia", "met_museum", "ddgs"]

        def _search_chat_item(self, item: str):
            """Lance une recherche web pour un seul élément détecté (bascule sur l'onglet Recherche web)."""
            if self.is_running:
                self._append_log("[!] Another operation is in progress, try again in a moment.")
                return
            self.tabview.set("Web Search")
            self.entry_web_query.delete(0, "end")
            self.entry_web_query.insert(0, item)
            self._start_web_search()

        def _search_all_chat_items(self):
            """Lance une recherche web groupée pour tous les éléments détectés dans la dernière réponse."""
            if self.is_running:
                self._append_log("[!] Another operation is in progress, try again in a moment.")
                return
            if not self.chat_list_items:
                return

            sources = self._selected_web_sources()
            items = list(self.chat_list_items)

            self.tabview.set("Web Search")
            self.active_job = "web_search"
            self.is_running = True
            self.btn_web_search.configure(state="disabled", text="Searching...")
            self.progress_bar.start()
            self.status_label.configure(text=f"Bulk search of {len(items)} item(s)...")

            self.worker_thread = threading.Thread(
                target=self._run_chat_batch_search_worker, args=(items, sources), daemon=True
            )
            self.worker_thread.start()

        def _run_chat_batch_search_worker(self, items, sources):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    seen_urls = set()
                    merged = []
                    for i, item in enumerate(items, start=1):
                        print(f"[...] ({i}/{len(items)}) Searching: \"{item}\"")
                        for result in web_search_all(item, sources=tuple(sources), max_results_per_source=3):
                            if result["url"] in seen_urls:
                                continue
                            seen_urls.add(result["url"])
                            result["_query"] = item
                            merged.append(result)
                    print(f"[OK] {len(merged)} unique result(s) total for {len(items)} item(s)")
                    self.result_queue.put(("web_results", merged))
            except Exception as e:
                print(f"[X] Error during bulk search: {e}")
                self.result_queue.put(("web_results", []))

        # --------------------------------------------------------
        # RECHERCHE WEB (alimentation du RAG) : recherche -> revue manuelle -> ingestion
        # --------------------------------------------------------

        def _build_websearch_tab(self, parent):
            parent.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(2, weight=2)
            parent.grid_rowconfigure(3, weight=1)

            # --- Barre de recherche ---
            search_frame = ctk.CTkFrame(parent)
            search_frame.grid(row=0, column=0, padx=0, pady=(10, 10), sticky="ew")
            search_frame.grid_columnconfigure(0, weight=1)

            self.entry_web_query = ctk.CTkEntry(
                search_frame,
                placeholder_text="E.g.: analytical cubism, sfumato techniques, Egon Schiele...",
            )
            self.entry_web_query.grid(row=0, column=0, padx=(15, 10), pady=15, sticky="ew")

            self.btn_web_search = ctk.CTkButton(
                search_frame, text="Search", font=FONT_LABEL, height=35, width=140,
                command=self._start_web_search,
            )
            self.btn_web_search.grid(row=0, column=1, padx=(0, 15), pady=15)

            # --- Sources + options ---
            sources_frame = ctk.CTkFrame(parent)
            sources_frame.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")

            ctk.CTkLabel(sources_frame, text="Sources:", font=FONT_LABEL).grid(
                row=0, column=0, padx=(15, 10), pady=15, sticky="w"
            )
            self.var_source_wikipedia = ctk.BooleanVar(value=True)
            self.var_source_met = ctk.BooleanVar(value=True)
            self.var_source_ddgs = ctk.BooleanVar(value=True)
            ctk.CTkCheckBox(sources_frame, text="Wikipedia", variable=self.var_source_wikipedia).grid(
                row=0, column=1, padx=10, pady=15
            )
            ctk.CTkCheckBox(sources_frame, text="Met Museum", variable=self.var_source_met).grid(
                row=0, column=2, padx=10, pady=15
            )
            ctk.CTkCheckBox(sources_frame, text="Web search (FR/EN)", variable=self.var_source_ddgs).grid(
                row=0, column=3, padx=10, pady=15
            )
            self.var_web_force = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(sources_frame, text="Force re-ingestion", variable=self.var_web_force).grid(
                row=0, column=4, padx=(20, 15), pady=15
            )

            # --- Résultats (revue manuelle, sélection par case à cocher) ---
            results_frame = ctk.CTkFrame(parent)
            results_frame.grid(row=2, column=0, padx=0, pady=(0, 10), sticky="nsew")
            results_frame.grid_columnconfigure(0, weight=1)
            results_frame.grid_rowconfigure(1, weight=1)

            header_row = ctk.CTkFrame(results_frame, fg_color="transparent")
            header_row.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")
            ctk.CTkLabel(header_row, text="Results (check what you want to ingest)", font=FONT_LABEL).pack(
                side="left"
            )
            ctk.CTkButton(
                header_row, text="Uncheck all", width=110, height=26,
                fg_color="gray30", hover_color="gray20",
                command=lambda: self._toggle_all_web_results(False),
            ).pack(side="right", padx=(5, 0))
            ctk.CTkButton(
                header_row, text="Check all", width=100, height=26,
                fg_color="gray30", hover_color="gray20",
                command=lambda: self._toggle_all_web_results(True),
            ).pack(side="right", padx=(5, 0))

            self.web_results_scroll = ctk.CTkScrollableFrame(results_frame, fg_color="transparent")
            self.web_results_scroll.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
            self.web_results_scroll.grid_columnconfigure(0, weight=1)

            # --- Aperçu du texte qui sera réellement ingéré ---
            preview_frame = ctk.CTkFrame(parent)
            preview_frame.grid(row=3, column=0, padx=0, pady=(0, 10), sticky="nsew")
            preview_frame.grid_columnconfigure(0, weight=1)
            preview_frame.grid_rowconfigure(1, weight=1)
            ctk.CTkLabel(preview_frame, text="Preview (click 'Preview' on a result)", font=FONT_LABEL).grid(
                row=0, column=0, padx=15, pady=(15, 5), sticky="w"
            )
            self.web_preview_box = ctk.CTkTextbox(preview_frame, font=FONT_LOG, wrap="word")
            self.web_preview_box.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
            self.web_preview_box.configure(state="disabled")

            # --- Action d'ingestion ---
            action_frame = ctk.CTkFrame(parent, fg_color="transparent")
            action_frame.grid(row=4, column=0, padx=0, pady=(0, 10), sticky="ew")
            self.btn_web_ingest = ctk.CTkButton(
                action_frame, text="Ingest selection", font=FONT_LABEL, height=38,
                command=self._start_web_ingest,
            )
            self.btn_web_ingest.pack(side="right", padx=15)

        def _toggle_all_web_results(self, value: bool):
            for var in self.web_result_vars:
                var.set(value)

        def _populate_web_results(self, results: list):
            for widget in self.web_results_scroll.winfo_children():
                widget.destroy()
            self.web_results = results
            self.web_result_vars = []

            if not results:
                ctk.CTkLabel(
                    self.web_results_scroll, text="No results for this search.", font=("Segoe UI", 11)
                ).grid(row=0, column=0, sticky="w", padx=5, pady=5)
                return

            source_labels = {"wikipedia": "Wikipedia", "met_museum": "Met Museum", "ddgs": "Web"}

            for i, result in enumerate(results):
                var = ctk.BooleanVar(value=False)
                self.web_result_vars.append(var)
                is_met = result.get("source") == "met_museum"

                row_frame = ctk.CTkFrame(self.web_results_scroll)
                row_frame.grid(row=i, column=0, sticky="ew", pady=4, padx=2)
                row_frame.grid_columnconfigure(2, weight=1)

                ctk.CTkCheckBox(row_frame, text="", variable=var, width=20).grid(
                    row=0, column=0, rowspan=2, padx=(10, 5), pady=8
                )

                # --- Vignette (uniquement pour les résultats Met Museum, chargée en arrière-plan) ---
                thumb_label = ctk.CTkLabel(row_frame, text="", width=56, height=56)
                if is_met:
                    thumb_label.grid(row=0, column=1, rowspan=2, padx=(0, 5), pady=8)
                    obj_data = result.get("_object_data") or {}
                    thumb_url = obj_data.get("primaryImageSmall") or obj_data.get("primaryImage")
                    if thumb_url and Image is not None:
                        threading.Thread(
                            target=self._load_met_thumbnail, args=(thumb_url, thumb_label), daemon=True
                        ).start()

                title_text = f"{result['title']}   ·   {source_labels.get(result['source'], result['source'])}"
                ctk.CTkLabel(
                    row_frame, text=title_text, font=FONT_LABEL, anchor="w", justify="left", wraplength=600
                ).grid(row=0, column=2, sticky="ew", padx=5, pady=(8, 0))

                snippet = result.get("snippet") or result["url"]
                if result.get("_query"):
                    snippet = f"[{result['_query']}]  {snippet}"
                ctk.CTkLabel(
                    row_frame, text=snippet, font=("Segoe UI", 10), text_color="gray60",
                    anchor="w", justify="left", wraplength=600,
                ).grid(row=1, column=2, sticky="ew", padx=5, pady=(0, 8))

                btn_col = ctk.CTkFrame(row_frame, fg_color="transparent")
                btn_col.grid(row=0, column=3, rowspan=2, padx=10)

                ctk.CTkButton(
                    btn_col, text="Preview", width=110, height=28,
                    fg_color="gray30", hover_color="gray20",
                    command=lambda r=result: self._preview_web_result(r),
                ).pack(pady=(0, 4))

                if is_met:
                    ctk.CTkButton(
                        btn_col, text="Send to analysis", width=110, height=28,
                        command=lambda r=result: self._send_met_result_to_analysis(r),
                    ).pack()

        def _load_met_thumbnail(self, url: str, label_widget):
            """Télécharge une vignette Met Museum en arrière-plan et l'affiche une fois prête (thread-safe via after)."""
            data = download_image_bytes(url)
            if not data:
                return
            try:
                pil_img = Image.open(io.BytesIO(data))
                pil_img.thumbnail((56, 56))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            except Exception:
                return
            self.after(0, lambda: (label_widget.configure(image=ctk_img, text=""), setattr(label_widget, "_img_ref", ctk_img)))

        def _set_web_preview(self, text: str):
            self.web_preview_box.configure(state="normal")
            self.web_preview_box.delete("1.0", "end")
            self.web_preview_box.insert("end", text)
            self.web_preview_box.configure(state="disabled")

        def _start_web_search(self):
            if self.is_running:
                return
            query = self.entry_web_query.get().strip()
            if not query:
                self._append_log("[!] La recherche est vide.")
                return

            sources = self._selected_web_sources()
            self.active_job = "web_search"
            self.is_running = True
            self.btn_web_search.configure(state="disabled", text="Searching...")
            self.progress_bar.start()
            self.status_label.configure(text=f"Recherche web : \"{query}\"...")

            self.worker_thread = threading.Thread(
                target=self._run_web_search_worker, args=(query, sources), daemon=True
            )
            self.worker_thread.start()

        def _run_web_search_worker(self, query, sources):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    print(f"[...] Recherche web \"{query}\" sur : {', '.join(sources)}")
                    results = web_search_all(query, sources=tuple(sources), max_results_per_source=6)
                    print(f"[OK] {len(results)} unique result(s) found")
                    self.result_queue.put(("web_results", results))
            except Exception as e:
                print(f"[X] Erreur pendant la recherche web : {e}")
                self.result_queue.put(("web_results", []))

        def _preview_web_result(self, result):
            if self.is_running:
                self._append_log("[!] Another operation is in progress, try again in a moment.")
                return

            self._set_web_preview("Fetching...")

            self.active_job = "web_preview"
            self.is_running = True
            self.btn_web_ingest.configure(state="disabled")
            self.progress_bar.start()
            self.status_label.configure(text=f"Fetching: {result['title']}...")

            self.worker_thread = threading.Thread(
                target=self._run_web_preview_worker, args=(result,), daemon=True
            )
            self.worker_thread.start()

        def _run_web_preview_worker(self, result):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    print(f"[...] Fetching content: {result['url']}")
                    content = fetch_web_content(result)
                    if "_error" in content:
                        print(f"[X] {content['_error']}")
                        self.result_queue.put(("web_preview", f"[ERREUR] {content['_error']}"))
                    else:
                        print(f"[OK] {len(content['text'])} characters extracted")
                        preview_text = f"{content['title']}\n{result['url']}\n\n{content['text'][:5000]}"
                        if len(content["text"]) > 5000:
                            preview_text += (
                                "\n\n[...] (preview truncated to 5000 characters; the full text "
                                "will be used if you ingest this source)"
                            )
                        self.result_queue.put(("web_preview", preview_text))
            except Exception as e:
                self.result_queue.put(("web_preview", f"[ERROR] {e}"))

        # --------------------------------------------------------
        # ENVOI D'UN RÉSULTAT MET MUSEUM VERS L'ONGLET "ANALYSE D'ŒUVRE"
        # --------------------------------------------------------

        def _send_met_result_to_analysis(self, result):
            if self.is_running:
                self._append_log("[!] Another operation is in progress, try again in a moment.")
                return

            obj = result.get("_object_data") or {}
            image_url = obj.get("primaryImage") or obj.get("primaryImageSmall")
            if not image_url:
                self._append_log("[!] This Met Museum artwork has no usable image.")
                return

            self.active_job = "met_to_analysis"
            self.is_running = True
            self.progress_bar.start()
            self.status_label.configure(text=f"Fetching image: {obj.get('title', '')}...")

            self.worker_thread = threading.Thread(
                target=self._run_send_met_to_analysis_worker, args=(image_url, obj), daemon=True
            )
            self.worker_thread.start()

        def _run_send_met_to_analysis_worker(self, image_url: str, obj: dict):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    print(f"[...] Downloading image: {image_url}")
                    data = download_image_bytes(image_url)
                    if not data:
                        print("[X] Image download failed.")
                        return
                    suffix = Path(urllib.parse.urlparse(image_url).path).suffix or ".jpg"
                    image_path = save_image_bytes_to_temp(data, suffix=suffix)
                    print(f"[OK] Image saved temporarily: {image_path}")
                    self.result_queue.put(("met_to_analysis", {"image_path": image_path, "obj": obj}))
            except Exception as e:
                print(f"[X] Error while sending to analysis: {e}")

        def _apply_met_result_to_analysis(self, payload: dict):
            """Bascule sur l'onglet 'Analyse d'œuvre' et préremplit l'image + les métadonnées connues (thread UI)."""
            image_path: Path = payload["image_path"]
            obj = payload["obj"]

            self.selected_image = image_path
            self.image_path_label.configure(text=str(self.selected_image), text_color="white")
            self._refresh_analysis_image_preview()

            def _fill(entry, value):
                if value:
                    entry.delete(0, "end")
                    entry.insert(0, value)

            _fill(self.entry_artiste_a, obj.get("artistDisplayName", ""))
            _fill(self.entry_oeuvre_a, obj.get("title", ""))
            _fill(self.entry_periode_a, obj.get("objectDate", "") or obj.get("period", ""))
            _fill(self.entry_technique_a, obj.get("medium", ""))

            self.analysis_mode.set("vision")
            self.tabview.set("Artwork Analysis")
            self.analysis_subtabview.set("Single Image")
            self._append_log(f"[OK] Artwork sent to analysis: {obj.get('title', '(untitled)')}")

        def _start_web_ingest(self):
            if self.is_running:
                return
            selected = [r for r, v in zip(self.web_results, self.web_result_vars) if v.get()]
            if not selected:
                self._append_log("[!] No result checked for ingestion.")
                return

            force = self.var_web_force.get()

            self.active_job = "web_ingest"
            self.is_running = True
            self.btn_web_ingest.configure(state="disabled", text="Ingesting...")
            self.progress_bar.start()
            self.status_label.configure(text=f"Ingesting {len(selected)} web source(s)...")

            self.worker_thread = threading.Thread(
                target=self._run_web_ingest_worker, args=(selected, force), daemon=True
            )
            self.worker_thread.start()

        def _run_web_ingest_worker(self, selected, force):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    if not check_ollama_models():
                        print("[ABORTED] Fix the missing models above before retrying.")
                        return

                    sqlite_conn = init_sqlite(DB_PATH)
                    chroma_collection = init_chroma()

                    for i, result in enumerate(selected, start=1):
                        print(f"\n[{i}/{len(selected)}] {result['title']} ({result['url']})")
                        try:
                            outcome = ingest_web_result(result, sqlite_conn, chroma_collection, force=force)
                        except KeyboardInterrupt:
                            print("[!] Interruption requested, stopping web ingestion.")
                            break
                        except Exception as e:
                            print(f"  [X] Unexpected error: {e}")
                            continue
                        tag = {"ok": "[OK]", "skip": "[SKIP]", "partial": "[!]", "error": "[X]"}.get(
                            outcome["status"], "[?]"
                        )
                        print(f"  {tag} {outcome['message']}")

                    print(f"\n[DONE] {len(selected)} source(s) processed.")
            except Exception as e:
                print(f"[X] Error during web ingestion: {e}")

        # --------------------------------------------------------
        # ONGLET PARAMÈTRES
        # --------------------------------------------------------

        def _build_settings_tab(self, parent):
            parent.grid_rowconfigure(0, weight=1)

            scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
            scroll.grid(row=0, column=0, padx=0, pady=(10, 0), sticky="nsew")
            scroll.grid_columnconfigure(0, weight=1)

            self.settings_widgets = {}  # key -> widget (CTkEntry/CTkOptionMenu)

            # Regroupe SETTINGS_SCHEMA par section en conservant l'ordre d'apparition
            sections = {}
            for key, cast, section, label, desc, needs_restart in SETTINGS_SCHEMA:
                sections.setdefault(section, []).append((key, cast, label, desc, needs_restart))

            row = 0
            for section, entries in sections.items():
                sect_frame = ctk.CTkFrame(scroll)
                sect_frame.grid(row=row, column=0, padx=5, pady=(0, 15), sticky="ew")
                sect_frame.grid_columnconfigure(0, weight=1)
                row += 1

                ctk.CTkLabel(sect_frame, text=section, font=FONT_LABEL).grid(
                    row=0, column=0, padx=15, pady=(12, 8), sticky="w"
                )

                for i, (key, cast, label, desc, needs_restart) in enumerate(entries, start=1):
                    field_row = ctk.CTkFrame(sect_frame, fg_color="transparent")
                    field_row.grid(row=i, column=0, padx=15, pady=4, sticky="ew")
                    field_row.grid_columnconfigure(1, weight=1)

                    label_text = label + ("  (restart required)" if needs_restart else "")
                    lbl = ctk.CTkLabel(field_row, text=label_text, font=("Segoe UI", 12), width=280, anchor="w")
                    lbl.grid(row=0, column=0, padx=(0, 10), pady=(2, 0), sticky="w")

                    if key == "AUTOTAG_GRANULARITY":
                        widget = ctk.CTkOptionMenu(field_row, values=["document", "chunk"])
                        widget.set(str(globals()[key]))
                        widget.grid(row=0, column=1, sticky="ew")
                    elif key == "LLM_BACKEND":
                        widget = ctk.CTkOptionMenu(field_row, values=["ollama", "lmstudio"])
                        widget.set(str(globals()[key]))
                        widget.grid(row=0, column=1, sticky="ew")
                    else:
                        widget = ctk.CTkEntry(field_row)
                        widget.insert(0, str(globals()[key]))
                        widget.grid(row=0, column=1, sticky="ew")

                    if desc:
                        ctk.CTkLabel(
                            field_row, text=desc, font=("Segoe UI", 10), text_color="gray60", anchor="w"
                        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 2))

                    self.settings_widgets[key] = (widget, cast)

            # --- Modèles disponibles sur le backend actif (aide au remplissage) ---
            models_frame = ctk.CTkFrame(scroll)
            models_frame.grid(row=row, column=0, padx=5, pady=(0, 15), sticky="ew")
            models_frame.grid_columnconfigure(0, weight=1)
            row += 1

            ctk.CTkLabel(models_frame, text="Detected models (active backend)", font=FONT_LABEL).grid(
                row=0, column=0, padx=15, pady=(12, 8), sticky="w"
            )
            self.settings_models_box = ctk.CTkTextbox(models_frame, font=FONT_LOG, height=90)
            self.settings_models_box.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")
            self.settings_models_box.configure(state="disabled")
            ctk.CTkButton(
                models_frame, text="List available models (Ollama or LM Studio, per Backend setting above)",
                command=self._settings_list_models
            ).grid(row=2, column=0, padx=15, pady=(0, 15), sticky="w")

            # --- Actions ---
            action_row = ctk.CTkFrame(parent, fg_color="transparent")
            action_row.grid(row=1, column=0, padx=5, pady=15, sticky="ew")

            ctk.CTkButton(
                action_row, text="Save settings", command=self._settings_save
            ).pack(side="left", padx=(10, 10))
            ctk.CTkButton(
                action_row, text="Reset to defaults", fg_color="gray30",
                command=self._settings_reset_defaults
            ).pack(side="left")

            self.settings_status_label = ctk.CTkLabel(parent, text="", font=("Segoe UI", 11), text_color="gray70")
            self.settings_status_label.grid(row=2, column=0, padx=15, pady=(0, 10), sticky="w")

        def _settings_list_models(self):
            def worker():
                try:
                    names = llm_list_models(timeout=10)
                    text = "\n".join(names) if names else "(no model found)"
                except Exception as e:
                    text = f"Error: {e}"

                def apply():
                    self.settings_models_box.configure(state="normal")
                    self.settings_models_box.delete("1.0", "end")
                    self.settings_models_box.insert("1.0", text)
                    self.settings_models_box.configure(state="disabled")

                self.after(0, apply)

            threading.Thread(target=worker, daemon=True).start()

        def _settings_save(self):
            errors = []
            for key, (widget, cast) in self.settings_widgets.items():
                raw = widget.get().strip() if hasattr(widget, "get") else ""
                try:
                    globals()[key] = cast(raw)
                except (TypeError, ValueError):
                    errors.append(key)

            # LLM_CONCURRENCY / EMBED_CONCURRENCY changent la taille des sémaphores : on les
            # recrée pour que la valeur soit effective sans relancer l'appli (best effort ;
            # les autres réglages marqués "redémarrage requis" touchent des ressources déjà
            # ouvertes — DB, ChromaDB — et ne sont eux pas réappliqués à chaud).
            global _embed_semaphore, _llm_semaphore
            _embed_semaphore = Semaphore(max(1, EMBED_CONCURRENCY))
            _llm_semaphore = Semaphore(max(1, LLM_CONCURRENCY))

            save_config()

            if errors:
                self.settings_status_label.configure(
                    text=f"Saved with errors on: {', '.join(errors)} (values ignored).",
                    text_color="orange",
                )
            else:
                self.settings_status_label.configure(
                    text="Settings saved to art_rag_config.json.", text_color="lightgreen"
                )

        def _settings_reset_defaults(self):
            if CONFIG_PATH.exists():
                try:
                    CONFIG_PATH.unlink()
                except Exception as e:
                    self.settings_status_label.configure(text=f"Error: {e}", text_color="orange")
                    return
            self.settings_status_label.configure(
                text="art_rag_config.json deleted — restart the application to return to default values.",
                text_color="gray70",
            )

        def _labeled_entry(self, parent, label_text, row, col):
            container = ctk.CTkFrame(parent, fg_color="transparent")
            container.grid(row=row, column=col, padx=15, pady=5, sticky="ew")
            ctk.CTkLabel(container, text=label_text, font=("Segoe UI", 10), text_color="gray70").pack(anchor="w")
            entry = ctk.CTkEntry(container)
            entry.pack(fill="x")
            return entry

        # --------------------------------------------------------
        # SÉLECTION DE FICHIERS
        # --------------------------------------------------------

        def _pick_files(self):
            paths = filedialog.askopenfilenames(
                title="Choose documents",
                filetypes=[("Supported documents", "*.pdf *.txt *.md"), ("All files", "*.*")],
            )
            for p in paths:
                path_obj = Path(p)
                if path_obj not in self.selected_files:
                    self.selected_files.append(path_obj)
            self._refresh_files_display()

        def _pick_directory(self):
            directory = filedialog.askdirectory(title="Choose a folder to ingest recursively")
            if not directory:
                return
            d = Path(directory)
            found = []
            for ext in SUPPORTED_EXTENSIONS:
                found.extend(sorted(d.rglob(f"*{ext}")))
            for f in found:
                if f not in self.selected_files:
                    self.selected_files.append(f)
            self._refresh_files_display()

        def _clear_files(self):
            self.selected_files = []
            self._refresh_files_display()

        def _refresh_files_display(self):
            self.files_display.configure(state="normal")
            self.files_display.delete("1.0", "end")
            if not self.selected_files:
                self.files_display.insert("end", "(no file selected)")
            else:
                for f in self.selected_files:
                    self.files_display.insert("end", f"{f}\n")
            self.files_display.configure(state="disabled")

        def _pick_image(self):
            path = filedialog.askopenfilename(
                title="Choose an artwork image",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")],
            )
            if not path:
                return
            self.selected_image = Path(path)
            self.image_path_label.configure(text=str(self.selected_image), text_color="white")
            self._refresh_analysis_image_preview()

        # --------------------------------------------------------
        # ANALYSE EN LOT D'UN RÉPERTOIRE D'IMAGES
        # --------------------------------------------------------

        def _on_batch_rename_toggled(self):
            state = "normal" if self.batch_auto_rename_var.get() else "disabled"
            self.batch_rename_pattern_entry.configure(state=state)

        def _pick_batch_directory(self):
            directory = filedialog.askdirectory(title="Choose a folder of images to analyze")
            if not directory:
                return
            d = Path(directory)
            found = []
            for ext in IMAGE_EXTENSIONS:
                found.extend(sorted(d.rglob(f"*{ext}")))
                found.extend(sorted(d.rglob(f"*{ext.upper()}")))
            # dédoublonnage en conservant l'ordre (rglob insensible à la casse peut dupliquer sur Windows)
            seen = set()
            unique_found = []
            for f in found:
                if f not in seen:
                    seen.add(f)
                    unique_found.append(f)

            self.batch_image_files = unique_found
            self.batch_dir_label.configure(text=str(d), text_color="white")
            self._refresh_batch_files_display()

        def _refresh_batch_files_display(self):
            self.batch_files_display.configure(state="normal")
            self.batch_files_display.delete("1.0", "end")
            if not self.batch_image_files:
                self.batch_files_display.insert("end", "(no image found)")
            else:
                self.batch_files_display.insert(
                    "end", f"{len(self.batch_image_files)} image(s) found:\n"
                )
                for f in self.batch_image_files:
                    self.batch_files_display.insert("end", f"{f.name}\n")
            self.batch_files_display.configure(state="disabled")

        def _start_batch_analysis(self):
            if self.is_running:
                self._append_log("[!] Another operation is in progress, try again in a moment.")
                return
            if not self.batch_image_files:
                self._append_log("[!] Choose a folder containing images first.")
                return

            auto_ingest = self.batch_auto_ingest_var.get()
            auto_rename = self.batch_auto_rename_var.get()
            rename_pattern = self.batch_rename_pattern_entry.get().strip() or "{artiste} - {oeuvre}"

            # on repart d'une file de révision propre pour ce nouveau lot
            for child in list(self.batch_review_frame.winfo_children()):
                child.destroy()
            self.batch_review_items = []

            self.active_job = "batch_analysis"
            self.is_running = True
            self.btn_start_batch.configure(state="disabled", text="Analyzing...")
            self.btn_generate_analysis.configure(state="disabled")
            self.btn_ingest_analysis.configure(state="disabled")
            self.progress_bar.start()
            self.status_label.configure(
                text=f"Batch analysis: 0/{len(self.batch_image_files)} images..."
            )

            self.worker_thread = threading.Thread(
                target=self._run_batch_analysis_worker,
                args=(list(self.batch_image_files), auto_ingest, auto_rename, rename_pattern),
                daemon=True,
            )
            self.worker_thread.start()

        def _run_batch_analysis_worker(self, image_files, auto_ingest, auto_rename, rename_pattern):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    if not check_ollama_models():
                        print("[ABORTED] Fix the missing models above before retrying.")
                        return

                    sqlite_conn = init_sqlite(DB_PATH) if auto_ingest else None
                    chroma_collection = init_chroma() if auto_ingest else None

                    total = len(image_files)
                    print(f"[...] Starting batch analysis of {total} image(s)...")

                    for i, img_path in enumerate(image_files, start=1):
                        self.result_queue.put(("batch_progress", (i, total, img_path.name)))

                        if not img_path.exists():
                            print(f"[!] ({i}/{total}) File not found (moved/deleted?): {img_path}")
                            continue

                        print(f"[...] ({i}/{total}) Analyzing: {img_path.name}")
                        analysis_text = analyze_artwork_vision(img_path)
                        if not analysis_text:
                            print(f"[X] ({i}/{total}) Analysis failed: {img_path.name}")
                            continue

                        metadata = autotag_chunk(analysis_text)

                        current_path = img_path
                        if auto_rename:
                            current_path = rename_artwork_file(img_path, rename_pattern, metadata)
                            if current_path != img_path:
                                print(f"  [OK] Renamed to: {current_path.name}")

                        if auto_ingest:
                            if not metadata.artiste and not metadata.oeuvre:
                                print(
                                    f"  [!] No artist/title detected for {current_path.name}, "
                                    "ingestion skipped (fill it in manually via review if needed)."
                                )
                                self.result_queue.put(("batch_review_item", {
                                    "image_path": current_path, "analysis_text": analysis_text,
                                    "metadata": metadata,
                                }))
                                continue
                            ingest_artwork_analysis(
                                analysis_text, sqlite_conn, chroma_collection,
                                artiste=metadata.artiste, oeuvre=metadata.oeuvre,
                                courant=metadata.courant, periode=metadata.periode,
                                technique=metadata.technique, source_confidence="vision",
                            )
                        else:
                            self.result_queue.put(("batch_review_item", {
                                "image_path": current_path, "analysis_text": analysis_text,
                                "metadata": metadata,
                            }))

                    print(f"[OK] Batch analysis complete: {total} image(s) processed.")
            except Exception as e:
                self.log_queue.put(f"[FATAL ERROR] {e}")

        def _add_batch_review_item(self, payload: dict):
            """Ajoute une carte de révision manuelle pour une analyse générée en lot (thread UI)."""
            image_path: Path = payload["image_path"]
            analysis_text: str = payload["analysis_text"]
            metadata = payload["metadata"]

            row = ctk.CTkFrame(self.batch_review_frame, fg_color="gray17", corner_radius=6)
            row.grid(row=len(self.batch_review_items), column=0, padx=0, pady=4, sticky="ew")
            row.grid_columnconfigure(0, weight=1)

            label_text = f"{image_path.name} — {metadata.artiste or '?'} / {metadata.oeuvre or '?'}"
            ctk.CTkLabel(row, text=label_text, font=FONT_LOG, anchor="w").grid(
                row=0, column=0, padx=10, pady=8, sticky="ew"
            )

            item = {"image_path": image_path, "analysis_text": analysis_text, "metadata": metadata, "row": row}

            btn_review = ctk.CTkButton(
                row, text="Review / edit", width=110,
                command=lambda it=item: self._load_batch_review_item(it),
            )
            btn_review.grid(row=0, column=1, padx=(0, 10), pady=8)

            self.batch_review_items.append(item)

        def _load_batch_review_item(self, item: dict):
            """Charge une analyse en attente de révision dans les champs principaux de l'onglet."""
            if self.is_running:
                self._append_log("[!] Another operation is in progress, try again in a moment.")
                return

            image_path: Path = item["image_path"]
            metadata = item["metadata"]

            self.selected_image = image_path
            self.image_path_label.configure(text=str(self.selected_image), text_color="white")
            self._refresh_analysis_image_preview()

            def _fill(entry, value):
                entry.delete(0, "end")
                if value:
                    entry.insert(0, value)

            _fill(self.entry_artiste_a, metadata.artiste)
            _fill(self.entry_oeuvre_a, metadata.oeuvre)
            _fill(self.entry_courant_a, metadata.courant)
            _fill(self.entry_periode_a, metadata.periode)
            _fill(self.entry_technique_a, metadata.technique)

            self.analysis_mode.set("manuel")
            self.analysis_text_box.delete("1.0", "end")
            self.analysis_text_box.insert("1.0", item["analysis_text"])

            self.analysis_subtabview.set("Single Image")
            self._append_log(
                f"[OK] Loaded for review: {image_path.name}. Edit if needed then click "
                "'Ingest this analysis'."
            )

        def _refresh_analysis_image_preview(self):
            """Charge self.selected_image et l'affiche en miniature dans l'onglet Analyse d'œuvre."""
            if self.selected_image is None:
                self._append_log("[!] Preview: no image selected (selected_image is None).")
                return
            if Image is None:
                self._append_log(
                    "[!] Preview unavailable: the PIL/Pillow module could not be imported "
                    f"('from PIL import Image' failed at startup) — detail: {PIL_IMPORT_ERROR}. "
                    "Check 'pip show pillow' with the SAME Python interpreter that runs this script."
                )
                return
            try:
                pil_img = Image.open(self.selected_image)
                pil_img = pil_img.convert("RGB")
                preview = pil_img.copy()
                preview.thumbnail((260, 260))
                ctk_img = ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)
                self.image_preview_label.configure(image=ctk_img, text="")
                self.image_preview_label._img_ref = ctk_img  # évite le garbage collection de l'image
                self.image_preview_label.update_idletasks()
                self._append_log(f"[OK] Preview loaded: {self.selected_image.name} ({preview.size[0]}x{preview.size[1]})")
            except Exception as e:
                self.image_preview_label.configure(image=None, text="(preview unavailable)")
                self._append_log(f"[!] Image preview failed: {e}")

        # --------------------------------------------------------
        # LOG (thread-safe via queue)
        # --------------------------------------------------------

        def _append_log(self, text: str):
            self.log_box.configure(state="normal")
            self.log_box.insert("end", text if text.endswith("\n") else text + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        def _poll_log_queue(self):
            try:
                while True:
                    line = self.log_queue.get_nowait()
                    self._append_log(line)
            except queue.Empty:
                pass

            try:
                while True:
                    kind, payload = self.result_queue.get_nowait()
                    if kind == "assistant_start":
                        self._append_chat_raw("Expert: ")
                    elif kind == "token":
                        self._append_chat_raw(payload)
                    elif kind == "assistant_end":
                        self._append_chat_raw("\n\n")
                    elif kind == "web_results":
                        self._populate_web_results(payload)
                    elif kind == "web_preview":
                        self._set_web_preview(payload)
                    elif kind == "chat_list_items":
                        self._show_chat_list_items(payload)
                    elif kind == "met_to_analysis":
                        self._apply_met_result_to_analysis(payload)
                    elif kind == "batch_progress":
                        i, total, name = payload
                        self.status_label.configure(text=f"Batch analysis: {i}/{total} — {name}")
                    elif kind == "batch_review_item":
                        self._add_batch_review_item(payload)
            except queue.Empty:
                pass

            if self.is_running and self.worker_thread is not None and not self.worker_thread.is_alive():
                self._on_job_finished()

            self.after(100, self._poll_log_queue)

        # --------------------------------------------------------
        # LANCEMENT DE L'INGESTION DE DOCUMENTS (dans un thread séparé)
        # --------------------------------------------------------

        def _on_granularity_changed(self, choice: str):
            """Ajuste l'aide contextuelle et l'état du champ échantillon selon la granularité choisie."""
            if self.GRANULARITY_LABELS.get(choice) == "chunk":
                self.entry_sample_chars.configure(state="disabled")
                self.label_granularity_hint.configure(
                    text="1 LLM call per chunk: slower, but captures variations in "
                         "movement/technique within a single book (e.g. a work that "
                         "covers several movements or several artists)."
                )
            else:
                self.entry_sample_chars.configure(state="normal")
                self.label_granularity_hint.configure(
                    text="A single LLM call on the start of the text, applied to all chunks — "
                         "fast, ideal for a homogeneous book/text. Increase the sample if "
                         "the beginning doesn't reflect the content well (technique/movement varying across chapters)."
                )

        def _start_ingestion(self):
            if self.is_running:
                return
            if not self.selected_files:
                self._append_log("[!] No file selected.")
                return

            manual_metadata = None
            courant = self.entry_courant.get().strip()
            artiste = self.entry_artiste.get().strip()
            periode = self.entry_periode.get().strip()
            technique = self.entry_technique.get().strip()
            if any([courant, artiste, periode, technique]):
                manual_metadata = {
                    "courant": courant,
                    "artiste": artiste,
                    "periode": periode,
                    "technique": technique,
                    "mots_cles": [],
                }

            autotag = self.var_autotag.get()
            force = self.var_force.get()
            files = list(self.selected_files)

            autotag_granularity = self.GRANULARITY_LABELS.get(self.option_granularity.get(), "document")
            sample_chars_raw = self.entry_sample_chars.get().strip()
            try:
                document_sample_chars = int(sample_chars_raw) if sample_chars_raw else DOCUMENT_AUTOTAG_SAMPLE_CHARS
                if document_sample_chars <= 0:
                    raise ValueError
            except ValueError:
                self._append_log(
                    f"[!] Invalid sample size ('{sample_chars_raw}'), "
                    f"using the default value ({DOCUMENT_AUTOTAG_SAMPLE_CHARS})."
                )
                document_sample_chars = DOCUMENT_AUTOTAG_SAMPLE_CHARS

            self.active_job = "documents"
            self.is_running = True
            self.btn_run.configure(state="disabled", text="Ingesting...")
            self.progress_bar.start()
            self.status_label.configure(text=f"Processing {len(files)} file(s)...")

            self.worker_thread = threading.Thread(
                target=self._run_ingestion_worker,
                args=(files, manual_metadata, autotag, force, autotag_granularity, document_sample_chars),
                daemon=True,
            )
            self.worker_thread.start()

        def _run_ingestion_worker(self, files, manual_metadata, autotag, force,
                                   autotag_granularity, document_sample_chars):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    print(f"=== {len(files)} file(s) to process ===\n")
                    sqlite_conn = init_sqlite(DB_PATH)
                    chroma_collection = init_chroma()

                    for f in files:
                        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                            print(f"[SKIP] Unsupported extension: {f.name}")
                            continue
                        ingest_file(
                            f,
                            sqlite_conn,
                            chroma_collection,
                            manual_metadata=manual_metadata,
                            autotag=autotag,
                            force=force,
                            autotag_granularity=autotag_granularity,
                            document_sample_chars=document_sample_chars,
                        )

                    print("=== Ingestion complete ===")
            except Exception as e:
                self.log_queue.put(f"[FATAL ERROR] {e}")

        # --------------------------------------------------------
        # GÉNÉRATION D'ANALYSE D'ŒUVRE VIA VISION (dans un thread séparé)
        # --------------------------------------------------------

        def _start_analysis_generation(self):
            if self.is_running:
                return
            if self.analysis_mode.get() != "vision":
                self._append_log("[!] Switch to 'Generate via Ollama (vision)' mode to use this button.")
                return
            if self.selected_image is None:
                self._append_log("[!] Choose an artwork image first.")
                return

            artiste = self.entry_artiste_a.get().strip()
            oeuvre = self.entry_oeuvre_a.get().strip()
            courant = self.entry_courant_a.get().strip()

            self._last_analysis_text = None
            self.active_job = "generation"
            self.is_running = True
            self.btn_generate_analysis.configure(state="disabled", text="Generating...")
            self.btn_ingest_analysis.configure(state="disabled")
            self.progress_bar.start()
            self.status_label.configure(text="Analyzing the image using the vision model...")

            self.worker_thread = threading.Thread(
                target=self._run_analysis_generation_worker,
                args=(self.selected_image, artiste, oeuvre, courant),
                daemon=True,
            )
            self.worker_thread.start()

        def _run_analysis_generation_worker(self, image_path, artiste, oeuvre, courant):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    if not check_ollama_models():
                        print("[ABORTED] Fix the missing models above before retrying.")
                        return
                    print(f"[...] Generating analysis via {OLLAMA_VISION_MODEL}...")
                    result = analyze_artwork_vision(image_path, artiste=artiste, oeuvre=oeuvre, courant=courant)
                    if result:
                        self._last_analysis_text = result
                        print("[OK] Analysis generated. Review/edit the text then click 'Ingest this analysis'.")
                    else:
                        print("[X] Analysis generation failed.")
            except Exception as e:
                self.log_queue.put(f"[FATAL ERROR] {e}")

        # --------------------------------------------------------
        # INGESTION DE L'ANALYSE D'ŒUVRE (dans un thread séparé)
        # --------------------------------------------------------

        def _start_analysis_ingestion(self):
            if self.is_running:
                return
            text = self.analysis_text_box.get("1.0", "end").strip()
            if not text:
                self._append_log("[!] The analysis field is empty (generate it or write it manually).")
                return

            artiste = self.entry_artiste_a.get().strip()
            oeuvre = self.entry_oeuvre_a.get().strip()
            if not artiste and not oeuvre:
                self._append_log("[!] Fill in at least the Artist or the Artwork before ingesting.")
                return

            courant = self.entry_courant_a.get().strip()
            periode = self.entry_periode_a.get().strip()
            technique = self.entry_technique_a.get().strip()
            source_confidence = "manuel" if self.analysis_mode.get() == "manuel" else "vision"

            self.active_job = "ingest_analysis"
            self.is_running = True
            self.btn_ingest_analysis.configure(state="disabled", text="Ingesting...")
            self.btn_generate_analysis.configure(state="disabled")
            self.progress_bar.start()
            self.status_label.configure(text=f"Ingesting analysis ({artiste or oeuvre})...")

            self.worker_thread = threading.Thread(
                target=self._run_analysis_ingestion_worker,
                args=(text, artiste, oeuvre, courant, periode, technique, source_confidence),
                daemon=True,
            )
            self.worker_thread.start()

        def _run_analysis_ingestion_worker(self, text, artiste, oeuvre, courant, periode, technique, source_confidence):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    sqlite_conn = init_sqlite(DB_PATH)
                    chroma_collection = init_chroma()
                    ingest_artwork_analysis(
                        text,
                        sqlite_conn,
                        chroma_collection,
                        artiste=artiste,
                        oeuvre=oeuvre,
                        courant=courant,
                        periode=periode,
                        technique=technique,
                        source_confidence=source_confidence,
                    )
            except Exception as e:
                self.log_queue.put(f"[FATAL ERROR] {e}")

        # --------------------------------------------------------
        # FIN DE JOB (commun aux workers)
        # --------------------------------------------------------

        def _on_job_finished(self):
            self.is_running = False
            self.progress_bar.stop()
            self.progress_bar.set(0)

            if self.active_job == "documents":
                self.btn_run.configure(state="normal", text="Start ingestion")
            elif self.active_job == "generation":
                self.btn_generate_analysis.configure(state="normal", text="Generate analysis (vision)")
                self.btn_ingest_analysis.configure(state="normal")
                if self._last_analysis_text:
                    self.analysis_text_box.delete("1.0", "end")
                    self.analysis_text_box.insert("1.0", self._last_analysis_text)
            elif self.active_job == "ingest_analysis":
                self.btn_ingest_analysis.configure(state="normal", text="Ingest this analysis")
                self.btn_generate_analysis.configure(state="normal")
            elif self.active_job == "batch_analysis":
                self.btn_start_batch.configure(state="normal", text="Analyze the folder")
                self.btn_generate_analysis.configure(state="normal")
                self.btn_ingest_analysis.configure(state="normal")
            elif self.active_job == "rag_query":
                self.btn_search.configure(state="normal", text="Send")
            elif self.active_job == "web_search":
                self.btn_web_search.configure(state="normal", text="Search")
            elif self.active_job == "web_preview":
                self.btn_web_ingest.configure(state="normal")
            elif self.active_job == "web_ingest":
                self.btn_web_ingest.configure(state="normal", text="Ingest selection")

            self.status_label.configure(text="Done.")
            self.active_job = None

    app = ArtRagIngestApp()
    app.mainloop()


# ============================================================
# CLI
# ============================================================

def _run_ingest(args, parser):
    if args.image or args.texte_manuel:
        if not args.artiste and not args.oeuvre:
            parser.error("Mode analyse d'œuvre : précise au moins --artiste ou --oeuvre")

        sqlite_conn = init_sqlite(DB_PATH)
        chroma_collection = init_chroma()

        analysis_text = args.texte_manuel
        if not analysis_text:
            if args.no_vision:
                parser.error("--image sans --texte-manuel nécessite l'appel vision (retire --no-vision)")
            if not check_ollama_models():
                print("\n[ABANDON] Corrige les modèles manquants ci-dessus avant de relancer.")
                sys.exit(1)
            image_path = Path(args.image)
            if not image_path.exists():
                print(f"Image introuvable : {image_path}")
                sys.exit(1)
            print(f"[...] Génération de l'analyse via {OLLAMA_VISION_MODEL} (peut prendre un moment)...")
            analysis_text = analyze_artwork_vision(
                image_path, artiste=args.artiste or "", oeuvre=args.oeuvre or "", courant=args.courant or ""
            )
            if not analysis_text:
                print("[ABANDON] Échec de la génération de l'analyse.")
                sys.exit(1)
            print("\n=== Analyse générée ===")
            print(analysis_text)
            print()

        ingest_artwork_analysis(
            analysis_text,
            sqlite_conn,
            chroma_collection,
            artiste=args.artiste or "",
            oeuvre=args.oeuvre or "",
            courant=args.courant or "",
            periode=args.periode or "",
            technique=args.technique or "",
            source_confidence="manuel" if args.texte_manuel else "vision",
            force=args.force,
        )
        return

    if not args.file and not args.dir:
        parser.error("Précise --file ou --dir, ou --image/--texte-manuel pour une analyse d'œuvre")

    manual_metadata = None
    if any([args.courant, args.artiste, args.periode, args.technique]):
        manual_metadata = {
            "courant": args.courant or "",
            "artiste": args.artiste or "",
            "periode": args.periode or "",
            "technique": args.technique or "",
            "mots_cles": [],
        }

    files = collect_files(args)
    if not files:
        print("Aucun fichier à ingérer (extensions supportées : .pdf, .txt, .md)")
        return

    print(f"=== {len(files)} fichier(s) à traiter ===\n")

    if not check_ollama_models():
        print("\n[ABANDON] Corrige les modèles manquants ci-dessus avant de relancer.")
        sys.exit(1)

    sqlite_conn = init_sqlite(DB_PATH)
    chroma_collection = init_chroma()

    for f in files:
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        ingest_file(
            f,
            sqlite_conn,
            chroma_collection,
            manual_metadata=manual_metadata,
            autotag=not args.no_autotag,
            force=args.force,
            autotag_granularity=args.autotag_granularity,
            document_sample_chars=args.document_sample_chars,
        )

    print("=== Ingestion terminée ===")
    print(f"SQLite : {DB_PATH}")
    print(f"ChromaDB : {CHROMA_DIR}")


def _run_query(args, parser):
    if not check_ollama_models():
        print("\n[ABANDON] Corrige les modèles manquants ci-dessus avant de relancer.")
        sys.exit(1)

    where_filter = build_where_filter(
        courant=args.courant or "",
        artiste=args.artiste or "",
        oeuvre=args.oeuvre or "",
        analyses_only=args.analyses_only,
    )

    print(f"[...] Recherche des {args.top_k} extraits les plus pertinents...")
    chunks = search_chunks(args.question, top_k=args.top_k, where_filter=where_filter)

    if not chunks:
        print("[!] Aucun résultat trouvé (base vide, filtre trop restrictif, ou erreur d'embedding)")
        return

    print(f"[OK] {len(chunks)} extrait(s) trouvé(s)\n")

    if args.show_sources:
        print("=== Extraits utilisés ===")
        for i, c in enumerate(chunks):
            meta = c["metadata"]
            print(f"[{i + 1}] {meta.get('source_file', '?')} "
                  f"(courant: {meta.get('courant', '-')}, distance: {c['distance']:.3f})")
            print(f"    {c['text'][:200]}...\n")

    print("[...] Génération de la réponse...\n")
    print("=== Réponse ===")
    stats = {}
    if args.no_stream:
        answer = ask_llm(args.question, chunks, stats=stats)
        print(answer)
    else:
        # Affiche chaque fragment dès qu'il arrive au lieu d'attendre la réponse complète
        ask_llm(args.question, chunks, on_token=lambda tok: print(tok, end="", flush=True), stats=stats)
        print()
    print()
    print(format_llm_stats(stats))


def _run_chat(args, parser):
    """
    Mode conversationnel en terminal : boucle de dialogue avec l'expert en arts graphiques.
    Utilise process_chat_turn() — le même point d'entrée que l'onglet "Discussion" du GUI — donc
    le routeur RAG et la génération de prompt image se comportent exactement pareil ici.
    Tape 'exit' ou Ctrl+C pour quitter, 'reset' pour repartir d'une conversation vierge.
    """
    if not check_ollama_models():
        print("\n[ABANDON] Corrige les modèles manquants ci-dessus avant de relancer.")
        sys.exit(1)

    print("=== Discussion avec l'expert en arts graphiques ===")
    print("Tape 'exit' pour quitter, 'reset' pour repartir d'une conversation vierge.\n")

    history: list = []
    while True:
        try:
            question = input("Toi > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[i] Fin de la conversation.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "reset":
            history = []
            print("[i] Nouvelle conversation démarrée.\n")
            continue

        history.append({"role": "user", "content": question})

        print("[...] Analyse du message...")
        stats = {}
        result = process_chat_turn(history, top_k=args.top_k, force_image_prompt=args.image_prompt, stats=stats)

        print("\nExpert > ", end="", flush=True)
        if result["kind"] == "image_prompt":
            content = result["content"]
            if "_error" in content:
                text = f"[ERREUR] {content['_error']}"
            else:
                text = format_image_prompt_json(content)
                if "_warning" in content:
                    text += f"\n[!] {content['_warning']}"
            print(text)
            history.append({"role": "assistant", "content": text})
        else:
            text = result["content"]
            print(text)
            history.append({"role": "assistant", "content": text})

        print(f"\n{format_llm_stats(stats)}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Art RAG — ingestion de documents, analyse d'œuvres et recherche sémantique (peinture / histoire de l'art)"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- gui ---
    subparsers.add_parser("gui", help="Lance l'interface graphique (CustomTkinter)")

    # --- ingest ---
    p_ingest = subparsers.add_parser("ingest", help="Ingère des documents ou une analyse d'œuvre")
    p_ingest.add_argument("--file", type=str, help="Chemin vers un fichier unique (PDF/TXT/MD)")
    p_ingest.add_argument("--dir", type=str, help="Dossier à ingérer récursivement")
    p_ingest.add_argument("--courant", type=str, help="Force le courant artistique (métadonnée manuelle)")
    p_ingest.add_argument("--artiste", type=str, help="Force l'artiste (métadonnée manuelle)")
    p_ingest.add_argument("--periode", type=str, help="Force la période (métadonnée manuelle)")
    p_ingest.add_argument("--technique", type=str, help="Force la technique (métadonnée manuelle)")
    p_ingest.add_argument("--no-autotag", action="store_true", help="Désactive l'auto-tagging Ollama")
    p_ingest.add_argument("--force", action="store_true", help="Ré-ingère même si le fichier a déjà été traité")
    p_ingest.add_argument("--autotag-granularity", choices=["document", "chunk"], default=AUTOTAG_GRANULARITY,
                           help="'document' (défaut, rapide : 1 appel LLM/fichier) ou 'chunk' (1 appel LLM/chunk, plus lent mais plus fin)")
    p_ingest.add_argument("--document-sample-chars", type=int, default=DOCUMENT_AUTOTAG_SAMPLE_CHARS,
                           help=f"Taille de l'échantillon (en caractères) utilisé pour l'auto-tagging niveau document "
                                f"(défaut: {DOCUMENT_AUTOTAG_SAMPLE_CHARS}). Ignoré en granularité 'chunk'.")
    # --- Mode analyse d'œuvre ---
    p_ingest.add_argument("--image", type=str, help="Chemin vers une image d'œuvre à analyser (mode analyse d'œuvre)")
    p_ingest.add_argument("--oeuvre", type=str, help="Titre de l'œuvre analysée")
    p_ingest.add_argument("--texte-manuel", type=str,
                           help="Texte d'analyse écrit à la main (remplace la génération par vision)")
    p_ingest.add_argument("--no-vision", action="store_true",
                           help="Avec --image mais sans --texte-manuel : n'appelle pas la vision, affiche juste un rappel")

    # --- query ---
    p_query = subparsers.add_parser("query", help="Pose une question à la base de connaissances")
    p_query.add_argument("question", type=str, help="La question à poser")
    p_query.add_argument("--top-k", type=int, default=TOP_K_DEFAULT, help=f"Nombre de chunks à récupérer (défaut: {TOP_K_DEFAULT})")
    p_query.add_argument("--courant", type=str, help="Filtre : ne cherche que dans ce courant artistique")
    p_query.add_argument("--artiste", type=str, help="Filtre : ne cherche que sur cet artiste")
    p_query.add_argument("--oeuvre", type=str, help="Filtre : ne cherche que sur cette œuvre")
    p_query.add_argument("--analyses-only", action="store_true",
                          help="Ne cherche que dans les analyses d'œuvres (exclut les documents généraux)")
    p_query.add_argument("--show-sources", action="store_true", help="Affiche les extraits bruts utilisés en plus de la réponse")
    p_query.add_argument("--no-stream", action="store_true",
                          help="Attend la réponse complète avant de l'afficher (par défaut : affichage au fur et à mesure)")

    # --- chat ---
    p_chat = subparsers.add_parser(
        "chat", help="Discussion en terminal avec l'expert en arts graphiques (routeur RAG + prompts image)"
    )
    p_chat.add_argument("--top-k", type=int, default=TOP_K_DEFAULT,
                         help=f"Nombre de chunks récupérés quand le routeur détecte un sujet artistique (défaut: {TOP_K_DEFAULT})")
    p_chat.add_argument("--image-prompt", action="store_true",
                         help="Force la génération d'un prompt image (JSON, 10 clés) dès le premier message")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Sans sous-commande : lance directement l'interface graphique (comportement historique de gui_ingest.py)
    if args.command is None or args.command == "gui":
        launch_gui()
        return

    try:
        if args.command == "ingest":
            _run_ingest(args, parser)
        elif args.command == "query":
            _run_query(args, parser)
        elif args.command == "chat":
            _run_chat(args, parser)
    except KeyboardInterrupt:
        print("\n[!] Interrompu par l'utilisateur. Les chunks déjà traités sont sauvegardés en base.\n"
              "    Relance exactement la même commande pour reprendre l'ingestion là où elle s'est arrêtée.")
        sys.exit(130)


if __name__ == "__main__":
    main()
