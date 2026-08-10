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
    fichier -> extraction texte -> chunking sémantique -> auto-tagging (Ollama)
            -> embedding (Ollama) -> stockage ChromaDB + SQLite

Pipeline de recherche :
    question -> embedding de la question (nomic-embed-text)
             -> recherche des chunks les plus proches sémantiquement (ChromaDB)
             -> injection de ces chunks dans le prompt du LLM de chat
             -> réponse du LLM, basée sur les extraits fournis, avec sources citées

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
    Ollama doit tourner localement (ollama serve) avec les modèles :
        - nomic-embed-text  (embeddings)
        - gemma4:31b ou équivalent (chat / auto-tagging, adapte OLLAMA_LLM_MODEL)
        - qwen3.5:35b ou équivalent (analyse vision d'œuvres, adapte OLLAMA_VISION_MODEL)
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
    # --- Connexion Ollama ---
    ("OLLAMA_HOST", str, "Ollama", "URL du serveur Ollama", "Ex: http://localhost:11434", False),
    ("OLLAMA_EMBED_MODEL", str, "Ollama", "Modèle d'embedding", "Utilisé pour indexer et interroger la base", False),
    ("OLLAMA_LLM_MODEL", str, "Ollama", "Modèle LLM de référence", "Analyse vision + valeur par défaut si les autres ne sont pas définis", False),
    ("OLLAMA_VISION_MODEL", str, "Ollama", "Modèle vision", "Utilisé pour l'analyse d'œuvres (onglet Analyse d'œuvre)", False),
    ("OLLAMA_ANSWER_MODEL", str, "Ollama", "Modèle de réponse (chat RAG)", "Utilisé pour répondre aux questions dans Discussion", False),
    ("OLLAMA_AUTOTAG_MODEL", str, "Ollama", "Modèle d'auto-tagging", "Utilisé pour extraire courant/artiste/période/technique", False),
    ("OLLAMA_KEEP_ALIVE", str, "Ollama", "Durée de maintien en VRAM", "Ex: 30m, 1h, -1 (indéfiniment)", False),
    # --- Performance ---
    ("OLLAMA_NUM_CTX", int, "Performance", "Taille de contexte (tokens)", "Fenêtre partagée prompt + réponse", False),
    ("OLLAMA_NUM_PREDICT", int, "Performance", "Longueur max de réponse (tokens)", "-1 pour illimité (borné par le contexte)", False),
    ("OLLAMA_EMBED_TIMEOUT", int, "Performance", "Timeout embedding (s)", "", False),
    ("OLLAMA_CHAT_TIMEOUT", int, "Performance", "Timeout chat (s)", "", False),
    ("OLLAMA_VISION_TIMEOUT", int, "Performance", "Timeout vision (s)", "", False),
    ("EMBED_CONCURRENCY", int, "Performance", "Parallélisme embedding", "Nécessite un redémarrage", True),
    ("LLM_CONCURRENCY", int, "Performance", "Parallélisme LLM", "Baisse à 1 en cas de timeouts d'embedding pendant l'ingestion", True),
    ("AUTOTAG_EMBED_BATCH_SIZE", int, "Performance", "Taille de lot autotag/embedding", "Granularité 'chunk' uniquement", False),
    ("TOP_K_DEFAULT", int, "Performance", "Nombre d'extraits RAG (top-k)", "Envoyés au LLM pour répondre", False),
    ("MAX_CHAT_HISTORY_MESSAGES", int, "Performance", "Historique de chat envoyé au LLM", "En nombre de messages (user+assistant)", False),
    # --- Base de connaissances ---
    ("DB_PATH", str, "Base de connaissances", "Chemin de la base SQLite", "Nécessite un redémarrage", True),
    ("CHROMA_DIR", str, "Base de connaissances", "Dossier ChromaDB", "Nécessite un redémarrage", True),
    ("CHROMA_COLLECTION", str, "Base de connaissances", "Nom de la collection", "Nécessite un redémarrage", True),
    ("AUTOTAG_GRANULARITY", str, "Base de connaissances", "Granularité auto-tag", "'document' (rapide) ou 'chunk' (fin)", False),
    ("DOCUMENT_AUTOTAG_SAMPLE_CHARS", int, "Base de connaissances", "Échantillon auto-tag (car.)", "Granularité 'document' uniquement", False),
    # --- Recherche web ---
    ("WIKIPEDIA_LANG", str, "Recherche web", "Langue Wikipedia", "Ex: fr, en", False),
    ("WEB_SEARCH_TIMEOUT", int, "Recherche web", "Timeout recherche (s)", "", False),
    ("WEB_FETCH_TIMEOUT", int, "Recherche web", "Timeout récupération page (s)", "", False),
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
    Vérifie qu'Ollama tourne et que les modèles requis sont bien installés.
    Un 404 sur /api/chat ou /api/embeddings vient quasi toujours d'un modèle
    absent, pas d'un problème réseau : on le détecte ici plutôt que de
    laisser échouer chunk par chunk sur un document de 300 pages.
    Retourne True si tout est OK, False sinon (avec message explicite).
    """
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        resp.raise_for_status()
        installed = {m["name"] for m in resp.json().get("models", [])}
        # Ollama liste parfois avec le tag (":latest") en plus du nom court
        installed_base = {name.split(":")[0] for name in installed}
    except requests.RequestException as e:
        print(f"[ERREUR] Impossible de contacter Ollama sur {OLLAMA_HOST} : {e}")
        print("         Vérifie qu'Ollama tourne (commande : ollama serve)")
        return False

    missing = []
    required_models = {OLLAMA_EMBED_MODEL, OLLAMA_LLM_MODEL, OLLAMA_ANSWER_MODEL, OLLAMA_AUTOTAG_MODEL}
    for required in required_models:
        base_name = required.split(":")[0]
        if required not in installed and base_name not in installed_base:
            missing.append(required)

    if missing:
        print("[ERREUR] Modèle(s) Ollama manquant(s) :")
        for m in missing:
            print(f"           - {m}  ->  ollama pull {m}")
        print(f"[INFO] Modèles actuellement installés : {sorted(installed) or '(aucun)'}")
        print("[INFO] Si tes modèles portent un autre nom (ex: gemma4:31b), adapte")
        print("       OLLAMA_LLM_MODEL / OLLAMA_ANSWER_MODEL / OLLAMA_AUTOTAG_MODEL / OLLAMA_EMBED_MODEL en haut de art_rag.py")
        return False

    print(f"[OK] Modèles Ollama détectés : {', '.join(sorted(required_models))}")
    return True


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
    """Appelle Ollama pour extraire les métadonnées structurées d'un chunk."""
    payload = {
        "model": OLLAMA_AUTOTAG_MODEL,
        "messages": [
            {"role": "system", "content": AUTOTAG_SYSTEM_PROMPT},
            {"role": "user", "content": text[:3000]},  # on limite pour la vitesse
        ],
        "stream": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
    }

    with _llm_semaphore:
        try:
            resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_CHAT_TIMEOUT)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            data = json.loads(content)
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
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text, "keep_alive": OLLAMA_KEEP_ALIVE}
    with _embed_semaphore:
        for attempt in range(retries):
            try:
                resp = requests.post(f"{OLLAMA_HOST}/api/embeddings", json=payload, timeout=OLLAMA_EMBED_TIMEOUT)
                resp.raise_for_status()
                return resp.json()["embedding"]
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

    payload = {
        "model": OLLAMA_VISION_MODEL,
        "messages": [
            {"role": "system", "content": ARTWORK_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyse cette œuvre.{context_hint}", "images": [image_b64]},
        ],
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
    }

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_VISION_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
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

    payload = {
        "model": OLLAMA_ANSWER_MODEL,
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": on_token is not None,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT},
    }

    try:
        if on_token is None:
            resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_CHAT_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            _collect_stats(stats, data)
            return data["message"]["content"]

        # --- mode streaming : Ollama renvoie une suite d'objets JSON, un par ligne ---
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_CHAT_TIMEOUT, stream=True)
        resp.raise_for_status()
        full_text = []
        for line in resp.iter_lines():
            if not line:
                continue
            piece = json.loads(line)
            token = piece.get("message", {}).get("content", "")
            if token:
                full_text.append(token)
                on_token(token)
            if piece.get("done"):
                _collect_stats(stats, piece)
                break
        return "".join(full_text)
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

    payload = {
        "model": OLLAMA_AUTOTAG_MODEL,
        "messages": [
            {"role": "system", "content": TITLE_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 32},
    }

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_CHAT_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()
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

    payload = {
        "model": OLLAMA_AUTOTAG_MODEL,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": conversation_summary},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX},
    }

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_CHAT_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
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

    payload = {
        "model": OLLAMA_ANSWER_MODEL,
        "messages": messages,
        "stream": on_token is not None,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT},
    }

    try:
        if on_token is None:
            resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_CHAT_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            _collect_stats(stats, data)
            return data["message"]["content"]

        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_CHAT_TIMEOUT, stream=True)
        resp.raise_for_status()
        full_text = []
        for line in resp.iter_lines():
            if not line:
                continue
            piece = json.loads(line)
            token = piece.get("message", {}).get("content", "")
            if token:
                full_text.append(token)
                on_token(token)
            if piece.get("done"):
                _collect_stats(stats, piece)
                break
        return "".join(full_text)
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

    payload = {
        "model": OLLAMA_ANSWER_MODEL,
        "messages": [
            {"role": "system", "content": IMAGE_PROMPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT},
    }

    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_CHAT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        _collect_stats(stats, data)
        raw = data["message"]["content"]
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

    WINDOW_TITLE = "Art RAG — Ingestion & Recherche"
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

            self._build_ui()
            self._poll_log_queue()

        # --------------------------------------------------------
        # CONSTRUCTION DE L'INTERFACE
        # --------------------------------------------------------

        def _build_ui(self):
            self.grid_columnconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=1)

            # --- En-tête ---
            header = ctk.CTkLabel(self, text="Art RAG — Ingestion & Recherche", font=FONT_TITLE)
            header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

            # --- Onglets : Documents / Analyse d'œuvre / Recherche RAG ---
            self.tabview = ctk.CTkTabview(self)
            self.tabview.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
            tab_documents = self.tabview.add("Documents")
            tab_analyse = self.tabview.add("Analyse d'œuvre")
            tab_recherche = self.tabview.add("Discussion")
            tab_websearch = self.tabview.add("Recherche web")
            tab_parametres = self.tabview.add("Paramètres")
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
            log_frame.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
            log_frame.grid_columnconfigure(0, weight=1)
            log_frame.grid_rowconfigure(1, weight=1)
            self.grid_rowconfigure(2, weight=1)

            ctk.CTkLabel(log_frame, text="Journal d'exécution", font=FONT_LABEL).grid(
                row=0, column=0, padx=15, pady=(10, 5), sticky="w"
            )

            self.log_box = ctk.CTkTextbox(log_frame, font=FONT_LOG, height=150)
            self.log_box.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
            self.log_box.configure(state="disabled")

            # --- Barre de progression partagée ---
            action_frame = ctk.CTkFrame(self, fg_color="transparent")
            action_frame.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")
            action_frame.grid_columnconfigure(0, weight=1)

            self.progress_bar = ctk.CTkProgressBar(action_frame)
            self.progress_bar.grid(row=0, column=0, sticky="ew")
            self.progress_bar.set(0)
            self.progress_bar.configure(mode="indeterminate")

            self.status_label = ctk.CTkLabel(action_frame, text="Prêt.", font=FONT_LABEL, text_color="gray70")
            self.status_label.grid(row=1, column=0, pady=(8, 0), sticky="w")

        def _build_documents_tab(self, parent):
            parent.grid_rowconfigure(2, weight=1)

            # --- Section sélection de fichiers ---
            selection_frame = ctk.CTkFrame(parent)
            selection_frame.grid(row=0, column=0, padx=0, pady=(10, 10), sticky="ew")
            selection_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(selection_frame, text="Fichiers / dossier à ingérer", font=FONT_LABEL).grid(
                row=0, column=0, columnspan=3, padx=15, pady=(15, 5), sticky="w"
            )

            self.files_display = ctk.CTkTextbox(selection_frame, height=80, font=FONT_LOG)
            self.files_display.grid(row=1, column=0, columnspan=3, padx=15, pady=5, sticky="ew")
            self.files_display.configure(state="disabled")

            btn_add_files = ctk.CTkButton(selection_frame, text="Ajouter des fichiers", command=self._pick_files)
            btn_add_files.grid(row=2, column=0, padx=(15, 5), pady=(5, 15), sticky="ew")

            btn_add_dir = ctk.CTkButton(selection_frame, text="Ajouter un dossier", command=self._pick_directory)
            btn_add_dir.grid(row=2, column=1, padx=5, pady=(5, 15), sticky="ew")

            btn_clear = ctk.CTkButton(
                selection_frame, text="Vider la liste", fg_color="#8B2635", hover_color="#6b1d29",
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
                text="Métadonnées manuelles (optionnel — laisse vide pour laisser l'auto-tagging décider)",
                font=FONT_LABEL,
            ).grid(row=0, column=0, columnspan=4, padx=15, pady=(15, 10), sticky="w")

            self.entry_courant = self._labeled_entry(meta_frame, "Courant artistique", row=1, col=0)
            self.entry_artiste = self._labeled_entry(meta_frame, "Artiste", row=1, col=1)
            self.entry_periode = self._labeled_entry(meta_frame, "Période", row=1, col=2)
            self.entry_technique = self._labeled_entry(meta_frame, "Technique", row=1, col=3)

            # --- Réglages d'auto-tagging pour CE lancement (adaptés au livre/texte en cours) ---
            autotag_settings_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
            autotag_settings_row.grid(row=2, column=0, columnspan=4, padx=15, pady=(5, 0), sticky="ew")

            granularity_container = ctk.CTkFrame(autotag_settings_row, fg_color="transparent")
            granularity_container.pack(side="left", padx=(0, 20))
            ctk.CTkLabel(
                granularity_container, text="Granularité auto-tagging",
                font=("Segoe UI", 10), text_color="gray70",
            ).pack(anchor="w")
            self.GRANULARITY_LABELS = {
                "Document (1 appel LLM, rapide)": "document",
                "Par chunk (1 appel/chunk, précis)": "chunk",
            }
            self.option_granularity = ctk.CTkOptionMenu(
                granularity_container,
                values=list(self.GRANULARITY_LABELS.keys()),
                command=self._on_granularity_changed,
                width=230,
            )
            self.option_granularity.set("Document (1 appel LLM, rapide)")
            self.option_granularity.pack(anchor="w")

            sample_container = ctk.CTkFrame(autotag_settings_row, fg_color="transparent")
            sample_container.pack(side="left")
            ctk.CTkLabel(
                sample_container, text="Échantillon document (caractères)",
                font=("Segoe UI", 10), text_color="gray70",
            ).pack(anchor="w")
            self.entry_sample_chars = ctk.CTkEntry(sample_container, width=100)
            self.entry_sample_chars.insert(0, str(DOCUMENT_AUTOTAG_SAMPLE_CHARS))
            self.entry_sample_chars.pack(anchor="w")

            self.label_granularity_hint = ctk.CTkLabel(
                autotag_settings_row,
                text="1 seul appel LLM sur le début du texte, appliqué à tous les chunks — "
                     "rapide, idéal pour un livre/texte homogène. Augmente l'échantillon si "
                     "le début ne reflète pas bien le contenu (technique/courant qui varie selon les chapitres).",
                font=("Segoe UI", 9), text_color="gray60", justify="left", wraplength=500,
            )
            self.label_granularity_hint.pack(side="left", padx=(20, 0))

            options_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
            options_row.grid(row=3, column=0, columnspan=4, padx=15, pady=(5, 15), sticky="w")

            self.var_autotag = ctk.BooleanVar(value=True)
            self.chk_autotag = ctk.CTkCheckBox(
                options_row, text="Activer l'auto-tagging via Ollama", variable=self.var_autotag
            )
            self.chk_autotag.pack(side="left", padx=(0, 20))

            self.var_force = ctk.BooleanVar(value=False)
            self.chk_force = ctk.CTkCheckBox(
                options_row, text="Forcer la ré-ingestion (ignore les fichiers déjà traités)", variable=self.var_force
            )
            self.chk_force.pack(side="left")

            # --- Bouton de lancement ---
            self.btn_run = ctk.CTkButton(
                parent, text="Lancer l'ingestion", font=FONT_LABEL, height=40,
                command=self._start_ingestion
            )
            self.btn_run.grid(row=2, column=0, pady=(5, 10), sticky="e")

        # --------------------------------------------------------
        # ONGLET ANALYSE D'ŒUVRE
        # --------------------------------------------------------

        def _build_analysis_tab(self, parent):
            # --- Sélection de l'image ---
            image_frame = ctk.CTkFrame(parent)
            image_frame.grid(row=0, column=0, padx=0, pady=(10, 10), sticky="ew")
            image_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(image_frame, text="Image de l'œuvre à analyser", font=FONT_LABEL).grid(
                row=0, column=0, columnspan=2, padx=15, pady=(15, 5), sticky="w"
            )

            self.image_path_label = ctk.CTkLabel(
                image_frame, text="(aucune image sélectionnée)", font=FONT_LOG, text_color="gray70", anchor="w"
            )
            self.image_path_label.grid(row=1, column=0, padx=15, pady=5, sticky="ew")

            btn_pick_image = ctk.CTkButton(image_frame, text="Choisir une image", command=self._pick_image)
            btn_pick_image.grid(row=1, column=1, padx=(5, 15), pady=5, sticky="e")

            self.image_preview_label = ctk.CTkLabel(
                image_frame, text="(aucun aperçu)", text_color="gray60", font=FONT_LOG,
                fg_color="gray17", corner_radius=6, width=260, height=260
            )
            self.image_preview_label.grid(row=2, column=0, columnspan=2, padx=15, pady=(0, 15))

            # --- Métadonnées de l'œuvre ---
            meta_frame = ctk.CTkFrame(parent)
            meta_frame.grid(row=1, column=0, padx=0, pady=10, sticky="ew")
            for i in range(4):
                meta_frame.grid_columnconfigure(i, weight=1)

            ctk.CTkLabel(
                meta_frame,
                text="Métadonnées de l'œuvre (Artiste ou Œuvre requis, le reste est optionnel)",
                font=FONT_LABEL,
            ).grid(row=0, column=0, columnspan=4, padx=15, pady=(15, 10), sticky="w")

            self.entry_artiste_a = self._labeled_entry(meta_frame, "Artiste", row=1, col=0)
            self.entry_oeuvre_a = self._labeled_entry(meta_frame, "Œuvre (titre)", row=1, col=1)
            self.entry_courant_a = self._labeled_entry(meta_frame, "Courant artistique", row=1, col=2)
            self.entry_periode_a = self._labeled_entry(meta_frame, "Période", row=1, col=3)
            self.entry_technique_a = self._labeled_entry(meta_frame, "Technique", row=2, col=0)

            # --- Mode de saisie : vision Ollama ou texte manuel ---
            self.analysis_mode = ctk.StringVar(value="vision")
            mode_row = ctk.CTkFrame(meta_frame, fg_color="transparent")
            mode_row.grid(row=2, column=1, columnspan=3, padx=15, pady=5, sticky="w")
            ctk.CTkRadioButton(
                mode_row, text="Générer via Ollama (vision)", variable=self.analysis_mode, value="vision"
            ).pack(side="left", padx=(0, 20))
            ctk.CTkRadioButton(
                mode_row, text="Saisie manuelle", variable=self.analysis_mode, value="manuel"
            ).pack(side="left")

            # --- Zone de texte de l'analyse (générée ou tapée à la main) ---
            text_frame = ctk.CTkFrame(parent)
            text_frame.grid(row=2, column=0, padx=0, pady=10, sticky="nsew")
            text_frame.grid_columnconfigure(0, weight=1)
            parent.grid_rowconfigure(2, weight=1)

            ctk.CTkLabel(
                text_frame,
                text="Texte de l'analyse (édition libre — généré par vision ou écrit à la main)",
                font=FONT_LABEL,
            ).grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")

            self.analysis_text_box = ctk.CTkTextbox(text_frame, font=FONT_LOG, height=160)
            self.analysis_text_box.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")

            # --- Boutons d'action ---
            btn_row = ctk.CTkFrame(parent, fg_color="transparent")
            btn_row.grid(row=3, column=0, pady=(5, 10), sticky="ew")
            btn_row.grid_columnconfigure(0, weight=1)

            self.btn_generate_analysis = ctk.CTkButton(
                btn_row, text="Générer l'analyse (vision)", font=FONT_LABEL,
                command=self._start_analysis_generation
            )
            self.btn_generate_analysis.grid(row=0, column=1, padx=(0, 10), sticky="e")

            self.btn_ingest_analysis = ctk.CTkButton(
                btn_row, text="Ingérer cette analyse", font=FONT_LABEL, height=40,
                command=self._start_analysis_ingestion
            )
            self.btn_ingest_analysis.grid(row=0, column=2, sticky="e")

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
                chat_frame, text="Discussion avec l'expert en arts graphiques", font=FONT_LABEL
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
                text="Éléments détectés dans la réponse (clique pour chercher sur le web)",
                font=FONT_LABEL,
            ).pack(side="left")
            ctk.CTkButton(
                chat_items_header, text="Tout rechercher", width=130, height=26,
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
                text="Forcer la génération d'un prompt image (JSON, 10 clés)",
                variable=self.var_force_image_prompt,
                font=("Segoe UI", 10),
            ).grid(row=1, column=0, padx=15, pady=(0, 15), sticky="w")

            self.btn_history = ctk.CTkButton(
                input_frame, text="Historique", font=FONT_LABEL, height=35, width=120,
                fg_color="gray30", hover_color="gray20", command=self._open_history_window,
            )
            self.btn_history.grid(row=1, column=1, padx=(5, 5), pady=(0, 15), sticky="e")

            self.btn_new_chat = ctk.CTkButton(
                input_frame, text="Nouvelle conversation", font=FONT_LABEL, height=35, width=170,
                fg_color="gray30", hover_color="gray20", command=self._reset_chat,
            )
            self.btn_new_chat.grid(row=1, column=2, padx=(5, 5), pady=(0, 15), sticky="e")

            self.btn_search = ctk.CTkButton(
                input_frame, text="Envoyer", font=FONT_LABEL, height=35, width=120,
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
                self._append_log("[!] Le message est vide.")
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
            self._append_chat("Toi", question)
            self.entry_question.delete("1.0", "end")

            self.active_job = "rag_query"
            self.is_running = True
            self.btn_search.configure(state="disabled", text="Réflexion...")
            self.progress_bar.start()
            self.status_label.configure(text="Analyse du message et recherche dans la base de connaissances...")

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
                        print("[...] Génération du titre de la conversation...")
                        title = generate_conversation_title(first_question)
                        update_conversation_title(self.history_conn, conversation_id, title)
                        print(f"[OK] Titre : {title}")

                    if not check_ollama_models():
                        print("[ABANDON] Corrige les modèles manquants ci-dessus avant de relancer.")
                        self.result_queue.put(("assistant_start", None))
                        self.result_queue.put(("token", "[ERREUR] Modèles Ollama manquants, voir le journal ci-dessous."))
                        self.result_queue.put(("assistant_end", None))
                        return

                    print("[...] Analyse du sujet du message...")
                    route = route_message(history_snapshot)
                    if force_image_prompt:
                        route["wants_image_prompt"] = True
                        route["art_related"] = True

                    chunks = []
                    if route["art_related"]:
                        print(f"[...] Sujet artistique détecté -> recherche RAG : \"{route['search_query']}\"")
                        chunks = search_chunks(
                            route["search_query"] or history_snapshot[-1]["content"], top_k=TOP_K_DEFAULT
                        )
                        print(f"[OK] {len(chunks)} extrait(s) trouvé(s)" if chunks
                              else "[!] Aucun extrait pertinent trouvé, réponse basée sur l'expertise générale")
                    else:
                        print("[...] Message hors du champ artistique, pas de recherche RAG pour ce tour")

                    stats = {}
                    self.result_queue.put(("assistant_start", None))

                    if route["wants_image_prompt"]:
                        print("[...] Génération du prompt image structuré (JSON, 10 clés)...")
                        result = generate_image_prompt(history_snapshot, chunks, stats=stats)
                        if "_error" in result:
                            text = f"[ERREUR] {result['_error']}"
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
                            print(f"[i] {len(list_items)} élément(s) de liste détecté(s) dans la réponse")
                        self.result_queue.put(("chat_list_items", list_items))

                    self.result_queue.put(("assistant_end", None))
                    print(format_llm_stats(stats))
            except Exception as e:
                self.result_queue.put(("token", f"[ERREUR] {e}"))
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
            self._append_log("[i] Nouvelle conversation démarrée.")

        def _open_history_window(self):
            """Ouvre une fenêtre listant les conversations enregistrées (titre + date), avec
            possibilité de reprendre ou de supprimer chacune d'entre elles."""
            if self.is_running:
                return

            win = ctk.CTkToplevel(self)
            win.title("Historique des conversations")
            win.geometry("560x480")
            win.transient(self)
            win.grab_set()

            ctk.CTkLabel(
                win, text="Conversations précédentes", font=FONT_LABEL
            ).pack(padx=15, pady=(15, 10), anchor="w")

            rows = list_conversations(self.history_conn)

            if not rows:
                ctk.CTkLabel(
                    win, text="Aucune conversation enregistrée pour l'instant.", text_color="gray60"
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
                    row, text="Reprendre", width=90, height=28,
                    command=lambda cid=conversation_id, w=win: self._load_conversation(cid, w),
                ).grid(row=0, column=1, padx=(5, 5), pady=8)

                ctk.CTkButton(
                    row, text="Régénérer titre", width=120, height=28,
                    fg_color="gray30", hover_color="gray20",
                    command=lambda cid=conversation_id, r=row: self._regenerate_conversation_title(cid, r),
                ).grid(row=0, column=2, padx=(0, 5), pady=8)

                ctk.CTkButton(
                    row, text="Supprimer", width=90, height=28,
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
                speaker = "Toi" if msg["role"] == "user" else "Expert"
                self._append_chat(speaker, msg["content"])

            self._show_chat_list_items([])
            self._append_log(f"[i] Conversation chargée ({len(messages)} message(s)).")

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
            label.configure(text=f"{current_title}\n(régénération du titre en cours...)")
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
                    error_msg = "conversation vide"
            except Exception as e:
                # Filet de sécurité : sans ce try/except, une exception ici tue le thread en
                # silence et self.after() n'est jamais appelé -> l'UI reste bloquée sur "en
                # cours..." sans jamais refléter ni le succès ni l'échec de l'appel LLM.
                error_msg = f"{type(e).__name__}: {e}"
                print(f"[!] Échec de la régénération du titre : {error_msg}")

            self.after(
                0, lambda: self._on_title_regenerated(row, label, buttons, new_title, date_part, error_msg)
            )

        def _on_title_regenerated(self, row, label, buttons, new_title, date_part, error_msg=None):
            if not row.winfo_exists():
                return
            if error_msg:
                label.configure(text=f"{new_title}\n{date_part}  [échec régénération : {error_msg}]")
                self._append_log(f"[!] Régénération du titre échouée : {error_msg}")
            else:
                label.configure(text=f"{new_title}\n{date_part}")
                self._append_log(f"[i] Titre régénéré : {new_title}")
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
                    row, text="Rechercher", width=100, height=26,
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
                self._append_log("[!] Une autre opération est en cours, réessaie dans un instant.")
                return
            self.tabview.set("Recherche web")
            self.entry_web_query.delete(0, "end")
            self.entry_web_query.insert(0, item)
            self._start_web_search()

        def _search_all_chat_items(self):
            """Lance une recherche web groupée pour tous les éléments détectés dans la dernière réponse."""
            if self.is_running:
                self._append_log("[!] Une autre opération est en cours, réessaie dans un instant.")
                return
            if not self.chat_list_items:
                return

            sources = self._selected_web_sources()
            items = list(self.chat_list_items)

            self.tabview.set("Recherche web")
            self.active_job = "web_search"
            self.is_running = True
            self.btn_web_search.configure(state="disabled", text="Recherche...")
            self.progress_bar.start()
            self.status_label.configure(text=f"Recherche groupée de {len(items)} élément(s)...")

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
                        print(f"[...] ({i}/{len(items)}) Recherche : \"{item}\"")
                        for result in web_search_all(item, sources=tuple(sources), max_results_per_source=3):
                            if result["url"] in seen_urls:
                                continue
                            seen_urls.add(result["url"])
                            result["_query"] = item
                            merged.append(result)
                    print(f"[OK] {len(merged)} résultat(s) unique(s) au total pour {len(items)} élément(s)")
                    self.result_queue.put(("web_results", merged))
            except Exception as e:
                print(f"[X] Erreur pendant la recherche groupée : {e}")
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
                placeholder_text="Ex: cubisme analytique, techniques du sfumato, Egon Schiele...",
            )
            self.entry_web_query.grid(row=0, column=0, padx=(15, 10), pady=15, sticky="ew")

            self.btn_web_search = ctk.CTkButton(
                search_frame, text="Rechercher", font=FONT_LABEL, height=35, width=140,
                command=self._start_web_search,
            )
            self.btn_web_search.grid(row=0, column=1, padx=(0, 15), pady=15)

            # --- Sources + options ---
            sources_frame = ctk.CTkFrame(parent)
            sources_frame.grid(row=1, column=0, padx=0, pady=(0, 10), sticky="ew")

            ctk.CTkLabel(sources_frame, text="Sources :", font=FONT_LABEL).grid(
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
            ctk.CTkCheckBox(sources_frame, text="Recherche web (FR/EN)", variable=self.var_source_ddgs).grid(
                row=0, column=3, padx=10, pady=15
            )
            self.var_web_force = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(sources_frame, text="Forcer la ré-ingestion", variable=self.var_web_force).grid(
                row=0, column=4, padx=(20, 15), pady=15
            )

            # --- Résultats (revue manuelle, sélection par case à cocher) ---
            results_frame = ctk.CTkFrame(parent)
            results_frame.grid(row=2, column=0, padx=0, pady=(0, 10), sticky="nsew")
            results_frame.grid_columnconfigure(0, weight=1)
            results_frame.grid_rowconfigure(1, weight=1)

            header_row = ctk.CTkFrame(results_frame, fg_color="transparent")
            header_row.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="ew")
            ctk.CTkLabel(header_row, text="Résultats (coche ce que tu veux ingérer)", font=FONT_LABEL).pack(
                side="left"
            )
            ctk.CTkButton(
                header_row, text="Tout décocher", width=110, height=26,
                fg_color="gray30", hover_color="gray20",
                command=lambda: self._toggle_all_web_results(False),
            ).pack(side="right", padx=(5, 0))
            ctk.CTkButton(
                header_row, text="Tout cocher", width=100, height=26,
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
            ctk.CTkLabel(preview_frame, text="Aperçu (clique 'Aperçu' sur un résultat)", font=FONT_LABEL).grid(
                row=0, column=0, padx=15, pady=(15, 5), sticky="w"
            )
            self.web_preview_box = ctk.CTkTextbox(preview_frame, font=FONT_LOG, wrap="word")
            self.web_preview_box.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
            self.web_preview_box.configure(state="disabled")

            # --- Action d'ingestion ---
            action_frame = ctk.CTkFrame(parent, fg_color="transparent")
            action_frame.grid(row=4, column=0, padx=0, pady=(0, 10), sticky="ew")
            self.btn_web_ingest = ctk.CTkButton(
                action_frame, text="Ingérer la sélection", font=FONT_LABEL, height=38,
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
                    self.web_results_scroll, text="Aucun résultat pour cette recherche.", font=("Segoe UI", 11)
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
                    btn_col, text="Aperçu", width=110, height=28,
                    fg_color="gray30", hover_color="gray20",
                    command=lambda r=result: self._preview_web_result(r),
                ).pack(pady=(0, 4))

                if is_met:
                    ctk.CTkButton(
                        btn_col, text="Envoyer vers analyse", width=110, height=28,
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
            self.btn_web_search.configure(state="disabled", text="Recherche...")
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
                    print(f"[OK] {len(results)} résultat(s) unique(s) trouvé(s)")
                    self.result_queue.put(("web_results", results))
            except Exception as e:
                print(f"[X] Erreur pendant la recherche web : {e}")
                self.result_queue.put(("web_results", []))

        def _preview_web_result(self, result):
            if self.is_running:
                self._append_log("[!] Une autre opération est en cours, réessaie dans un instant.")
                return

            self._set_web_preview("Récupération en cours...")

            self.active_job = "web_preview"
            self.is_running = True
            self.btn_web_ingest.configure(state="disabled")
            self.progress_bar.start()
            self.status_label.configure(text=f"Récupération de : {result['title']}...")

            self.worker_thread = threading.Thread(
                target=self._run_web_preview_worker, args=(result,), daemon=True
            )
            self.worker_thread.start()

        def _run_web_preview_worker(self, result):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    print(f"[...] Récupération du contenu : {result['url']}")
                    content = fetch_web_content(result)
                    if "_error" in content:
                        print(f"[X] {content['_error']}")
                        self.result_queue.put(("web_preview", f"[ERREUR] {content['_error']}"))
                    else:
                        print(f"[OK] {len(content['text'])} caractères extraits")
                        preview_text = f"{content['title']}\n{result['url']}\n\n{content['text'][:5000]}"
                        if len(content["text"]) > 5000:
                            preview_text += (
                                "\n\n[...] (aperçu tronqué à 5000 caractères ; le texte complet "
                                "sera utilisé si tu ingères cette source)"
                            )
                        self.result_queue.put(("web_preview", preview_text))
            except Exception as e:
                self.result_queue.put(("web_preview", f"[ERREUR] {e}"))

        # --------------------------------------------------------
        # ENVOI D'UN RÉSULTAT MET MUSEUM VERS L'ONGLET "ANALYSE D'ŒUVRE"
        # --------------------------------------------------------

        def _send_met_result_to_analysis(self, result):
            if self.is_running:
                self._append_log("[!] Une autre opération est en cours, réessaie dans un instant.")
                return

            obj = result.get("_object_data") or {}
            image_url = obj.get("primaryImage") or obj.get("primaryImageSmall")
            if not image_url:
                self._append_log("[!] Cette œuvre Met Museum n'a pas d'image exploitable.")
                return

            self.active_job = "met_to_analysis"
            self.is_running = True
            self.progress_bar.start()
            self.status_label.configure(text=f"Récupération de l'image : {obj.get('title', '')}...")

            self.worker_thread = threading.Thread(
                target=self._run_send_met_to_analysis_worker, args=(image_url, obj), daemon=True
            )
            self.worker_thread.start()

        def _run_send_met_to_analysis_worker(self, image_url: str, obj: dict):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    print(f"[...] Téléchargement de l'image : {image_url}")
                    data = download_image_bytes(image_url)
                    if not data:
                        print("[X] Échec du téléchargement de l'image.")
                        return
                    suffix = Path(urllib.parse.urlparse(image_url).path).suffix or ".jpg"
                    image_path = save_image_bytes_to_temp(data, suffix=suffix)
                    print(f"[OK] Image enregistrée temporairement : {image_path}")
                    self.result_queue.put(("met_to_analysis", {"image_path": image_path, "obj": obj}))
            except Exception as e:
                print(f"[X] Erreur pendant l'envoi vers l'analyse : {e}")

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
            self.tabview.set("Analyse d'œuvre")
            self._append_log(f"[OK] Œuvre envoyée vers l'analyse : {obj.get('title', '(sans titre)')}")

        def _start_web_ingest(self):
            if self.is_running:
                return
            selected = [r for r, v in zip(self.web_results, self.web_result_vars) if v.get()]
            if not selected:
                self._append_log("[!] Aucun résultat coché pour l'ingestion.")
                return

            force = self.var_web_force.get()

            self.active_job = "web_ingest"
            self.is_running = True
            self.btn_web_ingest.configure(state="disabled", text="Ingestion en cours...")
            self.progress_bar.start()
            self.status_label.configure(text=f"Ingestion de {len(selected)} source(s) web...")

            self.worker_thread = threading.Thread(
                target=self._run_web_ingest_worker, args=(selected, force), daemon=True
            )
            self.worker_thread.start()

        def _run_web_ingest_worker(self, selected, force):
            writer = QueueWriter(self.log_queue)
            try:
                with contextlib.redirect_stdout(writer):
                    if not check_ollama_models():
                        print("[ABANDON] Corrige les modèles manquants ci-dessus avant de relancer.")
                        return

                    sqlite_conn = init_sqlite(DB_PATH)
                    chroma_collection = init_chroma()

                    for i, result in enumerate(selected, start=1):
                        print(f"\n[{i}/{len(selected)}] {result['title']} ({result['url']})")
                        try:
                            outcome = ingest_web_result(result, sqlite_conn, chroma_collection, force=force)
                        except KeyboardInterrupt:
                            print("[!] Interruption demandée, arrêt de l'ingestion web.")
                            break
                        except Exception as e:
                            print(f"  [X] Erreur inattendue : {e}")
                            continue
                        tag = {"ok": "[OK]", "skip": "[SKIP]", "partial": "[!]", "error": "[X]"}.get(
                            outcome["status"], "[?]"
                        )
                        print(f"  {tag} {outcome['message']}")

                    print(f"\n[TERMINÉ] {len(selected)} source(s) traitée(s).")
            except Exception as e:
                print(f"[X] Erreur pendant l'ingestion web : {e}")

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

                    label_text = label + ("  (redémarrage requis)" if needs_restart else "")
                    lbl = ctk.CTkLabel(field_row, text=label_text, font=("Segoe UI", 12), width=280, anchor="w")
                    lbl.grid(row=0, column=0, padx=(0, 10), pady=(2, 0), sticky="w")

                    if key == "AUTOTAG_GRANULARITY":
                        widget = ctk.CTkOptionMenu(field_row, values=["document", "chunk"])
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

            # --- Modèles Ollama disponibles (aide au remplissage) ---
            models_frame = ctk.CTkFrame(scroll)
            models_frame.grid(row=row, column=0, padx=5, pady=(0, 15), sticky="ew")
            models_frame.grid_columnconfigure(0, weight=1)
            row += 1

            ctk.CTkLabel(models_frame, text="Modèles Ollama détectés", font=FONT_LABEL).grid(
                row=0, column=0, padx=15, pady=(12, 8), sticky="w"
            )
            self.settings_models_box = ctk.CTkTextbox(models_frame, font=FONT_LOG, height=90)
            self.settings_models_box.grid(row=1, column=0, padx=15, pady=(0, 10), sticky="ew")
            self.settings_models_box.configure(state="disabled")
            ctk.CTkButton(
                models_frame, text="Lister les modèles installés (Ollama)", command=self._settings_list_models
            ).grid(row=2, column=0, padx=15, pady=(0, 15), sticky="w")

            # --- Actions ---
            action_row = ctk.CTkFrame(parent, fg_color="transparent")
            action_row.grid(row=1, column=0, padx=5, pady=15, sticky="ew")

            ctk.CTkButton(
                action_row, text="Enregistrer les paramètres", command=self._settings_save
            ).pack(side="left", padx=(10, 10))
            ctk.CTkButton(
                action_row, text="Réinitialiser les valeurs par défaut", fg_color="gray30",
                command=self._settings_reset_defaults
            ).pack(side="left")

            self.settings_status_label = ctk.CTkLabel(parent, text="", font=("Segoe UI", 11), text_color="gray70")
            self.settings_status_label.grid(row=2, column=0, padx=15, pady=(0, 10), sticky="w")

        def _settings_list_models(self):
            def worker():
                try:
                    resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
                    resp.raise_for_status()
                    names = sorted(m["name"] for m in resp.json().get("models", []))
                    text = "\n".join(names) if names else "(aucun modèle trouvé)"
                except Exception as e:
                    text = f"Erreur : {e}"

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
                    text=f"Enregistré avec des erreurs sur : {', '.join(errors)} (valeurs ignorées).",
                    text_color="orange",
                )
            else:
                self.settings_status_label.configure(
                    text="Paramètres enregistrés dans art_rag_config.json.", text_color="lightgreen"
                )

        def _settings_reset_defaults(self):
            if CONFIG_PATH.exists():
                try:
                    CONFIG_PATH.unlink()
                except Exception as e:
                    self.settings_status_label.configure(text=f"Erreur : {e}", text_color="orange")
                    return
            self.settings_status_label.configure(
                text="art_rag_config.json supprimé — relance l'application pour revenir aux valeurs par défaut.",
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
                title="Choisir des documents",
                filetypes=[("Documents supportés", "*.pdf *.txt *.md"), ("Tous les fichiers", "*.*")],
            )
            for p in paths:
                path_obj = Path(p)
                if path_obj not in self.selected_files:
                    self.selected_files.append(path_obj)
            self._refresh_files_display()

        def _pick_directory(self):
            directory = filedialog.askdirectory(title="Choisir un dossier à ingérer récursivement")
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
                self.files_display.insert("end", "(aucun fichier sélectionné)")
            else:
                for f in self.selected_files:
                    self.files_display.insert("end", f"{f}\n")
            self.files_display.configure(state="disabled")

        def _pick_image(self):
            path = filedialog.askopenfilename(
                title="Choisir une image d'œuvre",
                filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp"), ("Tous les fichiers", "*.*")],
            )
            if not path:
                return
            self.selected_image = Path(path)
            self.image_path_label.configure(text=str(self.selected_image), text_color="white")
            self._refresh_analysis_image_preview()

        def _refresh_analysis_image_preview(self):
            """Charge self.selected_image et l'affiche en miniature dans l'onglet Analyse d'œuvre."""
            if self.selected_image is None:
                self._append_log("[!] Aperçu : aucune image sélectionnée (selected_image est None).")
                return
            if Image is None:
                self._append_log(
                    "[!] Aperçu impossible : le module PIL/Pillow n'a pas pu être importé "
                    f"('from PIL import Image' a échoué au démarrage) — détail : {PIL_IMPORT_ERROR}. "
                    "Vérifie 'pip show pillow' avec le MÊME interpréteur Python que celui qui lance ce script."
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
                self._append_log(f"[OK] Aperçu chargé : {self.selected_image.name} ({preview.size[0]}x{preview.size[1]})")
            except Exception as e:
                self.image_preview_label.configure(image=None, text="(aperçu indisponible)")
                self._append_log(f"[!] Aperçu de l'image impossible : {e}")

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
                        self._append_chat_raw("Expert : ")
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
                    text="1 appel LLM par chunk : plus lent, mais capture les variations de "
                         "courant/technique à l'intérieur d'un même livre (ex: ouvrage qui "
                         "couvre plusieurs mouvements ou plusieurs artistes)."
                )
            else:
                self.entry_sample_chars.configure(state="normal")
                self.label_granularity_hint.configure(
                    text="1 seul appel LLM sur le début du texte, appliqué à tous les chunks — "
                         "rapide, idéal pour un livre/texte homogène. Augmente l'échantillon si "
                         "le début ne reflète pas bien le contenu (technique/courant qui varie selon les chapitres)."
                )

        def _start_ingestion(self):
            if self.is_running:
                return
            if not self.selected_files:
                self._append_log("[!] Aucun fichier sélectionné.")
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
                    f"[!] Taille d'échantillon invalide ('{sample_chars_raw}'), "
                    f"utilisation de la valeur par défaut ({DOCUMENT_AUTOTAG_SAMPLE_CHARS})."
                )
                document_sample_chars = DOCUMENT_AUTOTAG_SAMPLE_CHARS

            self.active_job = "documents"
            self.is_running = True
            self.btn_run.configure(state="disabled", text="Ingestion en cours...")
            self.progress_bar.start()
            self.status_label.configure(text=f"Traitement de {len(files)} fichier(s)...")

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
                    print(f"=== {len(files)} fichier(s) à traiter ===\n")
                    sqlite_conn = init_sqlite(DB_PATH)
                    chroma_collection = init_chroma()

                    for f in files:
                        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                            print(f"[SKIP] Extension non supportée : {f.name}")
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

                    print("=== Ingestion terminée ===")
            except Exception as e:
                self.log_queue.put(f"[ERREUR FATALE] {e}")

        # --------------------------------------------------------
        # GÉNÉRATION D'ANALYSE D'ŒUVRE VIA VISION (dans un thread séparé)
        # --------------------------------------------------------

        def _start_analysis_generation(self):
            if self.is_running:
                return
            if self.analysis_mode.get() != "vision":
                self._append_log("[!] Passe en mode 'Générer via Ollama (vision)' pour utiliser ce bouton.")
                return
            if self.selected_image is None:
                self._append_log("[!] Choisis d'abord une image d'œuvre.")
                return

            artiste = self.entry_artiste_a.get().strip()
            oeuvre = self.entry_oeuvre_a.get().strip()
            courant = self.entry_courant_a.get().strip()

            self._last_analysis_text = None
            self.active_job = "generation"
            self.is_running = True
            self.btn_generate_analysis.configure(state="disabled", text="Génération en cours...")
            self.btn_ingest_analysis.configure(state="disabled")
            self.progress_bar.start()
            self.status_label.configure(text="Analyse de l'image via le modèle vision...")

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
                        print("[ABANDON] Corrige les modèles manquants ci-dessus avant de relancer.")
                        return
                    print(f"[...] Génération de l'analyse via {OLLAMA_VISION_MODEL}...")
                    result = analyze_artwork_vision(image_path, artiste=artiste, oeuvre=oeuvre, courant=courant)
                    if result:
                        self._last_analysis_text = result
                        print("[OK] Analyse générée. Relis/édite le texte puis clique sur 'Ingérer cette analyse'.")
                    else:
                        print("[X] Échec de la génération de l'analyse.")
            except Exception as e:
                self.log_queue.put(f"[ERREUR FATALE] {e}")

        # --------------------------------------------------------
        # INGESTION DE L'ANALYSE D'ŒUVRE (dans un thread séparé)
        # --------------------------------------------------------

        def _start_analysis_ingestion(self):
            if self.is_running:
                return
            text = self.analysis_text_box.get("1.0", "end").strip()
            if not text:
                self._append_log("[!] Le champ d'analyse est vide (génère-le ou écris-le à la main).")
                return

            artiste = self.entry_artiste_a.get().strip()
            oeuvre = self.entry_oeuvre_a.get().strip()
            if not artiste and not oeuvre:
                self._append_log("[!] Renseigne au moins l'Artiste ou l'Œuvre avant d'ingérer.")
                return

            courant = self.entry_courant_a.get().strip()
            periode = self.entry_periode_a.get().strip()
            technique = self.entry_technique_a.get().strip()
            source_confidence = "manuel" if self.analysis_mode.get() == "manuel" else "vision"

            self.active_job = "ingest_analysis"
            self.is_running = True
            self.btn_ingest_analysis.configure(state="disabled", text="Ingestion en cours...")
            self.btn_generate_analysis.configure(state="disabled")
            self.progress_bar.start()
            self.status_label.configure(text=f"Ingestion de l'analyse ({artiste or oeuvre})...")

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
                self.log_queue.put(f"[ERREUR FATALE] {e}")

        # --------------------------------------------------------
        # FIN DE JOB (commun aux workers)
        # --------------------------------------------------------

        def _on_job_finished(self):
            self.is_running = False
            self.progress_bar.stop()
            self.progress_bar.set(0)

            if self.active_job == "documents":
                self.btn_run.configure(state="normal", text="Lancer l'ingestion")
            elif self.active_job == "generation":
                self.btn_generate_analysis.configure(state="normal", text="Générer l'analyse (vision)")
                self.btn_ingest_analysis.configure(state="normal")
                if self._last_analysis_text:
                    self.analysis_text_box.delete("1.0", "end")
                    self.analysis_text_box.insert("1.0", self._last_analysis_text)
            elif self.active_job == "ingest_analysis":
                self.btn_ingest_analysis.configure(state="normal", text="Ingérer cette analyse")
                self.btn_generate_analysis.configure(state="normal")
            elif self.active_job == "rag_query":
                self.btn_search.configure(state="normal", text="Envoyer")
            elif self.active_job == "web_search":
                self.btn_web_search.configure(state="normal", text="Rechercher")
            elif self.active_job == "web_preview":
                self.btn_web_ingest.configure(state="normal")
            elif self.active_job == "web_ingest":
                self.btn_web_ingest.configure(state="normal", text="Ingérer la sélection")

            self.status_label.configure(text="Terminé.")
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
