@echo off
setlocal enabledelayedexpansion
title Art RAG - Ingestion de documents
cd /d "%~dp0"

echo ============================================================
echo   Art RAG - Verification de l'environnement
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 1. Verifier que Python est installe et accessible
REM ------------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou n'est pas dans le PATH.
    echo          Installe Python 3.10+ depuis https://www.python.org/downloads/
    echo          et coche "Add Python to PATH" pendant l'installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VERSION=%%v
echo [OK] Python detecte : %PY_VERSION%
echo.

REM ------------------------------------------------------------
REM 2. Creer le venv s'il n'existe pas
REM ------------------------------------------------------------
set VENV_DIR=venv
set VENV_PY=%VENV_DIR%\Scripts\python.exe
set VENV_PIP=%VENV_DIR%\Scripts\pip.exe

if not exist "%VENV_PY%" (
    echo [...] Environnement virtuel absent, creation en cours...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERREUR] Echec de la creation du venv.
        pause
        exit /b 1
    )
    echo [OK] Venv cree dans .\%VENV_DIR%
) else (
    echo [OK] Venv existant detecte
)
echo.

REM ------------------------------------------------------------
REM 3. Mettre a jour pip (silencieux)
REM ------------------------------------------------------------
echo [...] Mise a jour de pip...
"%VENV_PY%" -m pip install --upgrade pip -q
echo.

REM ------------------------------------------------------------
REM 4. Verifier / installer chaque dependance requise
REM ------------------------------------------------------------
echo ============================================================
echo   Verification des dependances
echo ============================================================
echo.

call :check_install customtkinter customtkinter
call :check_install pypdf pypdf
call :check_install requests requests
call :check_install chromadb chromadb
call :check_install trafilatura trafilatura
call :check_install ddgs ddgs
call :check_install pillow pillow

echo.

REM ------------------------------------------------------------
REM 5. Verification optionnelle : Ollama accessible ?
REM ------------------------------------------------------------
echo ============================================================
echo   Verification d'Ollama (optionnel)
echo ============================================================
echo.
"%VENV_PY%" -c "import requests; requests.get('http://localhost:11434', timeout=2)" >nul 2>nul
if errorlevel 1 (
    echo [ATTENTION] Ollama ne semble pas repondre sur localhost:11434.
    echo             L'ingestion fonctionnera mais l'auto-tagging et les
    echo             embeddings echoueront tant qu'Ollama n'est pas lance.
    echo             Lance "ollama serve" si besoin.
) else (
    echo [OK] Ollama repond sur localhost:11434
)
echo.

REM ------------------------------------------------------------
REM 6. Verifier que le script principal est bien present
REM ------------------------------------------------------------
if not exist "art_rag.py" (
    echo [ERREUR] art_rag.py introuvable dans ce dossier.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM 7. Lancer l'interface graphique
REM ------------------------------------------------------------
echo ============================================================
echo   Lancement de l'application
echo ============================================================
echo.
"%VENV_PY%" art_rag.py

if errorlevel 1 (
    echo.
    echo [ERREUR] L'application s'est terminee avec une erreur.
    pause
)

exit /b 0

REM ============================================================
REM Fonction : verifie si un module Python est importable dans le
REM venv, l'installe via pip sinon.
REM   %1 = nom du module a importer (ex: customtkinter)
REM   %2 = nom du package pip a installer (ex: customtkinter)
REM ============================================================
:check_install
"%VENV_PY%" -c "import %~1" >nul 2>nul
if errorlevel 1 (
    echo [...] %~2 absent, installation en cours...
    "%VENV_PIP%" install %~2 -q
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation de %~2
    ) else (
        echo [OK] %~2 installe avec succes
    )
) else (
    echo [OK] %~2 deja present
)
exit /b 0
