@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Installation du projet langue des signes
echo ============================================

set "PY="
py -3.12 --version >nul 2>nul && set "PY=py -3.12"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "C:\Program Files\Python312\python.exe" set "PY=C:\Program Files\Python312\python.exe"

if not defined PY (
    echo.
    echo [1/3] Python 3.12 introuvable : installation avec winget...
    winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
)
if not defined PY (
    echo.
    echo ERREUR : Python 3.12 n'a pas pu etre installe automatiquement.
    echo Installez-le depuis python.org puis relancez installer.bat
    pause
    exit /b 1
)
echo [1/3] Python trouve : %PY%

if not exist ".venv\Scripts\python.exe" (
    echo [2/3] Creation de l'environnement .venv...
    "%PY%" -m venv .venv 2>nul || %PY% -m venv .venv
)
echo [3/3] Installation des bibliotheques (quelques minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERREUR pendant l'installation des bibliotheques.
    pause
    exit /b 1
)
echo.
echo ============================================
echo   INSTALLATION TERMINEE - lancez lancer.bat
echo ============================================
pause
