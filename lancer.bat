@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Lancez d'abord installer.bat
    pause
    exit /b 1
)
echo Demarrage de l'application (la fenetre s'ouvre dans quelques secondes)...
".venv\Scripts\python.exe" app.py %*
if errorlevel 1 pause
