@echo off
:: ══════════════════════════════════════════════════════════════════════
:: setup.bat — Windows Setup Script
:: Double-click this file OR run it from VS Code terminal
:: ══════════════════════════════════════════════════════════════════════

echo.
echo  ============================================================
echo   Pneumonia Detection Dashboard — Windows Setup
echo  ============================================================
echo.

:: Check Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found.
    echo  Download Python 3.10 or 3.11 from: https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo  [1/4] Python found:
python --version
echo.

:: Create virtual environment
echo  [2/4] Creating virtual environment (.venv) ...
python -m venv .venv
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo  Done.
echo.

:: Activate and install
echo  [3/4] Installing packages (this takes 3-5 minutes) ...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo  [ERROR] Package installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo  Done.
echo.

:: Create models folder
if not exist "models" (
    mkdir models
    echo  [4/4] Created models\ folder.
) else (
    echo  [4/4] models\ folder already exists.
)

echo.
echo  ============================================================
echo   Setup complete!
echo  ============================================================
echo.
echo   NEXT STEPS:
echo   1. Copy your model file (.keras or .h5) into the models\ folder
echo   2. Run the dashboard:  run.bat
echo      OR in VS Code terminal:
echo        .venv\Scripts\activate
echo        python app.py
echo.
pause
