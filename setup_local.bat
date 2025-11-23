@echo off
REM Local development startup script for FallGuard (Windows)

echo ==================================================
echo FallGuard - Local Development Setup (Windows)
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [X] Python is not installed or not in PATH
    echo Please install Python 3.10 or higher
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [OK] %PYTHON_VERSION% found

REM Check if virtual environment exists
if not exist "venv\" (
    echo.
    echo [*] Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo.
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo.
echo [*] Installing dependencies...
pip install -r requirements.txt -q

if %ERRORLEVEL% neq 0 (
    echo.
    echo [X] Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

REM Create necessary directories
echo.
echo [*] Creating necessary directories...
if not exist "data\" mkdir data
if not exist "models\" mkdir models
if not exist "uploads\" mkdir uploads
echo [OK] Directories created

REM Copy .env if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        echo.
        echo [*] Creating .env from .env.example...
        copy .env.example .env
        echo [!] Please edit .env with your configuration
    )
)

echo.
echo ==================================================
echo [OK] Setup complete!
echo ==================================================
echo.
echo To start the application, run:
echo   python main.py
echo.
echo Or with Gunicorn (like Render):
echo   gunicorn --timeout 120 --workers 1 main:app
echo.
echo Access at: http://localhost:5000
echo ==================================================
echo.
pause
