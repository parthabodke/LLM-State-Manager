@echo off
echo ================================================
echo   Multi-LLM Router Startup Script
echo ================================================
echo.

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo ERROR: Could not activate virtual environment
    echo Please run: python -m venv venv
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Start backend in a new window
echo [2/3] Starting FastAPI backend on port 8000...
start "FastAPI Backend" cmd /k "venv\Scripts\activate && uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"
echo ✓ Backend starting in new window...
echo.

REM Wait for backend to start
echo [3/3] Waiting for backend to initialize (20 seconds)...
timeout /t 20 /nobreak >nul
echo ✓ Starting Streamlit frontend...
echo.

REM Start frontend (this will keep the window open)
streamlit run app.py

echo.
echo ================================================
echo   Application stopped
echo ================================================
pause