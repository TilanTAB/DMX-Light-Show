@echo off
title DMX Light Show - Launcher
color 0B

echo.
echo  ========================================================
echo     DMX LIGHT SHOW - AI-Powered Concert Lighting Engine
echo  ========================================================
echo.

:: ------------------------------------------------
:: PRE-FLIGHT CHECKS
:: ------------------------------------------------

:: Check Python venv exists
if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] Python virtual environment not found!
    echo  Run:  python -m venv .venv
    echo        .venv\Scripts\pip install fastapi uvicorn pydantic python-dotenv numpy pyaudiowpatch pyusb requests
    pause
    exit /b 1
)

:: Check .env exists
if not exist ".env" (
    echo  [ERROR] .env file not found!
    echo  Copy .env.example to .env and fill in your Azure credentials.
    pause
    exit /b 1
)

:: Check frontend node_modules
if not exist "frontend\node_modules" (
    echo  [INFO] Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
    echo.
)

echo  [1/3] Starting FastAPI backend on port 8000...
start /B "DMX-Backend" .venv\Scripts\python.exe app.py > nul 2>&1

:: Give the backend a moment to start
timeout /t 2 /nobreak > nul

echo  [2/3] Starting React frontend on port 5173...
start /B "DMX-Frontend" cmd /c "cd frontend && npx vite 2>&1 > nul"

:: Give Vite a moment to compile
timeout /t 3 /nobreak > nul

echo  [3/3] Opening browser...
start http://localhost:5173

echo.
echo  ========================================================
echo   RUNNING!
echo   Frontend:  http://localhost:5173
echo   API:       http://localhost:8000
echo   API Docs:  http://localhost:8000/docs
echo  ========================================================
echo.
echo   Press any key to STOP both servers and exit.
echo  ========================================================
pause > nul

echo.
echo  Shutting down...

:: Kill the Python backend (uvicorn)
taskkill /F /FI "WINDOWTITLE eq DMX-Backend" > nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a > nul 2>&1
)

:: Kill the Vite dev server
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /F /PID %%a > nul 2>&1
)

:: Kill any remaining node processes from vite
taskkill /F /IM "node.exe" /FI "WINDOWTITLE eq DMX-Frontend" > nul 2>&1

echo  [OK] All servers stopped. Goodbye!
timeout /t 2 /nobreak > nul
