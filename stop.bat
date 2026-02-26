@echo off
title DMX Light Show - Shutdown
color 0C

echo.
echo  ========================================================
echo   DMX LIGHT SHOW - Stopping all servers...
echo  ========================================================
echo.

:: Kill FastAPI/Uvicorn (Python on port 8000)
echo  [1/2] Stopping backend (port 8000)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a > nul 2>&1
    echo        Killed PID %%a
)

:: Kill Vite dev server (Node on port 5173)
echo  [2/2] Stopping frontend (port 5173)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING') do (
    taskkill /F /PID %%a > nul 2>&1
    echo        Killed PID %%a
)

echo.
echo  [OK] All servers stopped.
timeout /t 2 /nobreak > nul
