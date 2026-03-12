@echo off
title DMX Light Show - Build Script
color 0E

echo.
echo  ========================================================
echo     DMX LIGHT SHOW - PyInstaller Build
echo  ========================================================
echo.

:: ------------------------------------------------
:: PRE-FLIGHT CHECKS
:: ------------------------------------------------

:: Check Python venv
if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] Python virtual environment not found!
    echo  Run:  python -m venv .venv
    echo        .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Check PyInstaller is installed
.venv\Scripts\python.exe -c "import PyInstaller" 2>nul
if %ERRORLEVEL% neq 0 (
    echo  [INFO] Installing PyInstaller...
    .venv\Scripts\pip.exe install pyinstaller
    echo.
)

:: ------------------------------------------------
:: STEP 1: Build React frontend
:: ------------------------------------------------
echo  [1/4] Building React frontend...
if not exist "frontend\node_modules" (
    echo  [INFO] Installing frontend dependencies first...
    cd frontend
    call npm install
    cd ..
)
cd frontend
call npm run build
cd ..
if not exist "frontend\dist\index.html" (
    echo  [ERROR] Frontend build failed! No dist/index.html found.
    pause
    exit /b 1
)
echo  [OK] Frontend built.
echo.

:: ------------------------------------------------
:: STEP 2: Run PyInstaller
:: ------------------------------------------------
echo  [2/4] Running PyInstaller (this may take 2-5 minutes)...
.venv\Scripts\pyinstaller.exe --clean --noconfirm dmx_app.spec
if %ERRORLEVEL% neq 0 (
    echo  [ERROR] PyInstaller build failed! Check the output above.
    pause
    exit /b 1
)
echo  [OK] PyInstaller build complete.
echo.

:: ------------------------------------------------
:: STEP 3: Copy sidecar files that are NOT bundled
:: ------------------------------------------------
echo  [3/4] Copying sidecar files to dist...

:: ffmpeg.exe (~200MB - too large to embed in spec)
if exist "ffmpeg.exe" (
    echo    Copying ffmpeg.exe...
    copy /Y "ffmpeg.exe" "dist\dmx_app\ffmpeg.exe" >nul
) else (
    echo    [WARN] ffmpeg.exe not found. Users must provide it.
)

:: yt-dlp.exe (~18MB - frequently updated, better as sidecar)
if exist "yt-dlp.exe" (
    echo    Copying yt-dlp.exe...
    copy /Y "yt-dlp.exe" "dist\dmx_app\yt-dlp.exe" >nul
) else (
    echo    [WARN] yt-dlp.exe not found. Users must provide it.
)

:: .env.example (already bundled, but also copy to dist root for visibility)
if exist ".env.example" (
    copy /Y ".env.example" "dist\dmx_app\.env.example" >nul
)

:: Profiles directory (also bundled, but mutable copy)
if exist "profiles" (
    xcopy /Y /I /E "profiles" "dist\dmx_app\profiles" >nul 2>&1
)

echo  [OK] Sidecar files copied.
echo.

:: ------------------------------------------------
:: STEP 4: Summary
:: ------------------------------------------------
echo  [4/4] Build complete!
echo.
echo  ========================================================
echo   OUTPUT: dist\dmx_app\
echo  ========================================================
echo.
echo   To run the app:
echo     1. cd dist\dmx_app
echo     2. Copy your .env file (with Azure credentials) here
echo     3. Double-click dmx_app.exe
echo     4. Browser opens to http://localhost:8000
echo.
echo   To distribute:
echo     ZIP the entire dist\dmx_app\ folder.
echo     User needs: Windows 10+, uDMX adapter, internet for YouTube.
echo  ========================================================
echo.
pause
