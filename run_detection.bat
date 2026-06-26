@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt >nul

if "%~1"=="" (
    echo Usage: run_detection.bat image ^| video [additional args]
    echo Example: run_detection.bat image --nogui
    echo Example: run_detection.bat video --input data/input/traffic_signs.mp4 --max-frames 10
    exit /b 1
)

if /I "%~1"=="image" (
    shift
    python process_image.py %*
    exit /b %errorlevel%
)

if /I "%~1"=="video" (
    shift
    python process_video.py %*
    exit /b %errorlevel%
)

echo Unknown mode: %~1
exit /b 1
endlocal
