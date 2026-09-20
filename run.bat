@echo off
setlocal
title EcoVision AI Web Server

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Khong tim thay Python trong .venv.
    echo Hay tao/cai dat moi truong .venv truoc.
    pause
    exit /b 1
)

:: Giai phong port 8000 neu server cu van dang chay.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do taskkill /PID %%P /F >nul 2>&1

echo EcoVision AI dang khoi dong...
echo Mo trinh duyet tai: http://localhost:8000

:: Mo mot cua so rieng de server tiep tuc chay sau khi file bat ket thuc.
start "EcoVision AI Web Server" /D "%ROOT%" "%PYTHON%" -m uvicorn server:app --host 0.0.0.0 --port 8000
exit /b 0
