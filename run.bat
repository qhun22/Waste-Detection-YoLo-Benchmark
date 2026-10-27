@echo off
setlocal
title EcoVision AI

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Khong tim thay Python trong .venv
    echo Hay tao moi truong: python -m venv .venv ^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

:: Giai phong port 8000 neu dang chay
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING"') do taskkill /PID %%P /F >nul 2>&1

echo EcoVision AI dang khoi dong...
echo Mo trinh duyet tai: http://localhost:8000

start "EcoVision AI" /D "%ROOT%" "%PYTHON%" -m uvicorn app.server:app --host 0.0.0.0 --port 8000
exit /b 0
