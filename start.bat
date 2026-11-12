@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>nul
if errorlevel 1 (
    echo Node.js 20 or newer is required. Install it and run start.bat again.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo npm was not found. Reinstall Node.js with npm and run start.bat again.
    pause
    exit /b 1
)

if not exist "frontend\node_modules\next\package.json" (
    echo Installing frontend dependencies...
    pushd frontend
    call npm ci
    if errorlevel 1 (
        popd
        goto failed
    )
    popd
)

set "PYTHON=backend\.venv313\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=backend\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python 3.11, 3.12, or 3.13 is required. Install Python and run start.bat again.
        pause
        exit /b 1
    )
    echo Creating Python environment...
    py -3.13 -m venv backend\.venv
    if errorlevel 1 py -3.12 -m venv backend\.venv
    if errorlevel 1 py -3.11 -m venv backend\.venv
    if errorlevel 1 goto python_missing
    set "PYTHON=backend\.venv\Scripts\python.exe"
)

"%PYTHON%" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul
if errorlevel 1 (
    echo The backend environment must use Python 3.11, 3.12, or 3.13.
    pause
    exit /b 1
)

"%PYTHON%" -c "import fastapi, uvicorn, multipart, pydantic_settings, numpy, pandas, scipy, cv2, ultralytics, lap, fast_alpr, supervision, reportlab, httpx" >nul 2>nul
if errorlevel 1 (
    echo Installing backend dependencies...
    "%PYTHON%" -m pip install -r backend\requirements.txt
    if errorlevel 1 goto failed
)

if not exist "backend\.env" copy "backend\.env.example" "backend\.env" >nul

echo Starting TrafficAI. Keep both terminal windows open while using the app.
start "TrafficAI API" /D "%~dp0backend" cmd /k ""%~dp0%PYTHON%" -m uvicorn main:app --reload --port 8000"
start "TrafficAI Frontend" /D "%~dp0frontend" cmd /k "set NEXT_PUBLIC_API_URL=http://localhost:8000 && set NEXT_PUBLIC_DEMO_MODE=false && npm run dev"
set "SERVICES_READY="
for /L %%I in (1,1,30) do (
    curl.exe --fail --silent --output nul http://localhost:3000 >nul 2>nul
    if not errorlevel 1 (
        curl.exe --fail --silent --output nul http://localhost:8000/api/health >nul 2>nul
        if not errorlevel 1 set "SERVICES_READY=1"
    )
    if defined SERVICES_READY goto services_ready
    timeout /t 2 /nobreak >nul
)
echo Services are still starting. The dashboard will be at http://localhost:3000.
:services_ready
start "" http://localhost:3000
exit /b 0

:python_missing
echo Could not create a virtual environment. Install Python 3.11, 3.12, or 3.13.
pause
exit /b 1

:failed
echo Setup failed. Check the error above, then run start.bat again.
pause
exit /b 1
