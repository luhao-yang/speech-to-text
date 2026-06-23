@echo off
REM Launch the native push-to-talk dictation client (Windows side).
REM Double-click this, or run it from a terminal. Make sure the server
REM (python app.py) is already running in WSL.
setlocal
cd /d "%~dp0"

REM Prefer the Windows "py" launcher; fall back to python on PATH.
where py >nul 2>nul
if %errorlevel%==0 (
    py dictate_client.py %*
) else (
    python dictate_client.py %*
)

if %errorlevel% neq 0 (
    echo.
    echo Client exited with an error. If modules are missing, run:
    echo     pip install -r requirements-client.txt
    pause
)
endlocal
