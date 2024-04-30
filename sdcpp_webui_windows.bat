@echo off
setlocal

:: Get the directory of the batch script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist "venv" (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
)

echo Activating virtual environment...
if exist "venv\Scripts\activate" (
    call venv\Scripts\activate
) else (
    echo Error: Virtual environment activation script not found.
    exit /b 1
)
echo Virtual environment activated successfully.

pip freeze | findstr /i /x /g:requirements.txt >nul
if %errorlevel% equ 0 (
    echo Requirements are satisfied.
) else (
    echo Installing requirements...
    pip install -r requirements.txt
    echo Requirements installed.
)

echo Starting the WebUI...

python sdcpp_webui.py %*

endlocal
