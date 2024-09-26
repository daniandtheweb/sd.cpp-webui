@echo off
setlocal

set help=false
set valid=false

:: Get the directory of the batch script
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

for %%A in (%*) do (
    if "%%A"=="--help" (
        set help=true
        set valid=true
    ) else if "%%A"=="-h" (
        set help=true
        set valid=true
    ) else if "%%A"=="--listen" (
        set valid=true
    ) else if "%%A"=="--autostart" (
        set valid=true
    )
)

if "%valid%"=="false" (
    echo Unknown command parameter.
    exit /b
)

if "%help%"=="true" (
    echo.
    echo.
    echo Usage: sdcpp_webui_windows.bat [options]
    echo.
    echo Options:
    echo     -h or --help:            Show this help
    echo     --listen:                Share sd.cpp-webui on your local network
    echo     --autostart:             Open the UI automatically
    echo.
    echo.
    exit /b
)

if not exist "sd.exe" (
    echo.
    echo.
    echo Warning: stable-diffusion.cpp executable not found.
    echo For the command to work place the stable-diffusion.cpp executable in the main sd.cpp-webui folder.
    echo The executable must be called 'sd.exe'.
    echo.
    echo.
)

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
