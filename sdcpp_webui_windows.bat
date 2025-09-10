@echo off
setlocal

set "all_args_valid=true"

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: --- Argument Parsing Loop ---
:: This loop validates EVERY argument. If one is invalid, it will fail.
for %%A in (%*) do (
    if /i "%%A"=="--help" ( goto :print_help )
    if /i "%%A"=="-h" ( goto :print_help )
    if /i "%%A"=="--listen" ( goto :next_arg )
    if /i "%%A"=="--autostart" ( goto :next_arg )
    if /i "%%A"=="--darkmode" ( goto :next_arg )

    echo Error: Unknown command parameter: %%A
    set "all_args_valid=false"
    
    :next_arg
)

if "%all_args_valid%"=="false" ( exit /b 1 )

goto :main_logic

:: --- Help Text Section ---
:print_help
echo.
echo Usage: sdcpp_webui_windows.bat [options]
echo.
echo Options:
echo     -h or --help:         Show this help
echo     --listen:             Share sd.cpp-webui on your local network
echo     --autostart:          Open the UI automatically
echo     --darkmode:           Forces the UI to launch in dark mode
echo.
exit /b 0


:: --- Main Script Logic ---
:main_logic

if not exist "sd.exe" (
    echo.
    echo Warning: 'sd.exe' not found.
    echo Please place the stable-diffusion.cpp executable (renamed to 'sd.exe') in this folder.
    echo.
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate

pip freeze | findstr /i /x /g:requirements.txt >nul
if %errorlevel% equ 0 (
    echo Requirements are satisfied.
) else (
    echo Installing requirements...
    pip install -r requirements.txt --upgrade pip
)

echo Starting the WebUI...
python sdcpp_webui.py %*

endlocal
