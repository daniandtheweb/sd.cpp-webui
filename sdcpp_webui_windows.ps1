$parsedArgs = @{
    Help = $false
    Server = $false
    Listen = $false
    Autostart = $false
    Darkmode = $false
    Credentials = $false
    AllowInsecureDir = $false
}

foreach ($arg in $args) {
    switch -Regex ($arg) {
        '^(--help|-h)$'         { $parsedArgs.Help = $true }
        '^--server$'            { $parsedArgs.Server = $true}
        '^-Server$'             { $parsedArgs.Server = $true}
        '^--listen$'            { $parsedArgs.Listen = $true }
        '^-Listen$'             { $parsedArgs.Listen = $true }
        '^--autostart$'         { $parsedArgs.Autostart = $true }
        '^-Autostart$'          { $parsedArgs.Autostart = $true }
        '^--darkmode$'          { $parsedArgs.Darkmode = $true }
        '^-Darkmode$'           { $parsedArgs.Darkmode = $true }
        '^--credentials$'       { $parsedArgs.Credentials = $true }
        '^-Credentials$'        { $parsedArgs.Credentials = $true }
        '^--allow-insecure-dir$'{ $parsedArgs.AllowInsecureDir = $true }
        '^-Allow-Insecure-Dir$' { $parsedArgs.AllowInsecureDir = $true }
        default {
            Write-Host "Error: Unknown command parameter: $arg" -ForegroundColor Red
            Show-Help
        }
    }
}

# --- Assign variables for convenience ---
$Help = $parsedArgs.Help
$Server = $parsedArgs.Server
$Listen = $parsedArgs.Listen
$Autostart = $parsedArgs.Autostart
$Darkmode = $parsedArgs.Darkmode
$Credentials = $parsedArgs.Credentials
$AllowInsecureDir = $parsedArgs.AllowInsecureDir

$ErrorActionPreference = 'Stop'

# --- Script Directory ---
$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

# --- Help Function ---
function Show-Help {
    Write-Host @"

Usage: .\sdcpp_webui.ps1 [options]

Options:
    -h, --help                                   Show this help message
    -Server, --server                            Enable stable-diffusion.cpp's server mode
    -Listen, --listen                            Share sd.cpp-webui on your local network
    -Autostart, --autostart                      Open the UI automatically
    -Darkmode, --darkmode                        Forces the UI to launch in dark mode
    -Credentials, --credentials                  Enable password protection using credentials.json.
                                                 Expected format: {"username1": "password1", "username2": "password2"}
    -Allow-Insecure-Dir, --allow-insecure-dir    Allows the usage of external or linked directories based on config.json

"@
    exit 0
}

if ($Help) {
    Show-Help
}

# --- sd-server.exe Check ---
if ($Server) {
    if (-not (Test-Path -Path ".\sd-server.exe") -and -not (Get-Command sd-server -ErrorAction SilentlyContinue)) {
        Write-Host "Error: '--server' flag was used, but the 'sd-server.exe' executable was not found." -ForegroundColor Red
        Write-Host "Please place the 'sd-server.exe' executable in this folder." -ForegroundColor Red
        exit 1
    }
}

# --- sd.exe Check ---
if (-not (Test-Path -Path ".\sd-cli.exe") -and -not (Get-Command sd-cli -ErrorAction SilentlyContinue) -and `
    -not (Test-Path -Path ".\sd.exe") -and -not (Get-Command sd -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Neither 'sd-cli.exe' nor 'sd.exe' executables were found." -ForegroundColor Red
    Write-Host "Please place the 'sd-cli.exe' or 'sd.exe' executables in this folder." -ForegroundColor Red
    exit 1
}

# --- Python Virtual Environment ---
if (-not (Test-Path -Path ".\venv" -PathType Container)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
}

Write-Host "Activating virtual environment..."
. ".\venv\Scripts\Activate.ps1"

# --- Requirements Check ---
$requiredPackages = Get-Content -Path ".\requirements.txt"
if (pip freeze | Select-String -Pattern $requiredPackages -SimpleMatch -Quiet) {
    Write-Host "Requirements are satisfied."
}
else {
    Write-Host "Installing requirements..."
    python -m pip install -r .\requirements.txt --upgrade pip
    Write-Host "Requirements installed."
}

# --- Build Python Argument List ---
$pythonArgs = @()
if ($Server) { $pythonArgs += "--server" }
if ($Listen) { $pythonArgs += "--listen" }
if ($Autostart) { $pythonArgs += "--autostart" }
if ($Darkmode) { $pythonArgs += "--darkmode" }
if ($Credentials) { $pythonArgs += "--credentials" }
if ($AllowInsecureDir) { $pythonArgs += "--allow-insecure-dir" }

if ($pythonArgs.Count -eq 0) {
    Write-Host "Starting the WebUI...`n"
}
else {
    Write-Host "Starting the WebUI with arguments: $($pythonArgs -join ' ')`n"
}
python .\sdcpp_webui.py $pythonArgs
