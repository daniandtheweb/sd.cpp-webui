# --- Help Function ---
function Show-Help {
    param([int]$ExitCode = 0)
    Write-Host @"

Usage: .\sdcpp_webui.ps1 [options]

Options:
    -h, --help                                   Show this help message
    -Server, --server                            Enable stable-diffusion.cpp's server mode
    -Listen, --listen                            Share sd.cpp-webui on your local network
    -Autostart, --autostart                      Open the UI automatically
    -Credentials, --credentials                  Enable password protection using credentials.json.
                                                 Expected format: {"username1": "password1", "username2": "password2"}
    -Allow-Insecure-Dir, --allow-insecure-dir    Allows the usage of external or linked directories based on config.json

"@
    exit $ExitCode
}

$parsedArgs = @{
    Help = $false
    Server = $false
    Listen = $false
    Autostart = $false
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
        '^--credentials$'       { $parsedArgs.Credentials = $true }
        '^-Credentials$'        { $parsedArgs.Credentials = $true }
        '^--allow-insecure-dir$'{ $parsedArgs.AllowInsecureDir = $true }
        '^-Allow-Insecure-Dir$' { $parsedArgs.AllowInsecureDir = $true }
        default {
            Write-Host "Error: Unknown command parameter: $arg" -ForegroundColor Red
            Show-Help 1
        }
    }
}

# --- Assign variables for convenience ---
$Help = $parsedArgs.Help
$Server = $parsedArgs.Server
$Listen = $parsedArgs.Listen
$Autostart = $parsedArgs.Autostart
$Credentials = $parsedArgs.Credentials
$AllowInsecureDir = $parsedArgs.AllowInsecureDir

$ErrorActionPreference = 'Stop'

# --- Script Directory ---
$ScriptDir = $PSScriptRoot
Set-Location $ScriptDir

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
if (-not $Server) {
    if (-not (Test-Path -Path ".\sd-cli.exe") -and -not (Get-Command sd-cli -ErrorAction SilentlyContinue) -and `
        -not (Test-Path -Path ".\sd.exe") -and -not (Get-Command sd -ErrorAction SilentlyContinue)) {
        Write-Host "Error: Neither 'sd-cli.exe' nor 'sd.exe' executables were found." -ForegroundColor Red
        Write-Host "Please place the 'sd-cli.exe' or 'sd.exe' executables in this folder." -ForegroundColor Red
        exit 1
    }
}

# --- Python Virtual Environment ---
if (-not (Test-Path -Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: failed to create the virtual environment." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Activating virtual environment..."
. ".\venv\Scripts\Activate.ps1"

# --- Requirements Check ---
python -m pip install --quiet --disable-pip-version-check -r .\requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: failed to install the requirements." -ForegroundColor Red
    exit 1
}

# --- Build Python Argument List ---
$pythonArgs = @()
if ($Server) { $pythonArgs += "--server" }
if ($Listen) { $pythonArgs += "--listen" }
if ($Autostart) { $pythonArgs += "--autostart" }
if ($Credentials) { $pythonArgs += "--credentials" }
if ($AllowInsecureDir) { $pythonArgs += "--allow-insecure-dir" }

if ($pythonArgs.Count -eq 0) {
    Write-Host "Starting the WebUI...`n"
}
else {
    Write-Host "Starting the WebUI with arguments: $($pythonArgs -join ' ')`n"
}
python .\sdcpp_webui.py $pythonArgs
