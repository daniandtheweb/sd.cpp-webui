$parsedArgs = @{
    Help = $false
    Listen = $false
    Autostart = $false
    Darkmode = $false
    AllowInsecureDir = $false
}

foreach ($arg in $args) {
    switch -Regex ($arg) {
        '^(--help|-h)$'        { $parsedArgs.Help = $true }
        '^--listen$'            { $parsedArgs.Listen = $true }
        '^-Listen$'             { $parsedArgs.Listen = $true }
        '^--autostart$'         { $parsedArgs.Autostart = $true }
        '^-Autostart$'          { $parsedArgs.Autostart = $true }
        '^--darkmode$'          { $parsedArgs.Darkmode = $true }
        '^-Darkmode$'           { $parsedArgs.Darkmode = $true }
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
$Listen = $parsedArgs.Listen
$Autostart = $parsedArgs.Autostart
$Darkmode = $parsedArgs.Darkmode
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
    -Listen, --listen                            Share sd.cpp-webui on your local network
    -Autostart, --autostart                      Open the UI automatically
    -Darkmode, --darkmode                        Forces the UI to launch in dark mode
    -Allow-Insecure-Dir, --allow-insecure-dir    Allows the usage of external or linked directories based on config.json

"@
    exit 0
}

if ($Help) {
    Show-Help
}

# --- sd.exe Check ---
if (-not (Test-Path -Path ".\sd-cli.exe") -and -not (Get-Command sd-cli -ErrorAction SilentlyContinue) -and `
    -not (Test-Path -Path ".\sd.exe") -and -not (Get-Command sd -ErrorAction SilentlyContinue)) {
    Write-Warning "Neither 'sd-cli.exe' nor 'sd.exe' executables were found in this directory or your PATH."
    Write-Warning "Please place the stable-diffusion.cpp executable (renamed to 'sd-cli.exe' or 'sd.exe') in this folder."
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
if ($Listen) { $pythonArgs += "--listen" }
if ($Autostart) { $pythonArgs += "--autostart" }
if ($Darkmode) { $pythonArgs += "--darkmode" }
if ($AllowInsecureDir) { $pythonArgs += "--allow-insecure-dir" }

if ($pythonArgs.Count -eq 0) {
    Write-Host "Starting the WebUI...`n"
}
else {
    Write-Host "Starting the WebUI with arguments: $pythonArgs`n"
}
python .\sdcpp_webui.py $pythonArgs
