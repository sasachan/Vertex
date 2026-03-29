# Vertex — Dev launcher
# Starts both the Python engine (FastAPI) and the React UI (Vite dev server).
#
# Usage:
#   .\scripts\dev.ps1                  # default camera
#   .\scripts\dev.ps1 -Input video.mp4 # video file
#   .\scripts\dev.ps1 -Port 8000       # custom port

param(
    [string]$Input = "0",
    [int]$Port = 8000,
    [int]$UIPort = 5173
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Definition)
$engineDir = Join-Path $projectRoot "engine"
$uiDir = Join-Path $projectRoot "ui"

# --- Engine venv ---
$venvActivate = Join-Path $engineDir ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $venvActivate)) {
    Write-Host "Creating engine venv..." -ForegroundColor Yellow
    python -m venv (Join-Path $engineDir ".venv")
    & $venvActivate
    Push-Location $engineDir
    pip install -e ".[server,dev]"
    Pop-Location
} else {
    & $venvActivate
}

# --- UI deps ---
if (-not (Test-Path (Join-Path $uiDir "node_modules"))) {
    Write-Host "Installing UI dependencies..." -ForegroundColor Yellow
    Push-Location $uiDir
    npm install
    Pop-Location
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  VERTEX — Development Mode" -ForegroundColor Cyan
Write-Host "  Engine: http://127.0.0.1:$Port" -ForegroundColor DarkGray
Write-Host "  UI:     http://localhost:$UIPort" -ForegroundColor DarkGray
Write-Host "  Input:  $Input" -ForegroundColor DarkGray
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start engine in background
$engineJob = Start-Job -ScriptBlock {
    param($engineDir, $venvActivate, $input, $port)
    & $venvActivate
    Set-Location $engineDir
    python -m vertex.server --input $input --port $port
} -ArgumentList $engineDir, $venvActivate, $Input, $Port

# Start UI dev server (foreground — Ctrl+C stops both)
try {
    Push-Location $uiDir
    npm run dev -- --port $UIPort
}
finally {
    Write-Host "`nStopping engine..." -ForegroundColor Yellow
    Stop-Job $engineJob -ErrorAction SilentlyContinue
    Remove-Job $engineJob -ErrorAction SilentlyContinue
    Pop-Location
    Write-Host "Done." -ForegroundColor Green
}
