# Vertex - TheSighter: Quick Launch
#
# MODES
#   (default)   -- run live biomechanics analysis
#   -Extract    -- batch-extract best frames from a folder of videos (developer/validation use)
#
# USAGE
#   .\run.ps1                                    -- live camera
#   .\run.ps1 video.mp4                          -- analyse a single video file
#   .\run.ps1 photo.jpg                          -- analyse a single image
#   .\run.ps1 video.mp4 --NoDisplay              -- headless batch analysis
#
#   .\run.ps1 -Extract                           -- extract frames from data/video/ (new files only)
#   .\run.ps1 -Extract -Flush                    -- reset registry and reprocess all videos
#   .\run.ps1 -Extract -Profile image            -- extract from data/image/ instead
#   .\run.ps1 -Extract -Profile all              -- scan entire data/ tree
#   .\run.ps1 -Extract -Max 12 -SampleEvery 1   -- override frame count or sampling density

param(
    # --- Analysis mode (default) ---
    [string]$Input      = "",
    [switch]$NoDisplay,

    # --- Extract mode ---
    [switch]$Extract,

    # Extract options (ignored unless -Extract is set)
    # Profile wires Input/Output/Max/SampleEvery automatically
    [ValidateSet("video", "image", "all", "")]
    [string]$Profile    = "video",
    [string]$ExtractInput  = "",
    [string]$ExtractOutput = "",
    [int]$Max           = 0,
    [float]$SampleEvery = 0,
    [switch]$Flush
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvActivate = Join-Path $projectRoot "src\.venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Host "Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv (Join-Path $projectRoot "src\.venv")
    & $venvActivate
    Write-Host "Installing Vertex..." -ForegroundColor Yellow
    Push-Location $projectRoot
    pip install -e ".[dev]"
    Pop-Location
} else {
    & $venvActivate
}

Push-Location $projectRoot

# -------------------------------------------------------------------------
# EXTRACT MODE
# -------------------------------------------------------------------------
if ($Extract) {
    $DATA_ROOT = "D:\Code\Vertex-AI\data"
    $OUT_ROOT  = "D:\Code\Vertex-AI\data\extracted"

    $profileInput       = $DATA_ROOT
    $profileOutput      = $OUT_ROOT
    $profileMax         = 8
    $profileSampleEvery = 2.0

    switch ($Profile) {
        "video" {
            $profileInput       = "$DATA_ROOT\video"
            $profileOutput      = $OUT_ROOT
            $profileMax         = 10
            $profileSampleEvery = 2.0
        }
        "image" {
            $profileInput       = "$DATA_ROOT\image"
            $profileOutput      = $OUT_ROOT
            $profileMax         = 5
            $profileSampleEvery = 0
        }
        "all" {
            $profileInput       = $DATA_ROOT
            $profileOutput      = $OUT_ROOT
            $profileMax         = 8
            $profileSampleEvery = 2.0
        }
    }

    $resolvedInput       = if ($ExtractInput)      { $ExtractInput }  else { $profileInput }
    $resolvedOutput      = if ($ExtractOutput)     { $ExtractOutput } else { $profileOutput }
    $resolvedMax         = if ($Max -gt 0)         { $Max }           else { $profileMax }
    $resolvedSampleEvery = if ($SampleEvery -gt 0) { $SampleEvery }   else { $profileSampleEvery }

    $pyArgs = @(
        "$projectRoot\extract_frames.py",
        $resolvedInput,
        "--output", $resolvedOutput,
        "--max",    $resolvedMax
    )
    if ($resolvedSampleEvery -gt 0) {
        $pyArgs += @("--sample-every", $resolvedSampleEvery)
    }
    if ($Flush) {
        $pyArgs += "--flush"
    }

    Write-Host ""
    Write-Host "Vertex - Frame Extractor" -ForegroundColor Cyan
    Write-Host "  Profile:   $Profile" -ForegroundColor DarkGray
    Write-Host "  Input:     $resolvedInput" -ForegroundColor DarkGray
    Write-Host "  Output:    $resolvedOutput" -ForegroundColor DarkGray
    Write-Host "  Max/video: $resolvedMax    Sample every: ${resolvedSampleEvery}s" -ForegroundColor DarkGray
    if ($Flush) { Write-Host "  Mode:      FLUSH + full reprocess" -ForegroundColor Yellow }
    Write-Host ""

    python @pyArgs
    Pop-Location
    exit
}

# -------------------------------------------------------------------------
# ANALYSIS MODE (default)
# -------------------------------------------------------------------------
if ($Input -ne "") {
    Write-Host "Starting Vertex - TheSighter: $Input" -ForegroundColor Cyan
} else {
    Write-Host "Starting Vertex - TheSighter (default camera)" -ForegroundColor Cyan
}
Write-Host "Controls: ENTER=consent  Q=Quit  R=Reset  D=Debug  C=Coach  S=Snapshot  SPACE=pause" -ForegroundColor DarkGray

$args_list = @()
if ($Input -ne "") { $args_list += $Input }
if ($NoDisplay) { $args_list += "--no-display" }

if ($args_list.Count -gt 0) {
    python -m vertex @args_list
} else {
    python -m vertex
}

Pop-Location