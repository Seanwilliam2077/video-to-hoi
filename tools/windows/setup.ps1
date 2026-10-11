param([switch]$Online)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$env:UV_PROJECT_ENVIRONMENT = Join-Path $ProjectRoot '.venv'
$env:UV_LINK_MODE = 'copy'
# The pinned local SAM2 wheel includes its original CUDA hole-filling extension.
# Refuse a missing wheel rather than silently installing a degraded mask path.
if (-not (Test-Path -LiteralPath "$ProjectRoot/wheelhouse/windows/sam_2-1.0-cp312-cp312-win_amd64.whl")) {
    throw 'The preserved SAM2 Windows CUDA wheel is required.'
}
[string[]]$NetworkFlags = @()
if (-not $Online) { $NetworkFlags = @('--offline') }
Push-Location $ProjectRoot
try {
    if (-not (Test-Path '.venv/Scripts/python.exe')) {
        & uv venv @NetworkFlags --python 3.12 .venv
        if ($LASTEXITCODE) { throw 'Virtual environment creation failed.' }
    }
    # Reuse the existing CUDA wheel cache even when the lock URL is a CDN alias.
    & uv pip install @NetworkFlags --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu130 'torch==2.14.1+cu130' 'torchvision==0.29.1+cu130'
    if ($LASTEXITCODE) { throw 'CUDA package installation failed.' }
    # Resolve installed versions from local package cache records first. Some
    # cached wheel origins differ from the public wheel URLs in uv.lock.
    & uv pip install @NetworkFlags --no-deps --python .venv/Scripts/python.exe -r deployment/windows/requirements.cached.txt
    if ($LASTEXITCODE) { throw 'Cached runtime package installation failed.' }
    & uv sync @NetworkFlags --locked --project deployment/windows --group test
    if ($LASTEXITCODE) { throw 'Windows runtime synchronization failed.' }
} finally {
    Pop-Location
}
