param([string]$BootstrapPython = "python", [switch]$SkipDownloads)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$runtime = Join-Path $repo "work\human-runtime"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $runtime "python"
$env:UV_CACHE_DIR = Join-Path $runtime "uv-cache"
$env:UV_LINK_MODE = "copy"
$uv = Join-Path $repo ".venv\Scripts\uv.exe"
if (-not (Test-Path -LiteralPath $uv)) {
    $bootstrap = Join-Path $runtime "bootstrap"
    & $BootstrapPython -m venv $bootstrap
    if ($LASTEXITCODE -ne 0) { throw "Could not create the isolated bootstrap environment" }
    & (Join-Path $bootstrap "Scripts\python.exe") -m pip install "uv==0.12.21"
    if ($LASTEXITCODE -ne 0) { throw "uv installation failed" }
    $uv = Join-Path $bootstrap "Scripts\uv.exe"
}
& $uv python install 3.11.16
if ($LASTEXITCODE -ne 0) { throw "Python installation failed" }
$python = Join-Path $runtime "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    & $uv venv (Join-Path $runtime "venv") --python 3.11.16
    if ($LASTEXITCODE -ne 0) { throw "Human environment creation failed" }
}
& $uv pip install --python $python -r tools/human_runtime/requirements-windows.lock.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw "Human dependencies failed to install" }
& $uv pip check --python $python
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed" }
& $python tools/human_runtime/prepare_sources.py
if ($LASTEXITCODE -ne 0) { throw "Source preparation failed" }
if ($SkipDownloads) {
    & $python tools/human_runtime/prepare_assets.py
} else {
    & $python tools/human_runtime/prepare_assets.py --download
}
if ($LASTEXITCODE -ne 0) { throw "Asset preparation failed" }
& $python tools/human_runtime/prepare_converter.py
if ($LASTEXITCODE -ne 0) { throw "Track 1 converter preparation failed" }
& $uv pip freeze --python $python | Set-Content -LiteralPath (Join-Path $runtime "installed-packages.txt") -Encoding utf8
& $python -c "import torch,cv2,numpy; print('torch',torch.__version__,'CUDA',torch.version.cuda,'available',torch.cuda.is_available()); print('numpy',numpy.__version__,'opencv',cv2.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Core runtime import failed" }
Write-Host "Human environment prepared. Model inference is a separate step."
