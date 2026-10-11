# Dot-source to use the project-owned Windows runtime.
$ErrorActionPreference = 'Stop'
$CariRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$CariCuda = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
if (-not (Test-Path -LiteralPath "$CariCuda\bin\nvcc.exe")) {
    throw "CUDA Toolkit 13.0 is required at $CariCuda"
}
$CariVsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$CariVsPath = & $CariVsWhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $CariVsPath) { throw 'Visual Studio 2022 C++ build tools are required.' }
& "$CariVsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
$env:CUDA_HOME = $CariCuda
$env:CUDA_PATH = $CariCuda
$env:CUB_HOME = "$CariCuda\include\cccl"
$env:PATH = "$CariRoot\.venv\Scripts;$CariCuda\bin;$env:PATH"
if (Test-Path -LiteralPath 'C:\Tools\ffmpeg\bin\ffprobe.exe') {
    $env:PATH = "C:\Tools\ffmpeg\bin;$env:PATH"
    $env:IMAGEIO_FFMPEG_EXE = 'C:\Tools\ffmpeg\bin\ffmpeg.exe'
}
# Standard HTTPS avoids Xet transfer stalls behind the local proxy.
$env:HF_HUB_DISABLE_XET = '1'
$env:VIRTUAL_ENV = "$CariRoot\.venv"
$env:UV_PROJECT_ENVIRONMENT = "$CariRoot\.venv"
$env:DISTUTILS_USE_SDK = '1'
$env:TORCH_CUDA_ARCH_LIST = '12.0'
$env:MAX_JOBS = '4'
$env:CL = '/Zc:preprocessor'
$env:NVCC_PREPEND_FLAGS = '-Xcompiler=/Zc:preprocessor'
$env:OPENCV_IO_ENABLE_OPENEXR = '1'
$env:PYTHONUTF8 = '1'
Set-Location -LiteralPath $CariRoot
