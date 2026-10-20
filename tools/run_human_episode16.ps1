param(
    [ValidateSet("Smoke", "Infer", "Fit", "ReviewRaw", "Review", "Check")][string]$Stage = "Check",
    [ValidateSet("cpu", "cuda")][string]$Device = "cpu",
    [int]$SmokeFrames = 3,
    [switch]$Smoke,
    [ValidateSet("fit", "median")][string]$FitMethod = "fit"
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$python = Join-Path $repo "work\human-runtime\venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Run tools\setup_human_windows.ps1 first" }
$video = "data/v2d/track_1/videos/chunk-000/observation.images.exo_camera/episode_000016.mp4"
$url = "https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/5f68335f3acc802033d1e80728c1633197521de8/track_1/videos/chunk-000/observation.images.exo_camera/episode_000016.mp4"
$isSmoke = $Smoke -or ($Stage -eq "Smoke")
$runName = if ($isSmoke) { "human-ep16-$Device-smoke" } else { "human-ep16-$Device-full" }
$run = Join-Path "runs" $runName
$fitted = Join-Path $run "fixed-$FitMethod"
switch ($Stage) {
    "Check" {
        & $python tools/human_runtime/prepare_assets.py
        if ($LASTEXITCODE -ne 0) { throw "Asset integrity check failed" }
        & $python -c "import torch,cv2; print('torch',torch.__version__,'CUDA available',torch.cuda.is_available()); print('Device count',torch.cuda.device_count())"
    }
    { $_ -in "Smoke", "Infer" } {
        $arguments = @("tools/run_sam3d_body.py", "--video", $video, "--source-url", $url, "--episode", "16",
            "--output", $run, "--checkpoint", "weights/human/sam-3d-body-dinov3/model.ckpt",
            "--mhr-path", "weights/human/mhr_model.pt", "--device", $Device,
            "--bbox", "440", "360", "850", "1000", "--hfov", "60", "--threads", "8", "--resume")
        if ($isSmoke) { $arguments += @("--limit", "$SmokeFrames") }
        & $python @arguments
    }
    "Fit" {
        $arguments = @("tools/finalize_human.py", "--run", $run, "--output", $fitted,
            "--device", $Device, "--method", $FitMethod, "--model-batch", "1")
        if ($isSmoke) { $arguments += "--allow-partial" }
        & $python @arguments
    }
    { $_ -in "Review", "ReviewRaw" } {
        $predictions = if ($Stage -eq "ReviewRaw") { Join-Path $run "raw" } else { Join-Path $fitted "review_arrays" }
        $review = if ($Stage -eq "ReviewRaw") { Join-Path $run "review-raw" } else { Join-Path $fitted "review" }
        & $python tools/review_human.py --video $video --predictions $predictions --episode 16 --output $review
        if ($LASTEXITCODE -eq 2 -and $isSmoke) {
            Write-Warning "Partial smoke review written; incomplete video coverage is expected and remains flagged."
            exit 0
        }
    }
}
if ($LASTEXITCODE -ne 0) { throw "$Stage failed with exit code $LASTEXITCODE; see output and the run's state.json" }
