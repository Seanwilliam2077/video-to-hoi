# Reviewed manual cleanup. The agent's deletion tools were blocked by policy.
# Run without switches to preview; run with -Execute to remove only these targets.
param([switch]$Execute)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$CacheRoot = 'C:\Users\X\AppData\Local\uv\cache'
$MhrEnvironment = 'C:\Users\X\.cache\cari4d-mhr\venv'
$OldLink = 'D:\Projects\CARI4D_MHR_Win\.venv'
$CacheIds = @('zL0NVrxZLPVM5tx8', '0R0QVxZ5OxdWYj2x', '4WzKbTXKradDud5F',
    'TKiutcTF3G5lGSP9', 'PAlUKRY36SvfeD2O', 'mw_9cPbW5O6xEPHY', 'H-ty4itBA-mZTa4L')
$Targets = @($MhrEnvironment) + @($CacheIds | ForEach-Object { Join-Path $CacheRoot "archive-v0\$_" })
$Runtime = Get-Content -LiteralPath "$ProjectRoot\artifacts\windows-migration\runtime-check.json" -Raw | ConvertFrom-Json
$Transfer = Get-Content -LiteralPath "$ProjectRoot\artifacts\windows-migration\copy-verification.json" -Raw | ConvertFrom-Json
if ($Runtime.environment -ne "$ProjectRoot\.venv" -or $Runtime.checks.Count -lt 4 -or $Transfer.failures.Count -ne 0) {
    throw 'Successful migration and GPU verification are required before cleanup.'
}
$NewEnvironment = Get-Item -LiteralPath "$ProjectRoot\.venv" -Force
if ($NewEnvironment.LinkType) { throw 'The new project environment must be a real directory.' }
$Processes = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python(w)?|uv|uvx)\.exe$' })
$ExistingTargets = @()
foreach ($Target in $Targets) {
    if (-not (Test-Path -LiteralPath $Target)) { continue }
    $Item = Get-Item -LiteralPath $Target -Force
    $Resolved = (Resolve-Path -LiteralPath $Target).Path
    if ($Item.LinkType -or $Resolved -ne $Target -or -not (Test-Path -LiteralPath "$Target\pyvenv.cfg")) {
        throw "Unexpected environment target: $Target"
    }
    if ($Target -ne $MhrEnvironment -and -not $Resolved.StartsWith("$CacheRoot\archive-v0\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escaped the allowed cache directory: $Target"
    }
    foreach ($Process in $Processes) {
        if (($Process.ExecutablePath -and $Process.ExecutablePath.StartsWith("$Target\", [StringComparison]::OrdinalIgnoreCase)) -or
            ($Process.CommandLine -and $Process.CommandLine.Contains($Target))) {
            throw "Environment is in use by PID $($Process.ProcessId): $Target"
        }
    }
    $ExistingTargets += $Resolved
}
Write-Output 'Exact environment directories selected for removal:'
$ExistingTargets | Write-Output
if (-not $Execute) { Write-Output 'Preview only. Use -Execute to perform cleanup.'; return }
$Records = [System.Collections.Generic.List[object]]::new()
$ReportPath = "$ProjectRoot\artifacts\windows-migration\cleanup.json"
function Save-CleanupRecord([string]$Path, [string]$Kind) {
    $Records.Add([pscustomobject]@{path=$Path; kind=$Kind; removed=(-not (Test-Path -LiteralPath $Path))})
    [IO.File]::WriteAllText($ReportPath, ($Records | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))
}
if (Test-Path -LiteralPath $OldLink) {
    $LinkItem = Get-Item -LiteralPath $OldLink -Force
    if ($LinkItem.LinkType -ne 'Junction' -or $LinkItem.Target -ne $MhrEnvironment) { throw 'Unexpected old environment link.' }
    Remove-Item -LiteralPath $OldLink -Force
    Save-CleanupRecord $OldLink 'old environment junction'
}
foreach ($Target in $ExistingTargets) {
    Remove-Item -LiteralPath $Target -Recurse -Force
    Save-CleanupRecord $Target 'environment'
}
$EnvironmentIndex = Join-Path $CacheRoot 'environments-v2'
if (Test-Path -LiteralPath $EnvironmentIndex) {
    foreach ($IndexFile in Get-ChildItem -LiteralPath $EnvironmentIndex -Recurse -File) {
        $Value = [IO.File]::ReadAllText($IndexFile.FullName).Trim()
        if ($Value -in @($CacheIds | ForEach-Object { "archive-v0/$_" })) {
            Remove-Item -LiteralPath $IndexFile.FullName -Force
            Save-CleanupRecord $IndexFile.FullName 'environment cache index'
        }
    }
}
Write-Output "Cleanup records: $ReportPath"
