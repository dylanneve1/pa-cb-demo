# PA CB demo runner (Windows box).
# Extracts the BuildOV-Win artifact, sets up a venv, runs the demo and the
# parity check, and tees every run into logs\.
#
# Usage:
#   .\run_demo.ps1 -Artifact C:\path\to\openvino-genai-win-...-pa-cb-demo-pr1.zip -ModelDir C:\models\Qwen3-0.6B_int4_sym_group-1_dyn_stateful

param(
    [Parameter(Mandatory = $true)][string]$Artifact,
    [Parameter(Mandatory = $true)][string]$ModelDir,
    [string]$WorkDir = "$PSScriptRoot\work"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $Artifact)) {
    Write-Host "Artifact not found: $($Artifact)"
    exit 1
}
if (-not (Test-Path $ModelDir)) {
    Write-Host "Model dir not found: $($ModelDir)"
    exit 1
}

$logs = "$PSScriptRoot\logs"
New-Item -ItemType Directory -Force -Path $WorkDir, $logs | Out-Null

Write-Host "Extracting $($Artifact) ..."
tar -xf $Artifact -C $WorkDir

$wheelDir = Get-ChildItem -Recurse -Directory $WorkDir | Where-Object { $_.Name -eq 'wheels' } | Select-Object -First 1
if (-not $wheelDir) {
    Write-Host "No wheels directory inside the artifact."
    exit 1
}

Write-Host "Creating venv ..."
python -m venv "$WorkDir\venv"
& "$WorkDir\venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip | Out-Null

# Requirements first, build wheels second: the build's openvino / genai
# wheels must override whatever PyPI openvino the requirements dragged in.
python -m pip install -r "$PSScriptRoot\requirements.txt"
Get-ChildItem "$($wheelDir.FullName)\*.whl" | ForEach-Object { python -m pip install --force-reinstall --no-deps $_.FullName }

$env:MODEL_DIR = $ModelDir
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'

Write-Host "=== demo (quiet) ==="
python "$PSScriptRoot\pa_demo.py" 2>&1 | Tee-Object "$logs\$($stamp)_01_demo.log"
if ($LASTEXITCODE -ne 0) { Write-Host "demo failed"; exit 1 }

Write-Host "=== demo (NPUW log INFO, shows the PA front-end engaging) ==="
$env:OPENVINO_NPUW_LOG_LEVEL = 'INFO'
python "$PSScriptRoot\pa_demo.py" 2>&1 | Tee-Object "$logs\$($stamp)_02_demo_npuw_info.log"
$env:OPENVINO_NPUW_LOG_LEVEL = ''
if ($LASTEXITCODE -ne 0) { Write-Host "demo (INFO) failed"; exit 1 }

Write-Host "=== parity: plain CPU vs NPU + NPUW_PA ==="
python "$PSScriptRoot\pa_parity.py" 2>&1 | Tee-Object "$logs\$($stamp)_03_parity.log"
if ($LASTEXITCODE -ne 0) { Write-Host "PARITY FAILED"; exit 1 }

Write-Host "All runs green. Logs in $($logs)"
