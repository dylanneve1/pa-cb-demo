# PA CB demo runner (Windows).
#
# Two modes:
#   - Active environment (default): run from a python env that already has the
#     build's openvino + openvino_genai wheels. No artifact needed.
#       .\run_demo.ps1
#   - From an artifact: extract the build archive, create a venv, install
#     requirements, then the build's wheels on top.
#       .\run_demo.ps1 -Artifact C:\path\to\<build-archive>.zip
#
# ModelDir defaults to the NPU machines' standard layout; override for a
# different model or location. Every run is teed into logs\.

param(
    [string]$Artifact = '',
    [string]$ModelDir = 'C:\npuw\models\current\LLM\Qwen3-0.6B_int4_sym_group-1_dyn_stateful',
    [string]$WorkDir = "$PSScriptRoot\work"
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path $ModelDir)) {
    Write-Host "Model dir not found: $($ModelDir)"
    exit 1
}

$logs = "$PSScriptRoot\logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

if ($Artifact) {
    if (-not (Test-Path $Artifact)) {
        Write-Host "Artifact not found: $($Artifact)"
        exit 1
    }
    New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
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
} else {
    python -c "import openvino_genai" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "openvino_genai not importable in the active environment."
        Write-Host "Activate an env with the build's wheels, or pass -Artifact <build-archive>.zip"
        exit 1
    }
    Write-Host "Using the active python environment."
}

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
