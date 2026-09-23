# run_preprocess.ps1 - Execute the Multimodal CRISPR Preprocessing Pipeline in PowerShell
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

if (-not $env:CRISPR_DATA_DIR) {
    $env:CRISPR_DATA_DIR = Join-Path $ProjectRoot "data"
}

Write-Host "==========================================================================="
Write-Host "Running Multimodal CRISPR Preprocessing Pipeline (PowerShell)"
Write-Host "Project Root: $ProjectRoot"
Write-Host "Data Root   : $env:CRISPR_DATA_DIR"
Write-Host "==========================================================================="

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    & $VenvPython utils/preprocess.py @args
} else {
    python utils/preprocess.py @args
}
