#!/usr/bin/env pwsh
# Streamlit App Runner Script (PowerShell)
# Run the Credit Score Classification Streamlit App

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Credit Score Classification - Streamlit" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoDir = Resolve-Path (Join-Path $scriptDir "..")
Push-Location $repoDir

$venvPython = Join-Path $repoDir ".venv\Scripts\python.exe"
$pythonExec = $null
$pythonArgs = "-m streamlit run app\streamlit_app.py --logger.level=info"

if (Get-Command conda -ErrorAction SilentlyContinue) {
    $pythonExec = "conda"
    $pythonArgs = "run -n automlnb2017 python -m streamlit run app\streamlit_app.py --logger.level=info"
    Write-Host "Using conda environment automlnb2017 via conda run" -ForegroundColor Green
} elseif (Test-Path $venvPython) {
    $pythonExec = $venvPython
    Write-Host "Using virtualenv: $venvPython" -ForegroundColor Green
    Write-Host "Installing dependencies into .venv..." -ForegroundColor Yellow
    & $pythonExec -m pip install -q -r scripts\requirements.txt
} else {
    $pythonExec = "python"
    Write-Host "Using system Python: $pythonExec" -ForegroundColor Yellow
    Write-Host "NOTE: Make sure automlnb2017 or an equivalent environment is activated." -ForegroundColor Yellow
}

Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop the server`n" -ForegroundColor Yellow

& $pythonExec $pythonArgs

Pop-Location
