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
$pythonArgs = @("-m", "streamlit", "run", "app\streamlit_app.py", "--logger.level=info", "--server.headless=true")
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

if (Get-Command conda -ErrorAction SilentlyContinue) {
    $hasEnv = conda env list | Select-String -Pattern "^\s*bankclass-2\s"
    if ($hasEnv) {
        $pythonExec = "conda"
        $pythonArgs = @("run", "-n", "bankclass-2", "python", "-m", "streamlit", "run", "app\streamlit_app.py", "--logger.level=info")
        Write-Host "Using conda environment bankclass-2 via conda run" -ForegroundColor Green
    } elseif (Test-Path $venvPython) {
        $pythonExec = $venvPython
        Write-Host "Conda found, but environment bankclass-2 not found. Falling back to virtualenv: $venvPython" -ForegroundColor Yellow
        Write-Host "Installing dependencies into .venv..." -ForegroundColor Yellow
        & $pythonExec -m pip install -q -r scripts\requirements.txt
    } else {
        $pythonExec = "python"
        Write-Host "Conda found, but environment bankclass-2 not found. Using system Python." -ForegroundColor Yellow
        Write-Host "NOTE: Create the expected environment with .\scripts\create_env.ps1 if needed." -ForegroundColor Yellow
    }
} elseif (Test-Path $venvPython) {
    $pythonExec = $venvPython
    Write-Host "Using virtualenv: $venvPython" -ForegroundColor Green
    Write-Host "Installing dependencies into .venv..." -ForegroundColor Yellow
    & $pythonExec -m pip install -q -r scripts\requirements.txt
} else {
    $pythonExec = "python"
    Write-Host "Using system Python: $pythonExec" -ForegroundColor Yellow
    Write-Host "NOTE: Make sure bankclass-2 or an equivalent environment is activated." -ForegroundColor Yellow
}

Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host "URL: http://localhost:8501" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop the server`n" -ForegroundColor Yellow

& $pythonExec @pythonArgs

Pop-Location
