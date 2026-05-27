# Create conda environment and install pip requirements (PowerShell)
$envName = 'automlnb2017'
$py = '3.11'

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "conda not found. Install Miniconda/Anaconda or run manually."
    exit 1
}

Write-Output "Creating conda environment $envName with Python $py"
conda create -y -n $envName python=$py pip

Write-Output "Installing pip packages into $envName"
conda run -n $envName pip install -r "$PSScriptRoot\requirements.txt"

Write-Output "Environment $envName ready."
