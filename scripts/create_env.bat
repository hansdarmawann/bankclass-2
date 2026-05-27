@echo off
REM Create conda environment and install pip requirements
set ENV_NAME=bankclass-2
set PY=3.11

where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda not found. Install Miniconda/Anaconda or run manually.
    exit /b 1
)

echo Creating conda environment %ENV_NAME% with Python %PY%
conda create -y -n %ENV_NAME% python=%PY% pip

echo Installing pip packages into %ENV_NAME%
conda run -n %ENV_NAME% pip install -r "%~dp0requirements.txt"

echo Environment %ENV_NAME% ready.
pause
