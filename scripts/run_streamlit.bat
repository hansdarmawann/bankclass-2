@echo off
REM Streamlit App Runner Script
REM Run the Credit Score Classification Streamlit App

set SCRIPT_DIR=%~dp0
REM Use repository root (one level up from scripts) so app path resolves
set REPO_DIR=%SCRIPT_DIR%..\
cd /d "%REPO_DIR%"

set PYTHON_CMD=python
set PYTHON_ARGS=-m streamlit run app\streamlit_app.py --logger.level=info

where conda >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON_CMD=conda
    set PYTHON_ARGS=run -n automlnb2017 python -m streamlit run app\streamlit_app.py --logger.level=info
    echo Using conda environment automlnb2017 via conda run
    goto START_APP
)

set VENV_PYTHON=%REPO_DIR%.venv\Scripts\python.exe
if exist "%VENV_PYTHON%" (
    set PYTHON_CMD=%VENV_PYTHON%
    set PYTHON_ARGS=-m streamlit run app\streamlit_app.py --logger.level=info
    echo Using virtualenv: %VENV_PYTHON%
    echo Installing dependencies into .venv...
    "%PYTHON_CMD%" -m pip install -q -r "%REPO_DIR%scripts\requirements.txt"
) else (
    echo WARNING: .venv not found. If Streamlit or dependencies are missing, create a venv or use automlnb2017:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   python -m pip install -r scripts\requirements.txt
)

:START_APP

echo.
echo Starting Streamlit app...
echo URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

"%PYTHON_CMD%" %PYTHON_ARGS%

pause
