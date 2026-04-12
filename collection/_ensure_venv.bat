@echo off
REM Shared by run_masscollect.bat and build_masscollect_exe.bat
REM Expects: ROOT, VENV, SCRIPT_DIR set by caller; cwd should be ROOT.

if exist "%VENV%\Scripts\python.exe" goto install_deps

echo Creating virtual environment: %VENV%
python --version >nul 2>&1
if not errorlevel 1 (
    python -m venv "%VENV%"
    goto venv_created
)
py -3 --version >nul 2>&1
if not errorlevel 1 (
    py -3 -m venv "%VENV%"
    goto venv_created
)
echo No Python found. Install Python 3.12+ and ensure ^"python^" or ^"py^" is on PATH.
exit /b 1

:venv_created
if not exist "%VENV%\Scripts\python.exe" (
    echo Failed to create .venv
    exit /b 1
)

:install_deps
call "%VENV%\Scripts\activate.bat"
python -m pip install -q --upgrade pip
REM Prefer official wheels only for compiled stacks (avoids experimental MinGW NumPy builds).
set "PIP_BIN=--only-binary numpy --only-binary opencv-python"
if /i "%~1"=="build" (
    pip install -q %PIP_BIN% -r "%SCRIPT_DIR%requirements-build.txt"
) else (
    pip install -q %PIP_BIN% -r "%SCRIPT_DIR%requirements.txt"
)
if errorlevel 1 exit /b 1
exit /b 0
