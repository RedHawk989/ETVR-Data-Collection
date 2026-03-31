@echo off
REM If you see "Numpy built with MINGW-W64" or getlimits RuntimeWarnings, your NumPy wheel
REM is wrong for this Python. This reinstalls NumPy from PyPI binary wheels.
REM If it still fails, recreate .venv using python.org 64-bit CPython (not MSYS2).

set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
cd /d "%ROOT%"
set "VENV=%ROOT%\.venv"

if not exist "%VENV%\Scripts\activate.bat" (
    echo No .venv found. Run run_masscollect.bat once to create it.
    exit /b 1
)

call "%VENV%\Scripts\activate.bat"
echo Reinstalling NumPy from wheels only...
pip uninstall -y numpy 2>nul
pip install -q --upgrade pip
pip install -q --only-binary numpy "numpy>=2.1,<3"
if errorlevel 1 (
    echo pip could not install a NumPy wheel for this Python. Install CPython from python.org and recreate .venv.
    exit /b 1
)
echo OK. Run run_masscollect.bat again.
exit /b 0
