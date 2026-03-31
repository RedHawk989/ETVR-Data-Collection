@echo off
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
cd /d "%ROOT%"
set "VENV=%ROOT%\.venv"

call "%SCRIPT_DIR%_ensure_masscollect_venv.bat"
if errorlevel 1 exit /b 1

call "%VENV%\Scripts\activate.bat"
echo Running masscollect...
python "%SCRIPT_DIR%masscollect.py"
exit /b %ERRORLEVEL%
