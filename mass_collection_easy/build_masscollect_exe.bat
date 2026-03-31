@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
cd /d "%ROOT%"
set "VENV=%ROOT%\.venv"

call "%SCRIPT_DIR%_ensure_masscollect_venv.bat" build
if errorlevel 1 exit /b 1

call "%VENV%\Scripts\activate.bat"

echo Building masscollect.exe ^(single file^)...
python -m PyInstaller ^
    --onefile ^
    --console ^
    --name masscollect ^
    --clean ^
    --noconfirm ^
    --distpath "%SCRIPT_DIR%dist" ^
    --workpath "%SCRIPT_DIR%build" ^
    --specpath "%SCRIPT_DIR%" ^
    --hidden-import=cv2 ^
    --hidden-import=numpy ^
    --hidden-import=colorama ^
    --hidden-import=serial.tools.list_ports ^
    "%SCRIPT_DIR%masscollect.py"

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo Done: %SCRIPT_DIR%dist\masscollect.exe
endlocal
exit /b 0
