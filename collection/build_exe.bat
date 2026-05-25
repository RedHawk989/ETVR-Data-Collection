@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "ROOT=%SCRIPT_DIR%.."
cd /d "%ROOT%"
set "VENV=%ROOT%\.venv"

REM Strip trailing backslash for args where a trailing \" would escape the closing quote
set "SCRIPT_DIR_NS=%SCRIPT_DIR:~0,-1%"

call "%SCRIPT_DIR%_ensure_venv.bat" build
if errorlevel 1 exit /b 1

echo Building masscollect.exe (single file, overlay bundled)...
"%VENV%\Scripts\pyinstaller.exe" --clean --noconfirm --distpath "%SCRIPT_DIR_NS%\dist" --workpath "%SCRIPT_DIR_NS%\build" "%SCRIPT_DIR%masscollect.spec"

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo Done: %SCRIPT_DIR%dist\masscollect.exe (overlay bundled inside)
endlocal
exit /b 0
