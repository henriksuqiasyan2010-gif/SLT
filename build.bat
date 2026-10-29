@echo off
rem Build SLT as one folder: dist\SLT\SLT.exe (copy the whole dist\SLT folder to another computer).
rem   build.bat            tests, then the build
rem   build.bat notest     the build only
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
    echo .venv not found: create it and install requirements.txt first.
    exit /b 1
)
if /i not "%~1"=="notest" (
    "%PY%" -m pytest -q
    if errorlevel 1 (
        echo Tests failed: no build.
        exit /b 1
    )
)
"%PY%" -m PyInstaller --noconfirm --clean --log-level WARN slt.spec
if errorlevel 1 (
    echo PyInstaller failed.
    exit /b 1
)
rem The user's Learn-mode progress, so the learned signs show on the other computer too
if exist progress.json copy /y progress.json dist\SLT\progress.json >nul
echo.
echo Built: dist\SLT\SLT.exe
echo Demo without a camera: dist\SLT\SLT.exe --demo 1   (or the ? button - Demo)
endlocal
