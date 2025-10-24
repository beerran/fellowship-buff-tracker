@echo off
echo Building Fellowship Buff Tracker Standalone Executable...
echo.

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
    echo.
)

REM Stop any running instances of the executable
echo Stopping any running Fellowship-Buff-Tracker instances...
taskkill /f /im "Fellowship-Buff-Tracker.exe" >nul 2>&1
timeout /t 2 /nobreak >nul

REM Clean previous builds
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del *.spec

REM Build the executable
echo Creating standalone executable...
python -m PyInstaller --onefile --windowed --name "Fellowship-Buff-Tracker" buff_mirror.py

REM Check if build was successful
if exist "dist\Fellowship-Buff-Tracker.exe" (
    echo.
    echo ========================================
    echo BUILD SUCCESS!
    echo ========================================
    echo.
    echo The executable has been created:
    echo   dist\Fellowship-Buff-Tracker.exe
    echo.
    echo You can distribute this single file to users.
    echo No Python installation required!
    echo.
    echo File size: 
    dir "dist\Fellowship-Buff-Tracker.exe" | find "Fellowship-Buff-Tracker.exe"
    echo.
) else (
    echo.
    echo BUILD FAILED!
    echo Check the output above for errors.
    echo.
)

pause