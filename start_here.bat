@echo off
setlocal
cd /d "%~dp0"

echo JP Equity Backtest Console
echo.
echo This script will:
echo   1. install GUI dependencies if needed
echo   2. launch the desktop GUI
echo.

if not exist ".venv\Scripts\pythonw.exe" (
  call "%~dp0Install_GUI_Dependencies.bat"
  if errorlevel 1 (
    echo Dependency installation did not complete successfully.
    pause
    exit /b 1
  )
)

call "%~dp0OPEN_BACKTEST_GUI.bat"
exit /b %ERRORLEVEL%

endlocal
