@echo off
setlocal
cd /d "%~dp0"

set "PYW="
if exist "%~dp0.python\pythonw.exe" set "PYW=%~dp0.python\pythonw.exe"
if not defined PYW if exist "%~dp0..\.python\pythonw.exe" set "PYW=%~dp0..\.python\pythonw.exe"
if not defined PYW if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYW=%~dp0.venv\Scripts\pythonw.exe"

if not defined PYW (
  echo No suitable pythonw.exe was found.
  echo.
  echo Tried:
  echo   %~dp0.python\pythonw.exe
  echo   %~dp0..\.python\pythonw.exe
  echo   %~dp0.venv\Scripts\pythonw.exe
  echo.
  echo Running first-time dependency setup...
  call "%~dp0Install_GUI_Dependencies.bat"
  if errorlevel 1 (
    echo Setup did not complete successfully.
    pause
    exit /b 1
  )
  if exist "%~dp0.python\pythonw.exe" set "PYW=%~dp0.python\pythonw.exe"
  if not defined PYW if exist "%~dp0..\.python\pythonw.exe" set "PYW=%~dp0..\.python\pythonw.exe"
  if not defined PYW if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYW=%~dp0.venv\Scripts\pythonw.exe"
)

if not defined PYW (
  echo GUI launch failed: pythonw.exe is still unavailable.
  pause
  exit /b 1
)

echo Opening the GUI from:
echo   %CD%
echo.
echo Launch notes:
echo   - Window title: JP Equity Backtest Console
echo   - Uses direct linear factor weighting
echo   - J-Quants credentials are entered directly in the GUI
echo   - 12-1 momentum is used instead of residual momentum
echo   - Python: %PYW%
start "JP Equity Backtest Console" "%PYW%" "runtime\run_gui.py"
exit /b 0

endlocal
