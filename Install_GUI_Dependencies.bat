@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "PYTHON_EXE="
set "PYTHON_LABEL="
set "SCRIPT_DIR=%~dp0"
set "PARENT_PYTHON=%~dp0..\.python\python.exe"

if exist "%SCRIPT_DIR%.python\python.exe" (
  set "PYTHON_EXE=%SCRIPT_DIR%.python\python.exe"
  set "PYTHON_LABEL=local project Python"
) else if exist "%PARENT_PYTHON%" (
  set "PYTHON_EXE=%PARENT_PYTHON%"
  set "PYTHON_LABEL=parent workspace Python"
) else (
  where py >nul 2>nul
  if %ERRORLEVEL%==0 (
    py -3.10 -c "import sys" >nul 2>nul
    if %ERRORLEVEL%==0 (
      set "PYTHON_EXE=py -3.10"
      set "PYTHON_LABEL=Python Launcher 3.10"
    )
  )
  if "%PYTHON_EXE%"=="" (
    where python >nul 2>nul
    if %ERRORLEVEL%==0 (
      set "PYTHON_EXE=python"
      set "PYTHON_LABEL=python on PATH"
    )
  )
)

if "%PYTHON_EXE%"=="" (
  echo No suitable Python 3.10 interpreter was found.
  echo Please install Python 3.10 or place a local interpreter at .python\python.exe and run this script again.
  pause
  exit /b 1
)

if exist "%VENV_DIR%\pyvenv.cfg" (
  findstr /C:"pythoncore-3.14" "%VENV_DIR%\pyvenv.cfg" >nul 2>nul
  if %ERRORLEVEL%==0 (
    echo Detected an incompatible existing .venv created with Python 3.14.
    echo Please delete %CD%\%VENV_DIR% and run this installer again.
    pause
    exit /b 1
  )
)

echo Using %PYTHON_LABEL%...
%PYTHON_EXE% --version
if errorlevel 1 (
  echo Failed to run the selected Python interpreter.
  pause
  exit /b 1
)

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating virtual environment in %CD%\%VENV_DIR% ...
  %PYTHON_EXE% -c "import venv" >nul 2>nul
  if errorlevel 1 (
    echo The selected Python does not provide the built-in venv module.
    echo Trying virtualenv fallback...
    %PYTHON_EXE% -c "import virtualenv" >nul 2>nul
    if errorlevel 1 (
      echo virtualenv is not installed yet. Installing it with pip...
      %PYTHON_EXE% -m pip install virtualenv
      if errorlevel 1 (
        echo Failed to install virtualenv.
        pause
        exit /b 1
      )
    )
    %PYTHON_EXE% -m virtualenv "%VENV_DIR%"
  ) else (
    %PYTHON_EXE% -m venv "%VENV_DIR%"
  )
  if errorlevel 1 (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
  )
) else (
  echo Existing virtual environment found at %CD%\%VENV_DIR%
)

echo Upgrading pip ...
call "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo Failed to upgrade pip.
  pause
  exit /b 1
)

echo Installing Python dependencies ...
if exist "requirements-lock.txt" (
  echo Found requirements-lock.txt. Installing pinned dependency set...
  call "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements-lock.txt
) else (
  call "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 (
  echo Dependency installation failed.
  echo If this is a retry after a broken environment selection, delete .venv and run this script again.
  pause
  exit /b 1
)

echo.
echo Installation completed successfully.
echo You can now launch the GUI with OPEN_BACKTEST_GUI.bat or start_here.bat
pause
exit /b 0

endlocal
