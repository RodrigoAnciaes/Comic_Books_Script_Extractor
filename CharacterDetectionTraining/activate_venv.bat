@echo off
cd %~dp0
set VENV_PATH=ultralytics/ultralytics-venv

echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate"
echo Virtual environment activated.
cmd /k
