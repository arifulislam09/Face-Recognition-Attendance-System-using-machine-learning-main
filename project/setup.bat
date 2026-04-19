@echo off
REM Setup script for Face Recognition Attendance System
REM Run this after installing Python 3.11

cd /d "%~dp0"

echo Creating virtual environment with Python 3.11...
py -3.11 -m venv .venv

echo Activating virtual environment...
call .\.venv\Scripts\activate.bat

echo Upgrading pip, setuptools, wheel...
python -m pip install --upgrade pip setuptools wheel

echo Installing dependencies...
python -m pip install -r requirements.txt

echo Setup complete! Run 'python app.py' to start the app.
pause