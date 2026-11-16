@echo off
REM Run from the project root:  scripts\setup_windows.bat
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo Setup done. Edit .env and paste your FRED_API_KEY.
