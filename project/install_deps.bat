@echo off
cd /d "%~dp0"
.\.venv\Scripts\pip.exe install requests google-auth-oauthlib google-auth-httplib2
.\.venv\Scripts\python.exe app.py
pause
