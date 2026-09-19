@echo off
cd /d "%~dp0"
rem For the dashboard password: generate a hash with python -m core.dashboard_auth
rem and fill in the line below as "set AITB__DASHBOARD__PASSWORD_HASH=<hash>".
rem If left empty, the dashboard fully denies access (fail-closed).
".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless true
