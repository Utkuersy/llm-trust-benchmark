@echo off
cd /d "%~dp0"
rem Dashboard sifresi icin: python -m core.dashboard_auth ile bir hash uretip
rem asagidaki satiri "set AITB__DASHBOARD__PASSWORD_HASH=<hash>" olarak
rem doldurun. Bos birakilirsa dashboard erisimi tamamen reddeder (fail-closed).
".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless true
