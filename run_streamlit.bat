@echo off
REM Run Streamlit app using current Python
cd /d %~dp0
python -m pip install -r requirements.txt
python -m streamlit run src\streamlit_app.py