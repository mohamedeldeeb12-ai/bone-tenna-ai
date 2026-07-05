@echo off
TITLE Bone-Tenna SAR Microwave Diagnostic Launcher
ECHO ========================================================
ECHO Initializing Bone-Tenna SAR Tomographic Diagnostics...
ECHO Please wait while Streamlit server starts...
ECHO ========================================================

:: Ensure we are working in the current script directory
cd /d "%~dp0"

:: Using 'python -m streamlit' bypasses Windows PATH issues completely
python -m streamlit run sar_bone_app.py

PAUSE