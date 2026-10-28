@echo off
:: run.bat — Start the dashboard (Windows)
call .venv\Scripts\activate.bat
echo  Starting Pneumonia Detection Dashboard...
echo  Browser will open automatically at http://127.0.0.1:7860
echo  Press Ctrl+C to stop.
echo.
python app.py
pause
