@echo off
echo ==========================================
echo Starting Forensic.AI (Deepfake Detection)
echo ==========================================

:: Start the FastAPI backend in a new command window
echo Starting Backend...
cd backend
start "Forensic.AI Backend" cmd /k ".\venv\Scripts\activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000"
cd ..

:: Start the Vite React frontend in a new command window
echo Starting Frontend...
cd frontend
start "Forensic.AI Frontend" cmd /k "npm run dev"
cd ..

echo.
echo Both servers are starting!
echo - Backend will be available at: http://localhost:8000
echo - Frontend will be available at: http://localhost:5173
echo.
echo Close the newly opened terminal windows to stop the application.
