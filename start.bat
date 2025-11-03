@echo off
REM Space Invaders DQN Frontend Startup Script for Windows

echo 🚀 Starting Space Invaders DQN Frontend...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 16+ first.
    pause
    exit /b 1
)

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm is not installed. Please install npm first.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed

REM Install Python dependencies
echo 📦 Installing Python dependencies...
pip install -r requirements-frontend.txt

REM Install Node.js dependencies
echo 📦 Installing Node.js dependencies...
npm install

REM Create necessary directories
if not exist models mkdir models
if not exist logs mkdir logs

echo 🎯 Starting backend server...
REM Start backend in background
start /B python app.py

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

echo 🎮 Starting frontend server...
REM Start frontend
npm start

echo ✅ Both servers are starting...
echo 🌐 Frontend: http://localhost:3000
echo 🔧 Backend API: http://localhost:5000
echo.
echo Press any key to stop both servers
pause >nul

