#!/bin/bash

# Space Invaders DQN Frontend Startup Script

echo "🚀 Starting Space Invaders DQN Frontend..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements-frontend.txt

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

# Create necessary directories
mkdir -p models
mkdir -p logs

echo "🎯 Starting backend server..."
# Start backend in background
python app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

echo "🎮 Starting frontend server..."
# Start frontend
npm start &
FRONTEND_PID=$!

echo "✅ Both servers are starting..."
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait

