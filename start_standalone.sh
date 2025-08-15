#!/bin/bash

# Emergency Intersection Priority System - Standalone Startup Script
# Quick start for the standalone Python files

echo "🚨 Emergency Intersection Priority System - Standalone"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements_standalone.txt

# Create tmp directory if it doesn't exist
mkdir -p /tmp

echo ""
echo "🚀 Starting Emergency Intersection Priority System..."
echo ""

# Start the API server in the background
echo "🌐 Starting API server on port 8000..."
python emergency_intersection_standalone.py --server --port 8000 &
API_PID=$!

# Wait for API to start
echo "⏳ Waiting for API to start..."
sleep 5

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API server is running on http://localhost:8000"
else
    echo "❌ API server failed to start"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎨 Starting Web UI on port 8501..."
echo ""

# Start the Streamlit UI
streamlit run ui_standalone.py --server.port 8501 --server.address 0.0.0.0

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $API_PID 2>/dev/null
    pkill -f streamlit 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for background processes
wait
