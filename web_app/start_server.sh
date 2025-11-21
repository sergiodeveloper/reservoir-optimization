#!/bin/bash

# Kill any existing Flask processes on port 5003
echo "Cleaning up existing processes..."
lsof -ti:5003 | xargs kill -9 2>/dev/null
pkill -9 -f "python.*app.py" 2>/dev/null
# Also kill stopped processes
ps aux | grep "python.*app.py" | grep " T " | awk '{print $2}' | xargs kill -9 2>/dev/null
sleep 2

# Change to the web_app directory
cd "$(dirname "$0")"

# Activate virtual environment
source ../venv/bin/activate

# Start the server
echo "Starting Flask server on port 5003..."
python3 app.py
