#!/bin/bash

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

# Run the Flask app
echo "Starting Flask server..."
echo "Open http://localhost:5000 in your browser"
python app.py
