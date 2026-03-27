#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Running with system Python."
fi

# Check if required packages are installed, if not prompt the user but proceed anyway
if ! command -v uvicorn &> /dev/null; then
    echo "Warning: uvicorn is not installed or not in PATH."
    echo "If the server fails to start, try running: pip install -r requirements.txt"
fi

# Start the FastAPI application via uvicorn
echo "Starting Engati AI Assistant..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
