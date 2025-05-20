#!/bin/bash
# Simplified startup script for the similarity-scorer service

# Stop any existing processes first
if [ -f "terminate.sh" ]; then
    echo "Stopping any existing similarity-scorer processes..."
    ./terminate.sh
fi

# Set up environment variables
export MODEL_NAME="all-mpnet-base-v2"  # Model to use
export PYTHONPATH=$PWD
export MODEL_APP="model_service:app"
export MAIN_APP="main:app"

# Ensure the models directory exists
mkdir -p models

# Start the application using uvicorn with a single process and async workers
# This prevents multiple instances of the model being loaded
echo "Starting similarity-scorer application..."
uvicorn main:app --host 0.0.0.0 --port 16000 --workers 1 --loop uvloop --http httptools &

echo "Application started on http://localhost:16000"
echo "API documentation available at http://localhost:16000/docs"
