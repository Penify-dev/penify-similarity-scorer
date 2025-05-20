#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# First, check if an old server is running and terminate it
./force_terminate.sh

# Check if we need to install dependencies first
if [[ "$1" == "--install" ]]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    shift
fi

# Set environment variables
export PYTHONPATH=$DIR
export MODEL_NAME=${MODEL_NAME:-"all-mpnet-base-v2"}

# Configure Gunicorn to use /tmp instead of /dev/shm on macOS
export GUNICORN_CMD_ARGS="--worker-tmp-dir /tmp --max-requests 0"

# Run with gunicorn in production mode
exec gunicorn main:app -c gunicorn_conf.py "$@"
