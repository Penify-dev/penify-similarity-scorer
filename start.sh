#!/bin/bash

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if we need to install dependencies first
if [[ "$1" == "--install" ]]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    shift
fi

# Set environment variables
export PYTHONPATH=$DIR
export MODEL_NAME=${MODEL_NAME:-"all-mpnet-base-v2"}

# Run with gunicorn in production mode
exec gunicorn main:app -c gunicorn_conf.py "$@"
