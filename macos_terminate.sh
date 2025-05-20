#!/bin/bash
# macos_terminate.sh - macOS-specific script to terminate all similarity-scorer processes

echo "macOS-specific termination script for similarity-scorer"
echo "This script uses pkill to terminate all related processes"

# First try a gentle approach
echo "Attempting gentle termination with SIGTERM..."
pkill -f "gunicorn.*main:app"
pkill -f "model_service"

# Wait a bit
sleep 2

# Now use force
echo "Force terminating with SIGKILL (requires password)..."
sudo pkill -9 -f "gunicorn.*main:app"
sudo pkill -9 -f "model_service"

# Check if processes are still running
sleep 1
REMAINING=$(pgrep -f "gunicorn.*main:app\|model_service")

if [ -z "$REMAINING" ]; then
    echo "All similarity-scorer processes successfully terminated."
else
    echo "Warning: Some processes are still running: $REMAINING"
    echo "Try manually terminating them with:"
    echo "sudo kill -9 $REMAINING"
    
    # One last attempt with direct commands
    echo "Attempting one final termination method..."
    sudo killall -9 gunicorn 2>/dev/null
    sudo killall -9 python 2>/dev/null
    
    echo "Termination attempts completed. Please verify no processes remain with:"
    echo "ps aux | grep 'gunicorn\\|model_service' | grep -v grep"
fi
