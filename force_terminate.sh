#!/bin/bash
# force_terminate.sh - Script to forcefully terminate similarity-scorer servers without confirmation

echo "Force terminating all similarity-scorer processes..."

# Get Process ID list of all Gunicorn processes, including master processes
echo "Finding all gunicorn processes related to similarity-scorer..."

# Kill any process that has 'gunicorn' and 'main:app' in its command line
# This gets the master process which is causing the auto-restart
MASTER_PIDS=$(ps -ef | grep -v grep | grep "gunicorn" | grep "main:app" | awk '{print $2}')
echo "Master processes: $MASTER_PIDS"

# Kill any Python process mentioning model_service
MODEL_PIDS=$(ps -ef | grep -v grep | grep "python" | grep "model_service" | awk '{print $2}')
echo "Model service processes: $MODEL_PIDS"

# Combine all PIDs
ALL_PIDS="$MASTER_PIDS $MODEL_PIDS"

if [ -z "$ALL_PIDS" ]; then
    echo "No similarity-scorer processes found."
    exit 0
fi

echo "Killing master processes first..."
for PID in $MASTER_PIDS; do
    echo "Killing PID $PID with sudo..."
    sudo kill -15 $PID 2>/dev/null
done

# Small delay to let master process handle SIGTERM
sleep 2

# Now forcibly kill everything to be safe
for PID in $ALL_PIDS; do
    if ps -p $PID > /dev/null 2>&1; then
        echo "Force killing PID $PID with sudo..."
        sudo kill -9 $PID 2>/dev/null
    fi
done

# Check for any remaining or new processes after a brief delay
sleep 2
REMAINING_MASTER=$(ps -ef | grep -v grep | grep "gunicorn" | grep "main:app" | awk '{print $2}')
REMAINING_MODEL=$(ps -ef | grep -v grep | grep "python" | grep "model_service" | awk '{print $2}')

if [ ! -z "$REMAINING_MASTER$REMAINING_MODEL" ]; then
    echo "Warning: Some processes are still running or restarted:"
    if [ ! -z "$REMAINING_MASTER" ]; then
        echo "Master processes: $REMAINING_MASTER"
        echo "Attempting to kill with sudo and SIGKILL..."
        sudo kill -9 $REMAINING_MASTER
    fi
    if [ ! -z "$REMAINING_MODEL" ]; then
        echo "Model processes: $REMAINING_MODEL"
        echo "Attempting to kill with sudo and SIGKILL..."
        sudo kill -9 $REMAINING_MODEL
    fi
    
    # One last check
    sleep 1
    FINAL_CHECK=$(ps -ef | grep -v grep | grep "gunicorn\|model_service" | awk '{print $2}')
    if [ ! -z "$FINAL_CHECK" ]; then
        echo "CRITICAL: Unable to terminate all processes. Try the following command:"
        echo "sudo pkill -9 -f 'gunicorn.*main:app'"
    else
        echo "All processes terminated successfully."
    fi
else
    echo "All similarity-scorer processes successfully terminated."
fi
