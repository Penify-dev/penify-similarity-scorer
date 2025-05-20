#!/bin/bash
# terminate.sh - Script to terminate running similarity-scorer servers

echo "Looking for running similarity-scorer servers..."

# Find the master Gunicorn process (parent process)
echo "Searching for master Gunicorn processes..."
MASTER_PIDS=$(ps -ef | grep "gunicorn" | grep "main:app" | grep -v grep | grep -v "worker" | awk '{print $2}')

# Find worker Gunicorn processes
echo "Searching for Gunicorn worker processes..."
WORKER_PIDS=$(ps -ef | grep "gunicorn" | grep "main:app" | grep "worker" | grep -v grep | awk '{print $2}')

# Find the model service process
echo "Searching for Python model service processes..."
MODEL_SERVICE_PIDS=$(ps aux | grep "model_service.py" | grep -v grep | awk '{print $2}')

# Combine all PIDs
ALL_PIDS="$MASTER_PIDS $WORKER_PIDS $MODEL_SERVICE_PIDS"

# Count the processes
PROCESS_COUNT=$(echo $ALL_PIDS | wc -w | xargs)

if [ "$PROCESS_COUNT" -eq 0 ]; then
    echo "No running similarity-scorer servers found."
    exit 0
fi

# Show the processes
echo "Found $PROCESS_COUNT processes to terminate:"
echo "Master processes: $MASTER_PIDS"
echo "Worker processes: $WORKER_PIDS"
echo "Model service processes: $MODEL_SERVICE_PIDS"

for PID in $ALL_PIDS; do
    echo "PID: $PID - $(ps -p $PID -o command= 2>/dev/null || echo 'Process not found')"
done

# Tell the user we're handling this
echo "Terminating all processes. This may require your password for sudo access..."

# Kill the master processes first (this should prevent auto-restart of workers)
if [ ! -z "$MASTER_PIDS" ]; then
    echo "Killing master processes first to prevent auto-restart..."
    for PID in $MASTER_PIDS; do
        echo "Sending SIGTERM to master PID $PID..."
        sudo kill -15 $PID 2>/dev/null
    done
    sleep 2
fi

# Kill all remaining processes to be sure
for PID in $ALL_PIDS; do
    if ps -p $PID > /dev/null 2>&1; then
        echo "Force killing PID $PID..."
        sudo kill -9 $PID 2>/dev/null
    fi
done

# Final check
sleep 2
STILL_RUNNING=0
for PID in $ALL_PIDS; do
    if ps -p $PID > /dev/null 2>&1; then
        echo "WARNING: Process $PID is still running!"
        STILL_RUNNING=1
    fi
done

if [ $STILL_RUNNING -eq 0 ]; then
    echo "All similarity-scorer processes successfully terminated."
else
    echo "Some processes could not be terminated. The server might be running in a different context."
    echo "Try the following command to see all gunicorn processes:"
    echo "ps aux | grep gunicorn"
    echo "Then manually kill them with:"
    echo "sudo kill -9 <PID>"
fi

# Failsafe - check for any new gunicorn processes
echo "Checking for any new gunicorn processes..."
NEW_GUNICORN=$(ps aux | grep "gunicorn" | grep "main:app" | grep -v grep | awk '{print $2}')
if [ ! -z "$NEW_GUNICORN" ]; then
    echo "New gunicorn processes detected: $NEW_GUNICORN"
    echo "These might be auto-restarted. Killing them with sudo..."
    sudo kill -9 $NEW_GUNICORN 2>/dev/null
fi

# On macOS, use a more aggressive approach since normal SIGTERM might not work
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS. Using sudo to terminate processes. You may be prompted for your password."
    for PID in $ALL_PIDS; do
        echo "Killing PID $PID with sudo..."
        sudo kill -9 $PID 2>/dev/null
    done
else
    # Confirm with user on non-macOS systems
    read -p "Do you want to terminate these processes? (y/n): " CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
        echo "Operation cancelled."
        exit 1
    fi

    # Terminate processes
    echo "Terminating processes..."
    for PID in $ALL_PIDS; do
        echo "Sending SIGTERM to PID $PID..."
        kill -15 $PID 2>/dev/null
    done

    # Wait a moment
    echo "Waiting for processes to terminate gracefully..."
    sleep 3

    # Check if any processes are still running
    REMAINING_PIDS=""
    for PID in $ALL_PIDS; do
        if ps -p $PID > /dev/null; then
            REMAINING_PIDS="$REMAINING_PIDS $PID"
        fi
    done

    # If some processes are still running, force kill them
    if [ ! -z "$REMAINING_PIDS" ]; then
        echo "Some processes are still running. Sending SIGKILL..."
        for PID in $REMAINING_PIDS; do
            echo "Force killing PID $PID..."
            kill -9 $PID 2>/dev/null
        done
    fi
fi

# Final check
sleep 1
STILL_RUNNING=0
for PID in $ALL_PIDS; do
    if ps -p $PID > /dev/null; then
        echo "WARNING: Process $PID is still running!"
        STILL_RUNNING=1
    fi
done

if [ $STILL_RUNNING -eq 0 ]; then
    echo "All similarity-scorer processes successfully terminated."
else
    echo "Some processes could not be terminated. Try running the following command:"
    echo "sudo kill -9 $ALL_PIDS"
fi
