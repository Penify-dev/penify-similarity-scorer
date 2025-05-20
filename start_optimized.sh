#!/bin/bash
# Start script for the Similarity Scorer application with explicit memory monitoring

# Set environment variables for better memory management
export MALLOC_ARENA_MAX=2  # Limit memory arenas for glibc malloc (helps reduce memory fragmentation)
export PYTHONMALLOC=malloc  # Use system malloc instead of Python's pymalloc (can be more memory efficient)

# Optional: Force garbage collection more aggressively
export PYTHONDEVMODE=1
export PYTHONGC="threshold=10,autoscale=5"  # More aggressive GC settings

# Log memory usage
echo "Starting server with memory tracking..."
if command -v free >/dev/null 2>&1; then
    echo "Initial system memory:"
    free -h
fi

# Start the server with Gunicorn
echo "Starting server with optimized single-model configuration"
gunicorn -c gunicorn_conf.py main:app

# Log memory usage again on exit
if command -v free >/dev/null 2>&1; then
    echo "Final system memory:"
    free -h
fi
