"""Gunicorn configuration for FastAPI application."""
import multiprocessing
import os

# The socket to bind to
bind = os.getenv("BIND", "0.0.0.0:16000")

# Number of worker processes
# For memory-intensive models, we use a smaller number of workers
# than the typical recommendation of (2 x $num_cores) + 1
# For very limited memory, we use an even more conservative approach
workers = os.getenv("WORKERS", min(multiprocessing.cpu_count(), 2))

# Worker class to use
worker_class = "uvicorn.workers.UvicornWorker"

# The maximum number of pending connections
backlog = 2048

# Timeout in seconds
timeout = 300

# Restart workers when code changes (in development)
reload = os.getenv("RELOAD", "false").lower() in ["true", "1", "t", "y", "yes"]

# Logging configuration
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.getenv("LOG_LEVEL", "info")

# Limit the allowed request line
limit_request_line = 0

# Set process name
proc_name = "similarity-scorer"

# Preload the application
preload_app = True

# Max requests before worker restart to prevent memory leaks
# Set to 0 to disable worker restart (prevents issues with auto-restarting servers)
max_requests = int(os.getenv("MAX_REQUESTS", "0"))
max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "0"))

# Configuration specific to development mode
if reload:
    # Reduce workers in development mode
    workers = 2

# Security settings
# Keep Connection until completion
keepalive = 65
