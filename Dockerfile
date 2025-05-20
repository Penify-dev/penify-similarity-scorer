FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models directory for caching
RUN mkdir -p /app/models

# Copy the rest of the application
COPY . .
RUN chmod +x start.sh

# Expose the port the app runs on
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=all-MiniLM-L6-v2

# Command to run the application with Gunicorn
CMD ["./start.sh"]
