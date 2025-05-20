# Semantic Similarity Scorer

A FastAPI server that compares the semantic similarity between two strings using a sentence-transformers model.

## Features

- Semantic similarity comparison between text strings
- Uses sentence-transformers model (all-MiniLM-L6-v2 by default)
- Automatically utilizes GPU if available
- Simple REST API with JSON request/response
- Model caching to avoid repeated downloads on server restart

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/similarity-scorer.git
cd similarity-scorer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run directly with Python

#### Development Mode
```bash
uvicorn main:app --reload
```

#### Production Mode with Gunicorn
```bash
./start.sh
```

You can customize settings by setting environment variables:
```bash
MODEL_NAME=all-mpnet-base-v2 WORKERS=4 ./start.sh
```

The server will run at: `http://127.0.0.1:16000`

Interactive API documentation is available at: `http://127.0.0.1:16000/docs`

### Option 2: Run with Docker

```bash
docker build -t similarity-scorer .
docker run -p 16000:16000 similarity-scorer
```

### Option 3: Run with Docker Compose

```bash
docker-compose up
```

To run in detached mode:
```bash
docker-compose up -d
```

To stop the service:
```bash
docker-compose down
```

## Server Management

### Starting the Server

To start the server in production mode:

```bash
./start.sh
```

For memory-constrained environments:

```bash
./start_optimized.sh
```

### Stopping the Server

To stop any running server instances:

```bash
./terminate.sh
```

If regular termination doesn't work, use the force termination script (may require sudo):

```bash
sudo ./force_terminate.sh
```

These scripts will:

1. Identify all running processes related to the similarity scorer
2. Attempt to terminate them gracefully (regular script) or forcefully (force script)
3. Report the status after termination attempt

## API Endpoints

### POST /compare

Compares the semantic similarity between two sentences.

**Request Body:**
```json
{
  "sentence1": "How do I bake a cake?",
  "sentence2": "What is the process for making a cake?"
}
```

**Response:**
```json
{
  "sentence1": "How do I bake a cake?",
  "sentence2": "What is the process for making a cake?",
  "semantic_similarity": 0.87
}
```

## Customization

### Changing the Model

You can change the model in two ways:

1. By setting the `MODEL_NAME` environment variable:

   ```bash
   # When running with Python
   MODEL_NAME=all-mpnet-base-v2 uvicorn main:app --reload
   
   # When running with Docker
   docker run -p 16000:16000 -e MODEL_NAME=all-mpnet-base-v2 similarity-scorer
   
   # Or update the environment variable in docker-compose.yml
   # and then run docker-compose up
   ```

2. By directly editing the default in `main.py`:

   ```python
   model_name = os.environ.get("MODEL_NAME", "all-mpnet-base-v2")
   ```

Note that more accurate models like "all-mpnet-base-v2" require more computational resources but provide better similarity results.

## Performance Considerations

### Model Caching

The application is configured to cache downloaded models in the `models/` directory. This means:

1. The model will only be downloaded once, even if you restart the server multiple times
2. Subsequent server startups will be much faster
3. When using Docker, the model cache is stored in a named volume for persistence

This is especially important for larger models like "all-mpnet-base-v2" which can be several hundred MB in size.

### Shared Model Process

To optimize memory usage, the application uses a dedicated model service that runs in a separate process:

1. Only one copy of the model is loaded in memory regardless of how many Gunicorn workers are running
2. All worker processes communicate with the model service via IPC (inter-process communication)
3. This significantly reduces memory usage when running with multiple workers

This architecture is particularly beneficial for large models that would otherwise consume several GB of RAM if loaded separately in each worker process.

## Memory Optimization

This application is designed for memory efficiency, especially in environments with limited resources:

1. **Single Model Instance**: The model is loaded only once in a dedicated process and shared across all workers
2. **Conservative Worker Count**: The Gunicorn configuration uses fewer workers than typical to reduce memory usage
3. **Memory Monitoring**: The application logs memory usage at various points to track resource utilization
4. **Configurable Worker Count**: You can set the `WORKERS` environment variable to further limit workers
5. **Worker Lifecycle Management**: Workers are restarted after handling a certain number of requests to prevent memory leaks

### Optimized Startup

For environments with very limited memory, use the optimized startup script:

```bash
./start_optimized.sh
```

This script sets environment variables to optimize memory usage and provides memory usage tracking.

### GPU Support

The application automatically detects and uses available GPU resources:

#### macOS GPU Support (Apple Silicon/AMD)

On macOS, the application uses:
- Apple's Metal Performance Shaders (MPS) backend on Apple Silicon (M1/M2/M3) Macs
- AMD GPUs on Intel Macs with compatible graphics cards

To check if your Mac is using GPU acceleration:
1. Start the server
2. Visit `http://127.0.0.1:16000/system-info` to see device information
3. Look for `"mps_available": true` and `"current_device": "mps"` in the response

#### NVIDIA GPU Support (Linux/Windows)

On systems with NVIDIA GPUs, CUDA will be used automatically.

If running in Docker, make sure to include the GPU runtime configuration as specified in the docker-compose.yml file.

## License

[MIT License](LICENSE)
