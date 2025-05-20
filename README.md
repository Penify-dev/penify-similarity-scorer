# Semantic Similarity Scorer

A FastAPI server that compares the semantic similarity between two strings using a sentence-transformers model.

## Features

- Semantic similarity comparison between text strings
- Uses sentence-transformers model (all-MiniLM-L6-v2 by default)
- Automatically utilizes GPU if available
- Simple REST API with JSON request/response

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

1. Start the server:
```bash
uvicorn main:app --reload
```

2. The server will run at: `http://127.0.0.1:8000`

3. Interactive API documentation is available at: `http://127.0.0.1:8000/docs`

### Option 2: Run with Docker

```bash
docker build -t similarity-scorer .
docker run -p 8000:8000 similarity-scorer
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
   docker run -p 8000:8000 -e MODEL_NAME=all-mpnet-base-v2 similarity-scorer
   
   # Or update the environment variable in docker-compose.yml
   # and then run docker-compose up
   ```

2. By directly editing the default in `main.py`:

   ```python
   model_name = os.environ.get("MODEL_NAME", "all-mpnet-base-v2")
   ```

Note that more accurate models like "all-mpnet-base-v2" require more computational resources but provide better similarity results.

## License

[MIT License](LICENSE)
