# API Reference

Complete reference for all REST API endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required (development mode). See deployment guide for production authentication setup.

---

## Endpoints

### GET /

Get API information and available endpoints.

**Response:**
```json
{
  "message": "Document QA API with RAGAS Evaluation & HyDE",
  "version": "3.0.0",
  "features": {
    "rag": "Retrieval Augmented Generation",
    "evaluation": "RAGAS metrics with MLflow",
    "hyde": "Hypothetical Document Embeddings",
    "conversation": "Session-based chat history"
  },
  "endpoints": {
    "POST /upload": "Upload a PDF document",
    "POST /query": "Ask a question about the document",
    "POST /evaluate": "Evaluate the RAG pipeline with test cases",
    "POST /evaluate/single": "Evaluate a single query",
    "GET /evaluate/history": "Get evaluation history",
    "POST /reset": "Reset conversation history",
    "GET /health": "Check API health status"
  }
}
```

---

### GET /health

Check system health and component status.

**Response:**
```json
{
  "status": "healthy",
  "weaviate_connected": true,
  "llm_ready": true,
  "embeddings_ready": true,
  "mlflow_tracking_uri": "file:./mlruns",
  "hyde_available": true,
  "using_hyde": false
}
```

**Status Codes:**
- `200 OK` - System healthy
- `503 Service Unavailable` - System initializing or degraded

---

### POST /upload

Upload and process a PDF document.

**Request:**

Content-Type: `multipart/form-data`

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

**Parameters:**
- `file` (required): PDF file to upload

**Response:**
```json
{
  "message": "PDF uploaded and processed successfully",
  "filename": "document.pdf",
  "pages": 25,
  "chunks": 142
}
```

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid file type
- `500 Internal Server Error` - Processing error

**Processing Steps:**
1. PDF loaded and parsed
2. Text split into chunks (512 chars, 100 overlap)
3. Chunks embedded using BGE model
4. Vectors stored in Weaviate
5. RAG chain initialized

---

### POST /query

Ask a question about the uploaded document.

**Request:**

Content-Type: `application/json`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic?",
    "session_id": "user123",
    "use_hyde": false
  }'
```

**Request Body:**
```json
{
  "question": "string (required)",
  "session_id": "string (optional, default: 'default_session')",
  "use_hyde": "boolean (optional, default: false)"
}
```

**Parameters:**
- `question` (required): The question to ask
- `session_id` (optional): Session identifier for conversation context
- `use_hyde` (optional): Enable HyDE retrieval enhancement

**Response:**
```json
{
  "question": "What is the main topic?",
  "answer": "The document discusses artificial intelligence and its applications in modern technology...",
  "session_id": "user123",
  "retrieval_method": "standard",
  "sources": [
    {
      "content": "Artificial intelligence (AI) is transforming...",
      "metadata": {
        "source": "document.pdf",
        "page": 3
      }
    }
  ]
}
```

**Response Fields:**
- `question`: The original question
- `answer`: Generated answer
- `session_id`: Session identifier used
- `retrieval_method`: "standard" or "hyde"
- `sources`: Array of source documents with content and metadata

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - No document loaded
- `500 Internal Server Error` - Processing error

**Notes:**
- Maintains conversation history per session_id
- Follow-up questions can reference previous context
- HyDE adds ~2s latency but improves accuracy

---

### POST /evaluate

Evaluate the RAG pipeline with multiple test cases.

**Request:**

Content-Type: `application/json`

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "question": "What is X?",
        "ground_truth": "X is..."
      }
    ],
    "run_name": "baseline_eval",
    "use_hyde": false
  }'
```

**Request Body:**
```json
{
  "test_cases": [
    {
      "question": "string (required)",
      "ground_truth": "string (optional)"
    }
  ],
  "run_name": "string (optional)",
  "use_hyde": "boolean (optional, default: false)"
}
```

**Parameters:**
- `test_cases` (required): Array of test cases
  - `question` (required): Question to evaluate
  - `ground_truth` (optional): Expected answer for additional metrics
- `run_name` (optional): Name for this evaluation run
- `use_hyde` (optional): Use HyDE retrieval

**Response:**
```json
{
  "scores": {
    "faithfulness": 0.8523,
    "answer_relevancy": 0.9123,
    "context_precision": 0.7845,
    "context_recall": 0.8234,
    "answer_similarity": 0.8567,
    "answer_correctness": 0.8401
  },
  "mlflow_run_id": "abc123def456",
  "num_test_cases": 5,
  "detailed_results": [
    {
      "question": "What is X?",
      "answer": "X is...",
      "contexts": ["Context 1", "Context 2"],
      "faithfulness": 0.90,
      "answer_relevancy": 0.95
    }
  ]
}
```

**Response Fields:**
- `scores`: Average metric scores across all test cases
- `mlflow_run_id`: MLflow run identifier for tracking
- `num_test_cases`: Number of test cases evaluated
- `detailed_results`: Per-question results

**Metrics Computed:**

Without ground truth:
- `faithfulness`: Answer supported by context (0-1)
- `answer_relevancy`: Answer relevance to question (0-1)
- `context_precision`: Quality of retrieved context (0-1)

With ground truth (additional):
- `context_recall`: Completeness of retrieval (0-1)
- `answer_similarity`: Semantic match to ground truth (0-1)
- `answer_correctness`: Overall correctness (0-1)

**Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - No document loaded or invalid request
- `503 Service Unavailable` - Evaluation system not available

**Processing Time:**
- ~5-10 seconds per test case (standard)
- ~8-15 seconds per test case (HyDE)

---

### POST /evaluate/single

Evaluate a single query-answer pair.

**Request:**

```bash
curl -X POST http://localhost:8000/evaluate/single \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "answer": "RAG stands for Retrieval Augmented Generation...",
    "contexts": ["Context 1", "Context 2"],
    "ground_truth": "RAG is..."
  }'
```

**Request Body:**
```json
{
  "question": "string (required)",
  "answer": "string (required)",
  "contexts": ["string"] (required),
  "ground_truth": "string (optional)"
}
```

**Parameters:**
- `question` (required): The question asked
- `answer` (required): The generated answer
- `contexts` (required): Array of context strings used
- `ground_truth` (optional): Expected answer

**Response:**
```json
{
  "question": "What is RAG?",
  "scores": {
    "faithfulness": 0.90,
    "answer_relevancy": 0.95,
    "context_precision": 0.85
  },
  "message": "Single query evaluated successfully"
}
```

**Status Codes:**
- `200 OK` - Success
- `503 Service Unavailable` - Evaluator not initialized

**Use Case:**
Quick testing of specific query-answer pairs without full pipeline execution.

---

### GET /evaluate/history

Get history of all evaluation runs.

**Request:**
```bash
curl http://localhost:8000/evaluate/history
```

**Response:**
```json
{
  "message": "Found 10 evaluation runs",
  "runs": [
    {
      "run_id": "abc123",
      "run_name": "baseline_eval",
      "start_time": 1701234567890,
      "num_samples": 5,
      "has_ground_truth": true,
      "faithfulness": 0.85,
      "answer_relevancy": 0.91,
      "context_precision": 0.78
    }
  ]
}
```

**Response Fields:**
- `message`: Summary message
- `runs`: Array of evaluation run objects with:
  - `run_id`: MLflow run identifier
  - `run_name`: Name of the run
  - `start_time`: Timestamp (milliseconds)
  - Metric scores

**Status Codes:**
- `200 OK` - Success
- `503 Service Unavailable` - Evaluator not initialized

---

### GET /evaluate/compare

Compare multiple evaluation runs.

**Request:**
```bash
curl "http://localhost:8000/evaluate/compare?run_ids=run1,run2,run3"
```

**Query Parameters:**
- `run_ids` (required): Comma-separated list of MLflow run IDs

**Response:**
```json
{
  "message": "Comparing 3 runs",
  "comparison": [
    {
      "run_id": "run1",
      "start_time": 1701234567890,
      "faithfulness": 0.85,
      "answer_relevancy": 0.91
    },
    {
      "run_id": "run2",
      "start_time": 1701234567900,
      "faithfulness": 0.88,
      "answer_relevancy": 0.93
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Success
- `500 Internal Server Error` - Comparison error

---

### POST /reset

Reset conversation history for a session.

**Request:**
```bash
curl -X POST "http://localhost:8000/reset?session_id=user123"
```

**Query Parameters:**
- `session_id` (optional): Session to reset (default: "default_session")

**Response:**
```json
{
  "message": "Session 'user123' reset successfully"
}
```

**Status Codes:**
- `200 OK` - Success (even if session didn't exist)

**Use Case:**
Clear conversation history to start fresh or test different contexts.

---

### GET /debug/status

Debug endpoint to check initialization status.

**Request:**
```bash
curl http://localhost:8000/debug/status
```

**Response:**
```json
{
  "chatbot_initialized": true,
  "evaluator_initialized": true,
  "evaluation_available": true,
  "chatbot_ready": {
    "weaviate_connected": true,
    "llm_ready": true,
    "embeddings_ready": true,
    "rag_ready": true,
    "evaluator_ready": true
  },
  "rag_chain_ready": true,
  "mlflow_uri": "file:./mlruns"
}
```

**Status Codes:**
- `200 OK` - Always returns status

**Use Case:**
Troubleshooting initialization issues.

---

## Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common HTTP Status Codes

- `200 OK` - Request successful
- `400 Bad Request` - Invalid request (missing parameters, wrong format)
- `500 Internal Server Error` - Server-side error
- `503 Service Unavailable` - Service not initialized or unavailable

---

## Rate Limiting

Currently no rate limiting (development mode). For production:
- Recommend: 100 requests/minute per IP
- Evaluation endpoints: 10 requests/minute per IP

---

## Interactive Documentation

**Swagger UI:** http://localhost:8000/docs
- Interactive API testing
- Request/response examples
- Schema documentation
- Try endpoints directly

**ReDoc:** http://localhost:8000/redoc
- Alternative documentation view
- Better for reading/printing

---

## Code Examples

### Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": f}
    )
print(response.json())

# Query with HyDE
response = requests.post(
    f"{BASE_URL}/query",
    json={
        "question": "What is the main topic?",
        "use_hyde": True
    }
)
print(response.json()["answer"])

# Evaluate
response = requests.post(
    f"{BASE_URL}/evaluate",
    json={
        "test_cases": [
            {"question": "Q1", "ground_truth": "A1"}
        ],
        "run_name": "test"
    }
)
print(response.json()["scores"])
```

### JavaScript

```javascript
const BASE_URL = "http://localhost:8000";

// Upload document
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const uploadResponse = await fetch(`${BASE_URL}/upload`, {
  method: "POST",
  body: formData
});

// Query
const queryResponse = await fetch(`${BASE_URL}/query`, {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({
    question: "What is the main topic?",
    use_hyde: true
  })
});

const result = await queryResponse.json();
console.log(result.answer);
```

### cURL

```bash
# Upload
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is X?", "use_hyde": true}'

# Evaluate
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{"question": "Q1", "ground_truth": "A1"}],
    "run_name": "test"
  }'
```

---

## Webhooks

Not currently supported. Contact support for custom integration needs.

---

## Versioning

Current API version: `3.0.0`

Version format: `MAJOR.MINOR.PATCH`
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

---

## Support

- **API Issues:** Check `/health` and `/debug/status`
- **Documentation:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000

---

**Last Updated:** December 2024  
**API Version:** 3.0.0
