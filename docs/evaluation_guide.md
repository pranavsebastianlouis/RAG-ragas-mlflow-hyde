# Evaluation Guide

## 📊 Overview

This guide covers the comprehensive evaluation system built with RAGAS (Retrieval Augmented Generation Assessment) and MLflow tracking for measuring and improving RAG pipeline performance.

## 🎯 RAGAS Metrics Explained

### Metrics Without Ground Truth

These metrics can be computed without reference answers:

#### 1. Faithfulness (0-1, higher is better)

**What it measures:** Whether the answer is factually consistent with the retrieved context.

**How it works:**
- LLM identifies claims in the answer
- Checks if each claim is supported by context
- Score = (Supported claims) / (Total claims)

**Example:**
```
Context: "Python was created by Guido van Rossum in 1991."
Answer: "Python was created in 1991 by Guido." 
Faithfulness: 1.0 ✅ (fully supported)

Answer: "Python was created in 1995 by Guido."
Faithfulness: 0.5 ❌ (date is wrong)
```

**Good Score:** > 0.8

#### 2. Answer Relevancy (0-1, higher is better)

**What it measures:** How relevant the answer is to the question asked.

**How it works:**
- Computes semantic similarity between question and answer
- Uses embedding-based comparison
- Score = Cosine similarity of embeddings

**Example:**
```
Question: "What is the capital of France?"
Answer: "The capital of France is Paris."
Relevancy: 0.95 ✅ (highly relevant)

Answer: "France is a country in Europe with many cities."
Relevancy: 0.60 ❌ (related but doesn't answer)
```

**Good Score:** > 0.8

#### 3. Context Precision (0-1, higher is better)

**What it measures:** Whether the retrieved contexts are relevant to answering the question.

**How it works:**
- LLM judges if each retrieved chunk is useful
- Score = (Relevant chunks) / (Total retrieved chunks)

**Example:**
```
Question: "What causes climate change?"
Retrieved Chunks:
1. "Climate change is caused by greenhouse gases..." ✅ Relevant
2. "The greenhouse effect traps heat..." ✅ Relevant  
3. "Weather varies by season..." ❌ Not relevant
4. "Climate models use complex algorithms..." ⚠️ Tangentially relevant

Context Precision: 0.75
```

**Good Score:** > 0.7

### Metrics Requiring Ground Truth

These metrics need reference answers for comparison:

#### 4. Context Recall (0-1, higher is better)

**What it measures:** Whether all necessary information was retrieved.

**How it works:**
- Compares retrieved context to ground truth
- Checks if ground truth info is in retrieved chunks
- Score = (Retrieved relevant info) / (All relevant info in ground truth)

**Example:**
```
Ground Truth: "X is caused by Y and Z"
Retrieved Context: Discusses Y and Z ✅
Context Recall: 1.0 (complete)

Retrieved Context: Only discusses Y ❌
Context Recall: 0.5 (incomplete)
```

**Good Score:** > 0.7

#### 5. Answer Similarity (0-1, higher is better)

**What it measures:** Semantic similarity between generated answer and ground truth.

**How it works:**
- Embeds both answers
- Computes cosine similarity
- Focuses on meaning, not exact wording

**Example:**
```
Ground Truth: "Paris is the capital of France"
Answer: "France's capital city is Paris"
Similarity: 0.95 ✅ (same meaning)

Answer: "Paris is a major French city"
Similarity: 0.70 ⚠️ (related but incomplete)
```

**Good Score:** > 0.7

#### 6. Answer Correctness (0-1, higher is better)

**What it measures:** Overall correctness combining factual accuracy and semantic similarity.

**How it works:**
- Combines exact match scoring and semantic similarity
- Weighted combination: accuracy + similarity
- Most comprehensive metric

**Good Score:** > 0.7

## 🚀 Using the Evaluation API

### Step 1: Health Check

```bash
curl http://localhost:8000/health
```

**Expected Response:**
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

### Step 2: Upload Your Document

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@research_paper.pdf"
```

### Step 3: Create Test Cases

Create `test_cases.json`:

```json
{
  "test_cases": [
    {
      "question": "What is the main research question?",
      "ground_truth": "The study investigates the relationship between X and Y"
    },
    {
      "question": "What methodology was used?",
      "ground_truth": "A mixed-methods approach combining surveys and interviews"
    },
    {
      "question": "What are the key findings?",
      "ground_truth": "The findings show that X significantly affects Y"
    },
    {
      "question": "Who are the authors?"
      // No ground_truth - will compute 3 metrics instead of 6
    }
  ],
  "run_name": "baseline_evaluation",
  "use_hyde": false
}
```

**Note:** Ground truth is optional but enables more comprehensive metrics.

### Step 4: Run Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @test_cases.json
```

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
  "num_test_cases": 4,
  "detailed_results": [
    {
      "question": "What is the main research question?",
      "answer": "The study examines...",
      "faithfulness": 0.90,
      "answer_relevancy": 0.95,
      "context_precision": 0.85,
      "context_recall": 0.88,
      "answer_similarity": 0.92,
      "answer_correctness": 0.90
    }
  ]
}
```

### Step 5: Evaluate Single Query

For quick testing:

```bash
curl -X POST http://localhost:8000/evaluate/single \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "answer": "RAG stands for Retrieval Augmented Generation, a technique that combines retrieval with generation.",
    "contexts": [
      "Retrieval Augmented Generation (RAG) enhances LLMs by retrieving relevant information.",
      "RAG systems first search a knowledge base, then generate answers using retrieved context."
    ],
    "ground_truth": "RAG is Retrieval Augmented Generation, combining retrieval and generation"
  }'
```

### Step 6: View Evaluation History

```bash
curl http://localhost:8000/evaluate/history
```

**Response:**
```json
{
  "message": "Found 5 evaluation runs",
  "runs": [
    {
      "run_id": "abc123",
      "run_name": "baseline_evaluation",
      "start_time": 1234567890,
      "faithfulness": 0.85,
      "answer_relevancy": 0.91,
      "context_precision": 0.78
    }
  ]
}
```

## 📈 Viewing Results in MLflow

### Start MLflow UI

```bash
# Navigate to project directory
cd /path/to/your/project

# Start MLflow UI
mlflow ui --port 5000

# Open browser: http://localhost:5000
```

### MLflow UI Features

**1. Experiments Tab:**
- View all evaluation runs
- Filter by parameters
- Sort by metrics

**2. Runs Comparison:**
- Select multiple runs
- Compare metrics side-by-side
- View parameter differences

**3. Run Details:**
- Individual metric scores
- Parameters used
- Artifacts (CSV files)
- Tags and notes

**4. Charts:**
- Metric trends over time
- Parameter impact analysis
- Custom visualizations

## 🎨 Example Workflows

### Workflow 1: Baseline Evaluation

```bash
# 1. Upload document
curl -X POST http://localhost:8000/upload -F "file=@doc.pdf"

# 2. Create test cases
cat > test_cases.json << EOF
{
  "test_cases": [
    {"question": "What is X?", "ground_truth": "X is..."},
    {"question": "How does Y work?", "ground_truth": "Y works by..."}
  ],
  "run_name": "baseline",
  "use_hyde": false
}
EOF

# 3. Run evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @test_cases.json

# 4. View in MLflow
mlflow ui --port 5000
```

### Workflow 2: Compare Standard vs HyDE

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Test cases
test_cases = {
    "test_cases": [
        {"question": "What is X?", "ground_truth": "X is..."},
        {"question": "How does Y work?", "ground_truth": "Y works by..."}
    ]
}

# Evaluate with standard retrieval
standard = test_cases.copy()
standard["run_name"] = "standard_retrieval"
standard["use_hyde"] = False

response = requests.post(f"{BASE_URL}/evaluate", json=standard)
standard_scores = response.json()["scores"]

# Evaluate with HyDE
hyde = test_cases.copy()
hyde["run_name"] = "hyde_retrieval"
hyde["use_hyde"] = True

response = requests.post(f"{BASE_URL}/evaluate", json=hyde)
hyde_scores = response.json()["scores"]

# Compare
print("Metric Comparison:")
for metric in standard_scores:
    std = standard_scores[metric]
    hyd = hyde_scores[metric]
    improvement = ((hyd - std) / std * 100) if std > 0 else 0
    print(f"{metric:.<30} {std:.4f} → {hyd:.4f} ({improvement:+.2f}%)")
```

### Workflow 3: Automated Comparison Script

```bash
# Use the provided comparison script
python compare_hyde_vs_standard.py
```

**Script Output:**
```
================================================================================
STANDARD vs HyDE RETRIEVAL COMPARISON
================================================================================

📊 Running evaluation with STANDARD retrieval...
✓ Standard Retrieval Complete!
  Scores:
    faithfulness..................... 0.7821
    answer_relevancy................. 0.8234
    context_precision................ 0.7456

📊 Running evaluation with HyDE retrieval...
✓ HyDE Retrieval Complete!
  Scores:
    faithfulness..................... 0.8456
    answer_relevancy................. 0.8912
    context_precision................ 0.8234

================================================================================
COMPARISON RESULTS
================================================================================

Metric                          Standard         HyDE     Improvement
------------------------------------------------------------------------
faithfulness                      0.7821       0.8456     🟢    +8.12%
answer_relevancy                  0.8234       0.8912     🟢    +8.23%
context_precision                 0.7456       0.8234     🟢   +10.44%
------------------------------------------------------------------------
AVERAGE IMPROVEMENT                                       🟢    +8.93%

✅ HyDE shows significant improvement! Recommended for production.
```

## 📊 Interpreting Scores

### Score Ranges

| Score | Interpretation | Action |
|-------|---------------|--------|
| 0.9 - 1.0 | Excellent | Maintain current approach |
| 0.8 - 0.9 | Good | Minor optimizations possible |
| 0.7 - 0.8 | Acceptable | Consider improvements |
| 0.6 - 0.7 | Needs Work | Investigate issues |
| < 0.6 | Poor | Significant changes needed |

### Common Issues and Solutions

#### Low Faithfulness (< 0.6)

**Problem:** Answers contradict or aren't supported by context.

**Solutions:**
- Improve prompt to emphasize grounding in context
- Adjust retrieval to get better context
- Increase context window
- Add explicit instruction: "Only use information from the context"

#### Low Answer Relevancy (< 0.6)

**Problem:** Answers don't address the question.

**Solutions:**
- Improve question reformulation
- Tune retrieval parameters
- Adjust LLM temperature
- Refine prompt instructions

#### Low Context Precision (< 0.5)

**Problem:** Retrieving too many irrelevant chunks.

**Solutions:**
- Increase chunk overlap for better context
- Tune embedding model
- Adjust top-K parameter (reduce noise)
- Implement re-ranking
- Try HyDE enhancement

#### Low Context Recall (< 0.5)

**Problem:** Missing relevant information.

**Solutions:**
- Retrieve more chunks (increase K)
- Reduce chunk size for finer granularity
- Improve chunking strategy
- Try HyDE enhancement
- Use hybrid search (dense + sparse)

## 🎯 Best Practices

### 1. Create Diverse Test Cases

```json
{
  "test_cases": [
    // Factual questions
    {"question": "Who created Python?", "ground_truth": "Guido van Rossum"},
    
    // Analytical questions
    {"question": "Why is Python popular?", "ground_truth": "Easy syntax, large ecosystem"},
    
    // Complex questions
    {"question": "How does Python's GIL affect performance?", "ground_truth": "..."},
    
    // Multi-hop questions
    {"question": "Compare Python and Java for web development", "ground_truth": "..."}
  ]
}
```

### 2. Include Ground Truths When Possible

Ground truths enable 6 metrics instead of 3:
- ✅ All metrics: faithfulness, relevancy, precision, recall, similarity, correctness
- ❌ Without ground truth: only faithfulness, relevancy, precision

### 3. Run Systematic Comparisons

```python
# Change one variable at a time
test_runs = [
    {"run_name": "baseline", "chunk_size": 512},
    {"run_name": "larger_chunks", "chunk_size": 1024},
    {"run_name": "smaller_chunks", "chunk_size": 256}
]

for config in test_runs:
    # Update configuration
    # Run evaluation
    # Compare in MLflow
```

### 4. Use Descriptive Run Names

```python
# Good naming
"baseline_standard_chunk512"
"experiment_hyde_chunk1024_k10"
"prod_v1_standard"

# Poor naming
"test1"
"run_abc"
"evaluation"
```

### 5. Track Everything

MLflow automatically logs:
- ✅ Metric scores
- ✅ Parameters
- ✅ Timestamps
- ✅ Detailed results (CSV)

## 🔬 Advanced Evaluation

### Custom Test Dataset Creation

```python
# evaluation.py includes helper function
from evaluation import create_test_dataset_from_pdf

test_cases = create_test_dataset_from_pdf(
    pdf_path="document.pdf",
    num_questions=10
)

# Then manually add ground truths
for case in test_cases:
    case["ground_truth"] = input(f"Ground truth for '{case['question']}'? ")
```

### Batch Evaluation

```python
from evaluation import RAGEvaluator

evaluator = RAGEvaluator()

# Prepare batch
batch_test_cases = [
    {
        "question": "Q1",
        "answer": "A1",  # Pre-generated
        "contexts": ["C1", "C2"],
        "ground_truth": "GT1"
    },
    # ... more cases
]

results = evaluator.batch_evaluate(
    test_cases=batch_test_cases,
    run_name="batch_evaluation"
)
```

### Comparing Multiple Runs

```bash
# Get run IDs from MLflow UI or API
RUN1="abc123"
RUN2="def456"
RUN3="ghi789"

curl "http://localhost:8000/evaluate/compare?run_ids=$RUN1,$RUN2,$RUN3"
```

## 🐛 Troubleshooting

### Error: "No module named 'ragas'"

```bash
pip install ragas==0.2.7 datasets==3.1.0
```

### Error: "Evaluator not initialized"

```bash
# Check server logs
docker-compose logs app

# Verify dependencies
curl http://localhost:8000/debug/status
```

### Slow Evaluation

**Solutions:**
- Reduce number of test cases for quick iterations
- Use `/evaluate/single` for rapid testing
- Consider smaller LLM model
- Process in batches

### Memory Issues

**Solutions:**
- Reduce test case batch size
- Reduce context length
- Use quantized models
- Increase Docker memory allocation

### Inconsistent Scores

**Causes:**
- LLM non-determinism (set temperature=0)
- Different retrieved contexts
- Evaluation LLM variance

**Solutions:**
- Run multiple evaluations and average
- Use deterministic settings
- Monitor detailed results

## 📚 Additional Resources

- **RAGAS Documentation:** https://docs.ragas.io/
- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
- **Evaluation Best Practices:** https://docs.ragas.io/en/stable/concepts/metrics/

---

**Last Updated:** December 2024  
**Evaluation System:** Production Ready ✅
