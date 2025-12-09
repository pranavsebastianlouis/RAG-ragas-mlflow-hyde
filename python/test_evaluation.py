"""
Test script for RAG evaluation
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_upload_and_evaluate():
    """Test the complete workflow: upload PDF and run evaluation."""
    
    print("=" * 60)
    print("Testing Document QA Evaluation Pipeline")
    print("=" * 60)
    
    # 1. Check health
    print("\n1. Checking API health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.json()['status']}")
    print(f"   MLflow URI: {response.json()['mlflow_tracking_uri']}")
    
    # 2. Upload PDF (replace with your actual PDF path)
    print("\n2. Uploading PDF...")
    # Uncomment and modify this when you have a PDF
    # with open("your_document.pdf", "rb") as f:
    #     files = {"file": f}
    #     response = requests.post(f"{BASE_URL}/upload", files=files)
    #     print(f"   {response.json()['message']}")
    #     print(f"   Pages: {response.json()['pages']}, Chunks: {response.json()['chunks']}")
    
    # 3. Define test cases
    print("\n3. Preparing test cases...")
    test_cases = [
        {
            "question": "What is the main topic of the document?",
            "ground_truth": "Expected answer about the main topic"  # Optional
        },
        {
            "question": "Who are the key people mentioned?",
            # ground_truth is optional
        },
        {
            "question": "What are the key findings or conclusions?",
            "ground_truth": "Expected findings"
        }
    ]
    print(f"   Prepared {len(test_cases)} test cases")
    
    # 4. Run evaluation
    print("\n4. Running evaluation...")
    eval_request = {
        "test_cases": test_cases,
        "run_name": "test_evaluation_run"
    }
    
    response = requests.post(
        f"{BASE_URL}/evaluate",
        json=eval_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n   ✓ Evaluation Complete!")
        print(f"   MLflow Run ID: {result['mlflow_run_id']}")
        print(f"   Test Cases: {result['num_test_cases']}")
        print("\n   Scores:")
        for metric, score in result['scores'].items():
            print(f"      {metric}: {score:.4f}")
    else:
        print(f"   ✗ Evaluation failed: {response.json()}")
    
    # 5. Get evaluation history
    print("\n5. Checking evaluation history...")
    response = requests.get(f"{BASE_URL}/evaluate/history")
    if response.status_code == 200:
        history = response.json()
        print(f"   {history['message']}")
        if history['runs']:
            print(f"   Latest run: {history['runs'][0].get('tags.mlflow.runName', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


def test_single_evaluation():
    """Test single query evaluation."""
    
    print("\n" + "=" * 60)
    print("Testing Single Query Evaluation")
    print("=" * 60)
    
    # Example single evaluation
    eval_request = {
        "question": "What is RAG?",
        "answer": "RAG stands for Retrieval Augmented Generation, a technique that combines information retrieval with text generation.",
        "contexts": [
            "Retrieval Augmented Generation (RAG) is a method that enhances LLM responses by retrieving relevant information.",
            "RAG systems first search a knowledge base, then use the retrieved context to generate informed answers."
        ],
        "ground_truth": "RAG is Retrieval Augmented Generation"  # Optional
    }
    
    response = requests.post(
        f"{BASE_URL}/evaluate/single",
        json=eval_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n   ✓ Single Query Evaluated!")
        print(f"   Question: {result['question']}")
        print("\n   Scores:")
        for metric, score in result['scores'].items():
            print(f"      {metric}: {score:.4f}")
    else:
        print(f"   ✗ Evaluation failed: {response.json()}")


if __name__ == "__main__":
    # Test single evaluation (doesn't require PDF upload)
    test_single_evaluation()
    
    # Test full pipeline (requires PDF upload)
    # Uncomment when you have a PDF uploaded
    # test_upload_and_evaluate()