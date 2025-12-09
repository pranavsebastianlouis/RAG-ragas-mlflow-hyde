"""
Comparison script for Standard vs HyDE retrieval
Runs evaluations with both methods and compares results.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"


def run_comparison(test_cases):
    """
    Run evaluation with both standard and HyDE retrieval.
    
    Args:
        test_cases: List of test cases with questions and ground truths
    """
    print("=" * 80)
    print("STANDARD vs HyDE RETRIEVAL COMPARISON")
    print("=" * 80)
    
    results = {}
    
    # 1. Evaluate with Standard Retrieval
    print("\n📊 Running evaluation with STANDARD retrieval...")
    print("-" * 80)
    
    standard_request = {
        "test_cases": test_cases,
        "run_name": "standard_retrieval",
        "use_hyde": False
    }
    
    response = requests.post(
        f"{BASE_URL}/evaluate",
        json=standard_request
    )
    
    if response.status_code == 200:
        standard_result = response.json()
        results['standard'] = standard_result
        print("\n✓ Standard Retrieval Complete!")
        print(f"  MLflow Run ID: {standard_result['mlflow_run_id']}")
        print("\n  Scores:")
        for metric, score in standard_result['scores'].items():
            print(f"    {metric:.<30} {score:.4f}")
    else:
        print(f"\n✗ Standard evaluation failed: {response.json()}")
        return None
    
    print("\n" + "=" * 80)
    time.sleep(2)  # Brief pause between evaluations
    
    # 2. Evaluate with HyDE Retrieval
    print("\n📊 Running evaluation with HyDE retrieval...")
    print("-" * 80)
    
    hyde_request = {
        "test_cases": test_cases,
        "run_name": "hyde_retrieval",
        "use_hyde": True
    }
    
    response = requests.post(
        f"{BASE_URL}/evaluate",
        json=hyde_request
    )
    
    if response.status_code == 200:
        hyde_result = response.json()
        results['hyde'] = hyde_result
        print("\n✓ HyDE Retrieval Complete!")
        print(f"  MLflow Run ID: {hyde_result['mlflow_run_id']}")
        print("\n  Scores:")
        for metric, score in hyde_result['scores'].items():
            print(f"    {metric:.<30} {score:.4f}")
    else:
        print(f"\n✗ HyDE evaluation failed: {response.json()}")
        return results
    
    # 3. Compare Results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    if 'standard' in results and 'hyde' in results:
        print(f"\n{'Metric':<30} {'Standard':>12} {'HyDE':>12} {'Improvement':>15}")
        print("-" * 72)
        
        for metric in results['standard']['scores'].keys():
            standard_score = results['standard']['scores'][metric]
            hyde_score = results['hyde']['scores'][metric]
            improvement = ((hyde_score - standard_score) / standard_score * 100) if standard_score > 0 else 0
            
            improvement_str = f"{improvement:+.2f}%"
            symbol = "🟢" if improvement > 0 else "🔴" if improvement < 0 else "⚪"
            
            print(f"{metric:<30} {standard_score:>12.4f} {hyde_score:>12.4f} {symbol} {improvement_str:>12}")
        
        # Calculate average improvement
        avg_improvement = sum([
            ((results['hyde']['scores'][m] - results['standard']['scores'][m]) / results['standard']['scores'][m] * 100)
            for m in results['standard']['scores'].keys()
            if results['standard']['scores'][m] > 0
        ]) / len(results['standard']['scores'])
        
        print("-" * 72)
        print(f"{'AVERAGE IMPROVEMENT':<30} {'':<12} {'':<12} {avg_improvement:>15.2f}%")
        
        # Recommendation
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        if avg_improvement > 5:
            print("✅ HyDE shows significant improvement! Recommended for production.")
        elif avg_improvement > 0:
            print("⚠️  HyDE shows marginal improvement. Consider use case requirements.")
        else:
            print("❌ Standard retrieval performs better. Stick with standard method.")
        
        print(f"\n📊 View detailed results in MLflow:")
        print(f"   mlflow ui --port 5000")
        print(f"   Then compare runs:")
        print(f"   - Standard: {results['standard']['mlflow_run_id']}")
        print(f"   - HyDE:     {results['hyde']['mlflow_run_id']}")
    
    print("\n" + "=" * 80)
    
    return results


def main():
    """Main function to run the comparison."""
    
    # Example test cases - customize these for your document
    test_cases = [
        {
            "question": "What is the main topic of the document?",
            "ground_truth": "Your expected answer here"  # Optional but recommended
        },
        {
            "question": "What are the key findings?",
            "ground_truth": "Expected findings"
        },
        {
            "question": "Who are the authors?",
            # ground_truth is optional
        },
        {
            "question": "What methodology was used?",
            "ground_truth": "Expected methodology"
        },
        {
            "question": "What are the conclusions?",
            "ground_truth": "Expected conclusions"
        }
    ]
    
    print("\n🔍 Starting comparison with test cases:")
    for i, tc in enumerate(test_cases, 1):
        print(f"  {i}. {tc['question']}")
    
    input("\nPress Enter to start evaluation...")
    
    results = run_comparison(test_cases)
    
    if results:
        print("\n✅ Comparison complete!")
        
        # Save results to file
        with open("hyde_comparison_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n💾 Results saved to 'hyde_comparison_results.json'")
    else:
        print("\n❌ Comparison failed. Check server logs.")


if __name__ == "__main__":
    main()