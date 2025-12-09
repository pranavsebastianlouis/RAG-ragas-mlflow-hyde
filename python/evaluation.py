"""
Evaluation module for RAG pipeline using RAGAS and MLflow
"""

import os
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
import mlflow
import mlflow.langchain
from datetime import datetime
import pandas as pd


class RAGEvaluator:
    """Evaluates RAG pipeline performance using RAGAS metrics."""
    
    def __init__(
        self, 
        experiment_name: str = "document_qa_evaluation",
        ollama_model: str = "mistral",
        ollama_base_url: str = None
    ):
        """
        Initialize the evaluator.
        
        Args:
            experiment_name: Name for the MLflow experiment
            ollama_model: Ollama model to use for evaluation
            ollama_base_url: Base URL for Ollama server
        """
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
        # Get Ollama base URL from environment or use default
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Initialize Ollama LLM for RAGAS
        print(f"  Configuring RAGAS to use Ollama ({ollama_model})...")
        ollama_llm = ChatOllama(
            model=ollama_model,
            base_url=self.ollama_base_url,
            temperature=0
        )
        
        # Wrap for RAGAS
        self.ragas_llm = LangchainLLMWrapper(ollama_llm)
        
        # Initialize embeddings for RAGAS
        print("  Loading embeddings for RAGAS...")
        hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
        
        # Define RAGAS metrics with our custom LLM and embeddings
        self.metrics = [
            faithfulness,           # Factual consistency of answer with context
            answer_relevancy,       # How relevant is the answer to the question
            context_precision,      # Precision of retrieved context
            context_recall,         # Recall of retrieved context
            answer_similarity,      # Semantic similarity to ground truth
            answer_correctness      # Correctness compared to ground truth
        ]
        
        print("  ✓ RAGAS configured with Ollama")
        
    def prepare_evaluation_dataset(
        self, 
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None
    ) -> Dataset:
        """
        Prepare data in RAGAS format.
        
        Args:
            questions: List of questions asked
            answers: List of answers generated
            contexts: List of retrieved contexts (each is a list of strings)
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dataset in RAGAS format
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
            
        return Dataset.from_dict(data)
    
    def evaluate_pipeline(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate the RAG pipeline using RAGAS metrics.
        
        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved contexts
            ground_truths: Optional ground truth answers
            run_name: Name for this evaluation run
            
        Returns:
            Dictionary containing evaluation scores
        """
        # Prepare dataset
        dataset = self.prepare_evaluation_dataset(
            questions, answers, contexts, ground_truths
        )
        
        # Select metrics based on whether ground truth is provided
        metrics_to_use = self.metrics.copy()
        if not ground_truths:
            # Remove metrics that require ground truth
            metrics_to_use = [
                faithfulness,
                answer_relevancy,
                context_precision
            ]
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("num_samples", len(questions))
            mlflow.log_param("has_ground_truth", ground_truths is not None)
            mlflow.log_param("timestamp", datetime.now().isoformat())
            mlflow.log_param("llm_model", "ollama/mistral")
            mlflow.log_param("ollama_base_url", self.ollama_base_url)
            
            # Run RAGAS evaluation with custom LLM and embeddings
            print("  Running RAGAS evaluation with Ollama...")
            results = evaluate(
                dataset, 
                metrics=metrics_to_use,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings
            )
            
            # Extract scores - RAGAS returns EvaluationResult object
            # Convert to pandas DataFrame first
            results_df = results.to_pandas()
            
            # Extract metric scores from the DataFrame
            scores = {}
            for col in results_df.columns:
                # Skip non-metric columns
                if col not in ['question', 'contexts', 'answer', 'ground_truth', 'user_input', 'retrieved_contexts', 'response']:
                    try:
                        # Get the mean score for this metric across all samples
                        score_value = results_df[col].mean()
                        if pd.notna(score_value):  # Only add if not NaN
                            scores[col] = float(score_value)
                    except (TypeError, ValueError, KeyError) as e:
                        print(f"  Warning: Could not extract metric '{col}': {e}")
                        pass
            
            # Log metrics to MLflow
            for metric_name, score in scores.items():
                mlflow.log_metric(metric_name, score)
            
            # Log the full results as artifact
            results_csv = "evaluation_results.csv"
            results_df.to_csv(results_csv, index=False)
            mlflow.log_artifact(results_csv)
            
            # Clean up temp file
            if os.path.exists(results_csv):
                os.remove(results_csv)
            
            print(f"  ✓ Evaluation complete. Results logged to MLflow.")
            print(f"    Experiment: {self.experiment_name}")
            print(f"    Run ID: {mlflow.active_run().info.run_id}")
            print(f"    Scores: {scores}")
            
            return {
                "scores": scores,
                "detailed_results": results_df.to_dict('records'),
                "mlflow_run_id": mlflow.active_run().info.run_id
            }
    
    def evaluate_single_query(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = None
    ) -> Dict[str, float]:
        """
        Evaluate a single query-answer pair.
        
        Args:
            question: The question asked
            answer: The generated answer
            contexts: Retrieved context chunks
            ground_truth: Optional ground truth answer
            
        Returns:
            Dictionary of metric scores
        """
        result = self.evaluate_pipeline(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
            run_name=f"single_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return result["scores"]
    
    def batch_evaluate(
        self,
        test_cases: List[Dict[str, Any]],
        run_name: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple test cases at once.
        
        Args:
            test_cases: List of dicts with keys: question, answer, contexts, ground_truth (optional)
            run_name: Name for this evaluation run
            
        Returns:
            Evaluation results
        """
        questions = [tc["question"] for tc in test_cases]
        answers = [tc["answer"] for tc in test_cases]
        contexts = [tc["contexts"] for tc in test_cases]
        ground_truths = [tc.get("ground_truth") for tc in test_cases]
        
        # Check if any ground truths exist
        has_ground_truth = any(gt is not None for gt in ground_truths)
        
        return self.evaluate_pipeline(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths if has_ground_truth else None,
            run_name=run_name
        )
    
    @staticmethod
    def get_experiment_history(experiment_name: str = "document_qa_evaluation") -> pd.DataFrame:
        """
        Get history of all evaluation runs.
        
        Args:
            experiment_name: Name of the MLflow experiment
            
        Returns:
            DataFrame with run history
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return runs
    
    @staticmethod
    def compare_runs(run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple evaluation runs.
        
        Args:
            run_ids: List of MLflow run IDs to compare
            
        Returns:
            DataFrame comparing the runs
        """
        runs_data = []
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            runs_data.append({
                "run_id": run_id,
                "start_time": run.info.start_time,
                **run.data.metrics,
                **run.data.params
            })
        
        return pd.DataFrame(runs_data)


def create_test_dataset_from_pdf(
    pdf_path: str,
    num_questions: int = 5
) -> List[Dict[str, Any]]:
    """
    Helper function to create test cases from a PDF.
    You would manually create questions and ground truths.
    
    Args:
        pdf_path: Path to PDF file
        num_questions: Number of test questions to create
        
    Returns:
        List of test cases
    """
    # This is a template - you need to fill in with actual test cases
    test_cases = [
        {
            "question": "What is the main topic of the document?",
            "ground_truth": "Your expected answer here",
            # answer and contexts will be filled by the chatbot
        },
        # Add more test cases...
    ]
    
    return test_cases[:num_questions]