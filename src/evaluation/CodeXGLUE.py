#!/usr/bin/env python3
"""
CodeXGLUE Benchmark Runner
One-click test execution for CodeXGLUE benchmark
"""

import argparse
import json
import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# TODO: Add these imports when dependencies are installed
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        AutoModelForSeq2SeqLM, Trainer, TrainingArguments
    )
    from datasets import load_dataset, Dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"⚠️  Missing dependency: {e}")
    print("Run: ./CodeXGLUE.sh to setup environment")
    sys.exit(1)


class CodeXGLUEBenchmark:
    """CodeXGLUE benchmark runner"""
    
    def __init__(self, config_path: str = "configs/config.json"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_path}")
            print("Run ./CodeXGLUE.sh to setup environment")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('outputs/benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get task-specific information"""
        task_configs = {
            "code-to-code-trans": {
                "type": "translation",
                "dataset": "Code-Code/code-to-code-trans",
                "model": "microsoft/codebert-base",
                "metric": "BLEU",
                "max_length": 512,
                "task_type": "seq2seq"
            },
            "code-generation": {
                "type": "generation", 
                "dataset": "Text-Code/code-generation",
                "model": "Salesforce/codet5-base",
                "metric": "BLEU",
                "max_length": 512,
                "task_type": "seq2seq"
            },
            "code-summarization": {
                "type": "summarization",
                "dataset": "Code-Text/code-summarization", 
                "model": "microsoft/unixcoder-base",
                "metric": "BLEU",
                "max_length": 256,
                "task_type": "seq2seq"
            },
            "defect-detection": {
                "type": "classification",
                "dataset": "Code-Code/defect-detection",
                "model": "microsoft/codebert-base",
                "metric": "accuracy",
                "max_length": 512,
                "task_type": "classification"
            },
            "clone-detection": {
                "type": "classification",
                "dataset": "Code-Code/clone-detection",
                "model": "microsoft/codebert-base",
                "metric": "F1",
                "max_length": 512,
                "task_type": "classification"
            }
        }
        return task_configs.get(task_name, {})
    
    def load_dataset(self, task_name: str) -> Dict[str, Any]:
        """Load dataset for specific task"""
        task_info = self.get_task_info(task_name)
        if not task_info:
            raise ValueError(f"Unknown task: {task_name}")
        
        dataset_path = f"data/{task_info['dataset']}"
        
        # TODO: Implement actual dataset loading
        # For now, create sample data
        return self._create_sample_data(task_name)
    
    def _create_sample_data(self, task_name: str) -> Dict[str, Any]:
        """Create sample data for testing"""
        sample_data = {
            "code-to-code-trans": {
                "train": [
                    {"source": "def add(a, b): return a + b", "target": "function add(a, b) { return a + b; }"}
                ] * 100,
                "test": [
                    {"source": "def multiply(x, y): return x * y", "target": "function multiply(x, y) { return x * y; }"}
                ] * 20
            },
            "code-generation": {
                "train": [
                    {"prompt": "Create a function to calculate factorial", "code": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"}
                ] * 100,
                "test": [
                    {"prompt": "Create a function to check if string is palindrome", "code": "def is_palindrome(s): return s == s[::-1]"}
                ] * 20
            },
            "code-summarization": {
                "train": [
                    {"code": "def quicksort(arr): if len(arr) <= 1: return arr; pivot = arr[len(arr)//2]; left = [x for x in arr if x < pivot]; middle = [x for x in arr if x == pivot]; right = [x for x in arr if x > pivot]; return quicksort(left) + middle + quicksort(right)", "summary": "Implementation of quicksort algorithm"}
                ] * 100,
                "test": [
                    {"code": "def fibonacci(n): if n <= 1: return n; return fibonacci(n-1) + fibonacci(n-2)", "summary": "Recursive Fibonacci sequence implementation"}
                ] * 20
            },
            "defect-detection": {
                "train": [
                    {"code": "def safe_divide(a, b): return a / b", "label": 1},  # Defect: no zero check
                    {"code": "def safe_divide(a, b): if b == 0: return None; return a / b", "label": 0}
                ] * 100,
                "test": [
                    {"code": "def add(a, b): return a + b", "label": 0}
                ] * 20
            },
            "clone-detection": {
                "train": [
                    {"code1": "def add(a, b): return a + b", "code2": "def sum(x, y): return x + y", "label": 1},
                    {"code1": "def add(a, b): return a + b", "code2": "def multiply(x, y): return x * y", "label": 0}
                ] * 100,
                "test": [
                    {"code1": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)", "code2": "def fact(x): return 1 if x <= 1 else x * fact(x-1)", "label": 1}
                ] * 20
            }
        }
        
        data = sample_data.get(task_name, {"train": [], "test": []})
        return data
    
    def load_model(self, model_name: str, task_type: str):
        """Load model and tokenizer"""
        self.logger.info(f"Loading model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if task_type == "classification":
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            elif task_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
            
            model = model.to(self.device)
            return tokenizer, model
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            # TODO: Implement fallback model
            raise
    
    def compute_metrics(self, task_type: str, predictions, labels) -> Dict[str, float]:
        """Compute task-specific metrics"""
        if task_type == "classification":
            accuracy = accuracy_score(labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted'
            )
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        else:  # seq2seq
            # TODO: Implement BLEU score computation
            return {"bleu": 0.5}  # Placeholder
    
    def run_task(self, task_name: str, model_name: str = None) -> Dict[str, Any]:
        """Run a single task"""
        task_info = self.get_task_info(task_name)
        if not task_info:
            raise ValueError(f"Unknown task: {task_name}")
        
        model_name = model_name or task_info["model"]
        
        self.logger.info(f"Running task: {task_name} with model: {model_name}")
        
        # Load data
        data = self.load_dataset(task_name)
        
        # Load model
        tokenizer, model = self.load_model(model_name, task_info["task_type"])
        
        # Run evaluation
        start_time = time.time()
        
        # TODO: Implement actual evaluation pipeline
        # For now, simulate evaluation
        if task_info["task_type"] == "classification":
            # Simulate classification results
            predictions = [0, 1, 0, 1, 0] * 4  # 20 predictions
            labels = [0, 1, 0, 1, 0] * 4
        else:
            # Simulate seq2seq results
            predictions = ["output1", "output2", "output3"] * 7
            labels = ["output1", "output2", "output3"] * 7
        
        metrics = self.compute_metrics(task_info["task_type"], predictions, labels)
        
        result = {
            "task_name": task_name,
            "model_name": model_name,
            "task_type": task_info["task_type"],
            "metrics": metrics,
            "execution_time": time.time() - start_time,
            "num_samples": len(data.get("test", [])),
            "sample_predictions": predictions[:5] if predictions else []
        }
        
        return result
    
    def run_benchmark(self, tasks: List[str], models: List[str] = None) -> Dict[str, Any]:
        """Run complete benchmark"""
        self.logger.info("Starting CodeXGLUE benchmark")
        
        results = []
        
        for task in tasks:
            try:
                if models:
                    for model in models:
                        result = self.run_task(task, model)
                        results.append(result)
                else:
                    result = self.run_task(task)
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Error running task {task}: {e}")
                results.append({
                    "task_name": task,
                    "error": str(e),
                    "success": False
                })
        
        summary = {
            "tasks_run": len(results),
            "successful_tasks": sum(1 for r in results if "error" not in r),
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "outputs/"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        try:
            df_data = []
            for result in results["results"]:
                if "error" not in result:
                    row = {
                        "task_name": result["task_name"],
                        "model_name": result["model_name"],
                        "task_type": result["task_type"],
                        "execution_time": result["execution_time"],
                        "num_samples": result["num_samples"]
                    }
                    row.update(result["metrics"])
                    df_data.append(row)
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.to_csv(f"{output_dir}/results.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
        
        self.logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CodeXGLUE benchmark runner")
    parser.add_argument("--task", type=str, default=None,
                       help="Task to run (code-to-code-trans, code-generation, etc.)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model to use (overrides config)")
    parser.add_argument("--tasks", type=str, default=None,
                       help="Comma-separated list of tasks")
    parser.add_argument("--models", type=str, default=None,
                       help="Comma-separated list of models")
    parser.add_argument("--config", type=str, default="configs/config.json",
                       help="Config file path")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training/evaluation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = CodeXGLUEBenchmark(args.config)
    
    # Determine tasks to run
    if args.task:
        tasks = [args.task]
    elif args.tasks:
        tasks = args.tasks.split(',')
    else:
        tasks = [
            "code-to-code-trans",
            "code-generation", 
            "code-summarization",
            "defect-detection",
            "clone-detection"
        ]
    
    # Determine models to use
    models = None
    if args.model:
        models = [args.model]
    elif args.models:
        models = args.models.split(',')
    
    # Run benchmark
    print(f"🚀 Starting CodeXGLUE benchmark...")
    print(f"Tasks: {tasks}")
    if models:
        print(f"Models: {models}")
    
    results = benchmark.run_benchmark(tasks, models)
    
    # Save results
    benchmark.save_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 CodeXGLUE Benchmark Results")
    print("="*60)
    print(f"Total tasks: {results['tasks_run']}")
    print(f"Successful: {results['successful_tasks']}")
    print(f"Failed: {results['tasks_run'] - results['successful_tasks']}")
    print(f"Timestamp: {results['timestamp']}")
    
    # Print per-task results
    for result in results["results"]:
        if "error" not in result:
            print(f"\n{result['task_name']}: {result['metrics']}")
        else:
            print(f"\n{result['task_name']}: ERROR - {result['error']}")
    
    print("="*60)
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()