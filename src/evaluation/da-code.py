#!/usr/bin/env python3
"""
DA-Code Benchmark Runner
One-click test execution for DA-Code benchmark
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
import tempfile
import shutil

# TODO: Add these imports when dependencies are installed
try:
    import openai
    import anthropic
    import docker
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"⚠️  Missing dependency: {e}")
    print("Run: ./da-code.sh to setup environment")
    sys.exit(1)


class DACodeBenchmark:
    """DA-Code benchmark runner"""
    
    def __init__(self, config_path: str = "config/config.json"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Config file not found: {config_path}")
            print("Run ./da-code.sh to setup environment")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('output/benchmark.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_environment(self) -> bool:
        """Validate environment setup"""
        self.logger.info("Validating environment...")
        
        # Check API keys
        api_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        for key in api_keys:
            if os.getenv(key):
                self.logger.info(f"✅ Found {key}")
            else:
                self.logger.warning(f"⚠️  {key} not found")
        
        # Check Docker
        try:
            subprocess.run(['docker', '--version'], 
                         check=True, capture_output=True)
            self.logger.info("✅ Docker available")
        except subprocess.CalledProcessError:
            self.logger.error("❌ Docker not available")
            return False
        
        return True
    
    def load_tasks(self, dataset_path: str = "da_code/") -> List[Dict[str, Any]]:
        """Load benchmark tasks"""
        tasks_file = Path(dataset_path) / "tasks.json"
        
        if not tasks_file.exists():
            # TODO: Download actual dataset
            self.logger.warning("Using sample tasks...")
            return self._create_sample_tasks()
        
        with open(tasks_file, 'r') as f:
            data = json.load(f)
            return data.get('tasks', [])
    
    def _create_sample_tasks(self) -> List[Dict[str, Any]]:
        """Create sample tasks for testing"""
        return [
            {
                "task_id": "data_analysis_001",
                "type": "data_analysis",
                "description": "Load and analyze a CSV file",
                "prompt": "Load data.csv and create a summary report",
                "test_cases": [
                    {"type": "file_exists", "path": "output/report.txt"},
                    {"type": "code_executes", "code": "import pandas as pd; print('pandas imported')"}
                ],
                "timeout": 60
            },
            {
                "task_id": "visualization_001", 
                "type": "visualization",
                "description": "Create a matplotlib plot",
                "prompt": "Create a simple line plot of [1,2,3,4,5] vs [1,4,9,16,25]",
                "test_cases": [
                    {"type": "file_exists", "path": "output/plot.png"},
                    {"type": "code_executes", "code": "import matplotlib.pyplot as plt; plt.plot([1,2,3,4,5], [1,4,9,16,25]); plt.savefig('plot.png')"}
                ],
                "timeout": 60
            }
        ]
    
    def run_task(self, task: Dict[str, Any], model: str, max_steps: int) -> Dict[str, Any]:
        """Run a single task"""
        self.logger.info(f"Running task {task['task_id']}")
        
        result = {
            "task_id": task["task_id"],
            "model": model,
            "max_steps": max_steps,
            "success": False,
            "steps": [],
            "error": None,
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # TODO: Implement actual model interaction
            # For now, simulate task execution
            for step in range(max_steps):
                step_result = {
                    "step": step + 1,
                    "code": f"# Step {step + 1}\nprint('Processing task {task[\"task_id\"]}')",
                    "output": f"Step {step + 1} completed",
                    "success": True
                }
                result["steps"].append(step_result)
                
                # Simulate task completion
                if step >= 2:
                    result["success"] = True
                    break
                    
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Error in task {task['task_id']}: {e}")
        
        result["execution_time"] = time.time() - start_time
        return result
    
    def run_benchmark(self, model: str, max_steps: int = 20, 
                     num_workers: int = 4, subset: List[str] = None) -> Dict[str, Any]:
        """Run the complete benchmark"""
        
        if not self.validate_environment():
            return {"error": "Environment validation failed"}
        
        # Load tasks
        tasks = self.load_tasks()
        if subset:
            tasks = [t for t in tasks if t["task_id"] in subset]
        
        self.logger.info(f"Running {len(tasks)} tasks with model {model}")
        
        # Run tasks
        results = []
        for task in tqdm(tasks, desc="Running tasks"):
            result = self.run_task(task, model, max_steps)
            results.append(result)
        
        # Generate summary
        summary = {
            "total_tasks": len(tasks),
            "successful_tasks": sum(1 for r in results if r["success"]),
            "failed_tasks": sum(1 for r in results if not r["success"]),
            "average_time": sum(r["execution_time"] for r in results) / len(results),
            "results": results
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "output/"):
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(f"{output_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary = {
            "benchmark_info": {
                "name": "DA-Code",
                "version": "1.0.0",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "summary": {
                "total_tasks": results["total_tasks"],
                "successful_tasks": results["successful_tasks"],
                "failed_tasks": results["failed_tasks"],
                "success_rate": results["successful_tasks"] / results["total_tasks"],
                "average_time": results["average_time"]
            }
        }
        
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV for easy analysis
        try:
            import pandas as pd
            df = pd.DataFrame([
                {
                    "task_id": r["task_id"],
                    "success": r["success"],
                    "time": r["execution_time"],
                    "error": r.get("error", "")
                }
                for r in results["results"]
            ])
            df.to_csv(f"{output_dir}/results.csv", index=False)
        except ImportError:
            pass
        
        self.logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DA-Code benchmark runner")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="Model to use (gpt-4, gpt-3.5-turbo, claude-3)")
    parser.add_argument("--max_steps", type=int, default=20,
                       help="Maximum steps per task")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--output_dir", type=str, default="output/",
                       help="Output directory")
    parser.add_argument("--subset", type=str, default=None,
                       help="Comma-separated list of task IDs to run")
    parser.add_argument("--dataset_path", type=str, default="da_code/",
                       help="Path to dataset")
    parser.add_argument("--config", type=str, default="config/config.json",
                       help="Config file path")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = DACodeBenchmark(args.config)
    
    # Parse subset
    subset = args.subset.split(',') if args.subset else None
    
    # Run benchmark
    print(f"🚀 Starting DA-Code benchmark with model: {args.model}")
    results = benchmark.run_benchmark(
        model=args.model,
        max_steps=args.max_steps,
        num_workers=args.num_workers,
        subset=subset
    )
    
    # Save results
    benchmark.save_results(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 DA-Code Benchmark Results")
    print("="*50)
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Successful: {results['successful_tasks']}")
    print(f"Failed: {results['failed_tasks']}")
    print(f"Success rate: {results['successful_tasks']/results['total_tasks']*100:.2f}%")
    print(f"Average time: {results['average_time']:.2f}s")
    print("="*50)
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()