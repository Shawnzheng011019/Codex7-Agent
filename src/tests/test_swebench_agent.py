"""
SWE-bench evaluation for Codex7-Agent.
Tests the agent's ability to solve software engineering problems.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio
from datasets import load_dataset
from docker import DockerClient, from_env
from docker.errors import ImageNotFound
from docker.models.containers import Container, ExecResult
from tqdm import tqdm

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.codex7_agent.agent.orchestrator import AgentLoopOrchestrator, AgentConfig


def docker_exec(container: Container, command: str) -> tuple[int, str]:
    """
    Execute a command in a docker container.

    Args:
        container: The docker container object.
        command: The command to execute.

    Returns:
        A tuple of (return_code, output).
    """
    exec_result: ExecResult = container.exec_run(cmd=command)
    return_code = exec_result[0]
    output = exec_result[1].decode("utf-8")
    return return_code, output


class Codex7AgentSWEBenchEvaluation:
    """SWE-bench evaluation using Codex7-Agent."""
    
    def __init__(
        self,
        working_dir: str,
        config_file: str,
        dataset: str = "SWE-bench_Verified",
        docker_env_config: str = "",
        swebench_harness_path: str = "",
        run_id: str = "codex7-agent",
        swebench_config: str = "",
    ):
        """
        Initialize the Codex7AgentSWEBenchEvaluation class.

        Args:
            working_dir: The working directory for storing results.
            config_file: Path to the Codex7-Agent config file.
            dataset: The SWE-bench dataset to evaluate.
            docker_env_config: Path to docker environment config file.
            swebench_harness_path: Path to SWE-bench harness.
            run_id: Unique identifier for this evaluation run.
        """
        assert dataset in ["SWE-bench", "SWE-bench_Lite", "SWE-bench_Verified"], (
            f"Invalid dataset name: {dataset}"
        )
        
        self.dataset = load_dataset(f"princeton-nlp/{dataset}", split="test")
        self.dataset_name = dataset
        
        self.docker_client: DockerClient = from_env()
        self.image_status: Dict[Any, Any] = {}
        self.working_dir = Path(working_dir)
        self.swebench_harness_path = swebench_harness_path
        self.run_id = run_id
        
        # Load docker environment configuration
        if docker_env_config and os.path.exists(docker_env_config):
            with open(docker_env_config, "r") as f:
                self.docker_env_config: Dict[str, Dict[str, str]] = json.load(f)
        else:
            self.docker_env_config = {}
        
        # Load SWE-bench configuration
        if swebench_config and os.path.exists(swebench_config):
            with open(swebench_config, "r") as f:
                self.swebench_config: Dict[str, Any] = json.load(f)
        else:
            # Default configuration
            self.swebench_config = {
                "evaluation": {
                    "max_loops": 200,
                    "timeout_seconds": 3600
                },
                "docker": {
                    "base_image": "ubuntu:22.04"
                },
                "agent": {
                    "workspace_path": "/testbed/",
                    "max_context_size": 50,
                    "enable_task_tool": True,
                    "enable_cache": True,
                    "log_level": "INFO"
                }
            }
        
        # Create working directory if it doesn't exist
        if not self.working_dir.exists():
            self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = config_file
        
        # Copy config file to working directory
        shutil.copyfile(self.config_file, self.working_dir / "codex7_config.json")
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.working_dir / "evaluation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.pull_images()
    
    def _image_name(self, instance_id: str) -> str:
        """Get the Docker image name for a given instance ID."""
        key = f"swebench/sweb.eval.x86_64.{instance_id.lower()}:latest"
        key = key.replace("__", "_1776_")
        return key
    
    def _check_images(self):
        """Check the existence of required Docker images."""
        for item in tqdm(self.dataset, desc="Checking image status"):
            instance_id: str = item["instance_id"]
            image_name = self._image_name(instance_id)
            try:
                _ = self.docker_client.images.get(image_name)
                self.image_status[instance_id] = True
            except ImageNotFound:
                self.image_status[instance_id] = False
        
        # Ensure base Ubuntu image exists
        base_image = self.swebench_config["docker"]["base_image"]
        try:
            _ = self.docker_client.images.get(base_image)
        except Exception:
            self.docker_client.images.pull(base_image)
    
    def pull_images(self):
        """Pull required Docker images."""
        self._check_images()
        self.logger.info(f"Total number of images: {len(self.image_status)}")
        
        instance_ids = [
            instance_id for instance_id in self.image_status 
            if not self.image_status[instance_id]
        ]
        self.logger.info(f"Number of images to download: {len(instance_ids)}")
        
        if len(instance_ids) == 0:
            return
            
        for instance_id in tqdm(instance_ids, desc="Downloading images"):
            image_name = self._image_name(instance_id)
            self.docker_client.images.pull(image_name)
    
    def prepare_codex7_agent(self):
        """Prepare Codex7-Agent artifacts for Docker containers."""
        tars = ["codex7-agent.tar", "uv.tar", "uv_shared.tar"]
        all_exist = True
        
        for tar in tars:
            tar_path = self.working_dir / tar
            if not tar_path.exists():
                all_exist = False
                break
        
        if all_exist:
            self.logger.info("Found built codex7-agent and uv artifacts. Skipping building.")
            return
        
        base_image = self.swebench_config["docker"]["base_image"]
        try:
            image = self.docker_client.images.get(base_image)
        except Exception:
            image = self.docker_client.images.pull(base_image)
        
        repo_root_path = project_root
        assert (repo_root_path / "src" / "codex7_agent" / "__init__.py").is_file()
        
        container = self.docker_client.containers.run(
            image=image,
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                self.working_dir.absolute().as_posix(): {"bind": "/codex7-workspace", "mode": "rw"},
                repo_root_path.absolute().as_posix(): {"bind": "/codex7-src", "mode": "ro"},
            },
            environment=self.docker_env_config.get("preparation_env", {}),
        )
        
        commands = [
            "apt-get update",
            "apt-get install -y curl git python3 python3-pip python3-venv",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "rm -rf /codex7-workspace/codex7-agent && mkdir /codex7-workspace/codex7-agent",
            "cp -r /codex7-src/src /codex7-workspace/codex7-agent/",
            "cp /codex7-src/codex7_config.json /codex7-workspace/codex7-agent/",
            "cp /codex7-src/requirements.txt /codex7-workspace/codex7-agent/",
            "cp /codex7-src/README.md /codex7-workspace/codex7-agent/",
            "cd /codex7-workspace/codex7-agent && source $HOME/.local/bin/env && uv sync",
        ]
        
        for command in tqdm(commands, desc="Building codex7-agent inside base Docker container"):
            try:
                new_command = f'/bin/bash -c "{command}"'
                return_code, output = docker_exec(container, new_command)
            except Exception:
                self.logger.error(f"{command} failed.")
                self.logger.error(traceback.format_exc())
                break
            if return_code is not None and return_code != 0:
                self.logger.error(f"Docker exec error. Error message: {output}")
                sys.exit(-1)
        
        # Create tar archives
        with open(self.working_dir / "codex7-agent.tar", "wb") as f:
            bits, _ = container.get_archive("/codex7-workspace/codex7-agent")
            for chunk in bits:
                f.write(chunk)
        
        with open(self.working_dir / "uv.tar", "wb") as f:
            bits, _ = container.get_archive("/root/.local/bin/uv")
            for chunk in bits:
                f.write(chunk)
        
        with open(self.working_dir / "uv_shared.tar", "wb") as f:
            bits, _ = container.get_archive("/root/.local/share/uv")
            for chunk in bits:
                f.write(chunk)
        
        container.stop()
        container.remove()
    
    def prepare_experiment_container(self, instance: Dict[str, str]) -> Container:
        """Prepare an experiment Docker container for a given instance."""
        image_name = self._image_name(instance["instance_id"])
        
        instance_dir = self.working_dir / instance["instance_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        # Write problem statement
        with open(instance_dir / "problem_statement.txt", "w") as f:
            f.write(instance["problem_statement"])
        
        container: Container = self.docker_client.containers.run(
            image_name,
            command="/bin/bash",
            detach=True,
            tty=True,
            stdin_open=True,
            volumes={
                self.working_dir.absolute().as_posix(): {"bind": "/codex7-workspace", "mode": "rw"}
            },
            working_dir="/codex7-workspace",
            environment=self.docker_env_config.get("experiment_env", {}),
            stream=True,
        )
        
        commands = [
            "tar xf codex7-agent.tar",
            "tar xf uv.tar",
            "mkdir -p /root/.local/bin",
            "mv uv /root/.local/bin/",
            "tar xf uv_shared.tar",
            "mkdir -p /root/.local/share",
            "mv uv /root/.local/share/",
        ]
        
        for command in commands:
            try:
                new_command = f'/bin/bash -c "{command}"'
                return_code, output = docker_exec(container, new_command)
                if return_code is not None and return_code != 0:
                    self.logger.error(f"Docker exec error. Error message: {output}")
            except Exception:
                self.logger.error(f"{command} failed.")
                self.logger.error(traceback.format_exc())
                break
        
        return container
    
    async def run_one_instance(self, instance_id: str):
        """Run a single instance using Codex7-Agent."""
        instance: Optional[Dict[str, str]] = None
        for inst in self.dataset:
            if inst["instance_id"] == instance_id:
                instance = inst
                break
        
        if instance is None:
            self.logger.error(f"Instance {instance_id} not found.")
            return
        
        container = self.prepare_experiment_container(instance)
        instance_dir = instance["instance_id"]
        problem_statement_path = instance_dir + "/problem_statement.txt"
        patch_file_path = instance_dir + f"/{instance['instance_id']}.patch"
        traj_path = instance_dir + f"/{instance['instance_id']}.json"
        
        # Create a simple Python script to run the agent
        agent_script = f"""
import asyncio
import sys
import json
from pathlib import Path

# Add the agent path
sys.path.insert(0, '/codex7-workspace/codex7-agent/src')

from codex7_agent.agent.orchestrator import AgentLoopOrchestrator, AgentConfig

async def main():
    # Load configuration
    config = AgentConfig(
        workspace_path="/testbed/",
        config_path="/codex7-workspace/codex7-agent/codex7_config.json",
        max_loops=200
    )
    
    # Create agent
    agent = AgentLoopOrchestrator(config)
    
    # Read problem statement
    with open('{problem_statement_path}', 'r') as f:
        problem_statement = f.read()
    
    # Run agent
    result = await agent.process_message(problem_statement)
    
    # Save trajectory
    with open('{traj_path}', 'w') as f:
        json.dump(result, f, indent=2)
    
    # Generate patch if available
    if 'patch' in result:
        with open('{patch_file_path}', 'w') as f:
            f.write(result['patch'])

if __name__ == "__main__":
    asyncio.run(main())
"""
        
        # Write agent script
        script_path = self.working_dir / f"{instance_id}_agent_script.py"
        with open(script_path, "w") as f:
            f.write(agent_script)
        
        command = f'cd /codex7-workspace/codex7-agent && source .venv/bin/activate && python /codex7-workspace/{instance_id}_agent_script.py'
        new_command = f"/bin/bash -c '{command}'"
        
        try:
            return_code, output = docker_exec(container, new_command)
            if return_code is not None and return_code != 0:
                self.logger.error(f"Docker exec error. Error message: {output}")
        except Exception:
            self.logger.error(f"{command} failed.")
            self.logger.error(traceback.format_exc())
        
        container.stop()
        container.remove()
    
    async def run_all(self):
        """Run all instances in the dataset."""
        for instance in tqdm(self.dataset, desc="Running all instances"):
            await self.run_one_instance(instance["instance_id"])
    
    def run_eval(self):
        """Run evaluation using the SWE-bench harness."""
        if not self.swebench_harness_path:
            self.logger.error("SWE-bench harness path not provided. Skipping evaluation.")
            return
        
        swebench_harness_path = Path(self.swebench_harness_path)
        swebench_python_path = "swebench_venv/bin/python"
        
        cmd = [
            swebench_python_path,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            f"princeton-nlp/{self.dataset_name}",
            "--predictions_path",
            (self.working_dir / "predictions.json").absolute().as_posix(),
            "--run_id",
            self.run_id,
            "--cache_level",
            "instance",
            "--instance_image_tag",
            "latest",
        ]
        
        process = subprocess.run(cmd, capture_output=True, cwd=swebench_harness_path.as_posix())
        self.logger.info(process.stdout.decode())
        self.logger.error(process.stderr.decode())
        
        result_filename = f"codex7-agent.{self.run_id}.json"
        self.logger.info(f"Evaluation completed and file saved to {result_filename}")
    
    def get_all_preds(self, instance_ids: Optional[List[str]] = None):
        """Get all predictions for a list of instance IDs."""
        preds: List[Dict[str, str]] = []
        if not instance_ids:
            instance_ids = [instance["instance_id"] for instance in self.dataset]
        
        for instance_id in instance_ids:
            patch_path = self.working_dir / instance_id / f"{instance_id}.patch"
            if not patch_path.exists():
                continue
            
            with open(patch_path, "r") as f:
                patch = f.read()
            
            preds.append({
                "instance_id": instance_id,
                "model_name_or_path": "codex7-agent",
                "model_patch": patch,
            })
        
        with open(self.working_dir / "predictions.json", "w") as f:
            json.dump(preds, f, indent=2)


async def main():
    """Main function to run SWE-bench evaluation."""
    argument_parser = argparse.ArgumentParser(description="Codex7-Agent SWE-bench Evaluation")
    argument_parser.add_argument("--dataset", type=str, default="SWE-bench_Verified")
    argument_parser.add_argument("--working-dir", type=str, default="./codex7-workspace")
    argument_parser.add_argument("--config-file", type=str, default="codex7_config.json")
    argument_parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    argument_parser.add_argument(
        "--swebench-harness-path",
        type=str,
        default="",
        required=False,
        help="Path to SWE-bench harness (only used for evaluation).",
    )
    argument_parser.add_argument("--docker-env-config", type=str, default="", required=False)
    argument_parser.add_argument(
        "--run-id",
        type=str,
        required=False,
        default="codex7-agent",
        help="Run ID for SWE-bench evaluation.",
    )
    argument_parser.add_argument(
        "--mode",
        type=str,
        choices=["e2e", "expr", "eval"],
        default="e2e",
        help="e2e: both expr and eval, expr: only generate patches, eval: only evaluation patches",
    )
    
    args = argument_parser.parse_args()
    
    evaluation = Codex7AgentSWEBenchEvaluation(
        args.working_dir,
        args.config_file,
        args.dataset,
        args.docker_env_config,
        args.swebench_harness_path,
        args.run_id,
    )
    
    if args.mode == "e2e" or args.mode == "expr":
        evaluation.prepare_codex7_agent()
        
        if args.instance_ids:
            print(f"Running instances: {args.instance_ids}")
            for instance_id in tqdm(args.instance_ids, desc="Running instances"):
                await evaluation.run_one_instance(instance_id)
        else:
            print("Running all instances")
            await evaluation.run_all()
    
    if args.mode == "e2e" or args.mode == "eval":
        evaluation.get_all_preds(args.instance_ids)
        evaluation.run_eval()


if __name__ == "__main__":
    asyncio.run(main()) 