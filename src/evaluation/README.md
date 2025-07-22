# SWE-bench Evaluation for Codex7 Agent

This directory contains the setup and scripts for running SWE-bench evaluation on the Codex7 agent. SWE-bench is a benchmark for evaluating AI agents on real-world software engineering tasks from GitHub.

## Overview

The evaluation process involves:
1. **Setup**: Installing SWE-bench and dependencies
2. **Image Preparation**: Downloading Docker images for evaluation environments
3. **Agent Execution**: Running the Codex7 agent on SWE-bench tasks
4. **Evaluation**: Assessing the generated patches using SWE-bench harness

## Quick Start

### Prerequisites

- Docker (with at least 120GB free space)
- Conda (recommended) or Python 3.10+
- 16GB+ RAM and 8+ CPU cores
- Git

### 1. Initial Setup

Run the setup script to configure everything:

```bash
# Make the setup script executable
chmod +x src/evaluation/swebench_setup.sh

# Run the setup
./src/evaluation/swebench_setup.sh
```

This script will:
- Create a conda environment `codex7-swe` with Python 3.10
- Clone and install SWE-bench at the specified commit
- Install Codex7 agent dependencies

### 2. Activate Environment

```bash
conda activate codex7-swe
```

### 3. Run Evaluation

#### Basic Usage

```bash
# Run full evaluation (default: SWE-bench_Verified dataset)
python3 src/evaluation/swebench.py

# Run evaluation on specific instances
python3 src/evaluation/swebench.py --instance_ids django__django-11179 pandas-dev__pandas-15941

# Run on different dataset
python3 src/evaluation/swebench.py --dataset SWE-bench_Lite
```

#### Advanced Usage

```bash
# Run only patch generation (skip evaluation)
python3 src/evaluation/swebench.py --mode expr

# Run only evaluation (if patches already generated)
python3 src/evaluation/swebench.py --mode eval

# Custom working directory and config
python3 src/evaluation/swebench.py \
    --working-dir ./custom_workspace \
    --config-file custom_config.json \
    --run-id my_experiment_1
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset to use | `SWE-bench_Verified` |
| `--working-dir` | Working directory for outputs | `./codex7-workspace` |
| `--config-file` | Codex7 agent configuration file | `codex7_config_local.json` |
| `--instance_ids` | Specific instances to run (space-separated) | All instances |
| `--mode` | Run mode: `e2e`, `expr`, `eval` | `e2e` |
| `--run-id` | Identifier for this run | `codex7-agent` |
| `--swebench-harness-path` | Path to SWE-bench harness (for evaluation) | Auto-detected |
| `--docker-env-config` | Docker environment configuration file | None |

## Available Datasets

- **SWE-bench_Verified**: 500 curated, verified tasks (recommended for testing)
- **SWE-bench_Lite**: 300 lighter tasks for faster evaluation
- **SWE-bench**: Full dataset with 2,294 tasks

## Workflow Details

### 1. Image Preparation
The system automatically:
- Checks for required Docker images (~40GB download)
- Downloads missing images from DockerHub
- Prepares Ubuntu base image for agent deployment

### 2. Agent Deployment
For each task:
- Creates Docker container with the specific codebase
- Deploys Codex7 agent with dependencies
- Runs agent on the GitHub issue
- Generates patch files and trajectories

### 3. Evaluation
- Collects all generated patches
- Runs SWE-bench harness for automated testing
- Produces evaluation results and metrics

## Output Structure

```
codex7-workspace/
├── <instance_id>/                    # Per-instance directories
│   ├── problem_statement.txt        # Original GitHub issue
│   ├── <instance_id>.patch          # Generated patch
│   └── <instance_id>.json           # Agent trajectory
├── predictions.json                 # All patches for evaluation
├── codex7-agent.tar                 # Agent deployment package
└── evaluation_results/              # SWE-bench evaluation results
```

## Environment Variables

You can customize the Docker environment by creating a JSON config file:

```json
{
  "preparation_env": {
    "OPENAI_API_KEY": "your-key-here",
    "ANTHROPIC_API_KEY": "your-key-here"
  },
  "experiment_env": {
    "OPENAI_API_KEY": "your-key-here",
    "ANTHROPIC_API_KEY": "your-key-here"
  }
}
```

Then use with:
```bash
python3 src/evaluation/swebench.py --docker-env-config env_config.json
```

## Troubleshooting

### Common Issues

1. **Docker space issues**: Increase Docker disk space to 120GB+
2. **Memory issues**: Reduce `--max_workers` or increase system RAM
3. **Network timeouts**: Use VPN or configure Docker proxy
4. **Permission errors**: Ensure Docker is running and user has permissions

### Manual Setup

If the setup script fails, you can manually set up:

```bash
# 1. Create and activate environment
conda create -n codex7-swe python=3.10 -y
conda activate codex7-swe

# 2. Install SWE-bench
cd src/evaluation/SWE-bench
pip install -e .

# 3. Install Codex7 dependencies
cd ../../../
pip install -r requirements.txt
pip install -e .

# 4. Verify installation
python -c "import swebench; print('SWE-bench installed successfully')"
```

### Testing Setup

Verify your setup with a single instance:

```bash
# Test with one instance
python3 src/evaluation/swebench.py \
    --dataset SWE-bench_Lite \
    --instance_ids sympy__sympy-20590 \
    --mode e2e
```

## Performance Tips

- **For development**: Start with `SWE-bench_Lite` and 1-2 instances
- **For testing**: Use `SWE-bench_Verified` for reliable results
- **For production**: Use `SWE-bench` full dataset with appropriate compute
- **Parallel execution**: Use `--max_workers` based on available CPU cores (recommended: min(0.75 * cpu_count, 24))

## Resources

- [SWE-bench Documentation](https://swebench.com/SWE-bench/)
- [Original SWE-bench Repository](https://github.com/SWE-bench/SWE-bench)
- [Dataset Explorer](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)
- [Evaluation Guide](https://swebench.com/SWE-bench/guides/evaluation/)

## Support

For issues with:
- **SWE-bench setup**: Check [SWE-bench issues](https://github.com/SWE-bench/SWE-bench/issues)
- **Codex7 integration**: Create an issue in this repository
- **Docker/setup**: Ensure Docker daemon is running and has sufficient resources