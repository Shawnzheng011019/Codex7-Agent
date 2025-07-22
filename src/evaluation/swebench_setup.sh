#!/bin/bash
set -e

SWE_BENCH_COMMIT_HASH="2bf15e1be3c995a0758529bd29848a8987546090"

# 检查是否使用conda的codex7-swe环境
echo "Setting up SWE-bench for Codex7 agent with conda environment..."

# 检查codex7-swe环境是否存在
if ! conda info --envs | grep -q "codex7-swe"; then
    echo "Creating conda environment 'codex7-swe'..."
    conda create -n codex7-swe python=3.10 -y
else
    echo "Conda environment 'codex7-swe' already exists"
fi

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate codex7-swe

# Clone SWE-bench repo if not exists
echo "Checking SWE-bench directory..."
if [ ! -d "SWE-bench" ]; then
    echo "Cloning SWE-bench repository..."
    git clone https://github.com/SWE-bench/SWE-bench.git
else
    echo "SWE-bench directory already exists, skipping clone"
fi

cd SWE-bench

# Checkout specific commit if not already on it
if [ "$(git rev-parse HEAD)" != "$SWE_BENCH_COMMIT_HASH" ]; then
    echo "Checking out commit $SWE_BENCH_COMMIT_HASH..."
    git checkout $SWE_BENCH_COMMIT_HASH
else
    echo "Already on correct commit, skipping checkout"
fi

# Install SWE-bench if not installed
if ! python3 -c "import swebench" 2>/dev/null; then
    echo "Installing SWE-bench..."
    pip install -e .
else
    echo "SWE-bench already installed, skipping"
fi

# Install Codex7 agent dependencies
echo "Installing Codex7 agent dependencies..."
cd ..
if [ -f "requirements.txt" ] && [ -s "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Install project in development mode
echo "Installing Codex7 agent in development mode..."
if [ -f "pyproject.toml" ]; then
    pip install -e .
fi

echo ""
echo "SWE-bench setup completed for Codex7 agent!"
echo ""
echo "To activate the environment: conda activate codex7-swe"
echo "To run SWE-bench: python3 src/evaluation/swebench.py --help"