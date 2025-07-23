#!/bin/bash

# CodeXGLUE Benchmark One-Click Setup Script
# Usage: ./CodeXGLUE.sh

set -e

echo "🚀 CodeXGLUE Benchmark Setup Starting..."

# Check prerequisites
if ! python3 --version | grep -E "Python 3\.[7-9]|Python 3\.[1-9][0-9]" > /dev/null; then
    echo "❌ Python 3.7+ required"
    exit 1
fi

# Check Git
if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install Git first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Clone CodeXGLUE repository
if [ ! -d "CodeXGLUE" ]; then
    echo "📥 Cloning CodeXGLUE repository..."
    git clone https://github.com/microsoft/CodeXGLUE.git
fi

cd CodeXGLUE

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install common dependencies
echo "📚 Installing dependencies..."
cat > requirements.txt << EOF
torch>=1.9.0
transformers>=4.0.0
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.24.0
tqdm>=4.50.0
datasets>=1.0.0
seqeval>=1.2.0
rouge-score>=0.0.4
bleu>=0.1.0
tree-sitter>=0.20.0
black>=21.0.0
flake8>=3.8.0
pytest>=6.0.0
jupyter>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
EOF

pip install -r requirements.txt

# Create directory structure
echo "📁 Setting up directory structure..."
mkdir -p {data,models,outputs,configs,scripts}

# Download datasets for key tasks
echo "📊 Downloading datasets..."

# Clone datasets for major tasks
if [ ! -d "data/Code-Code/code-to-code-trans" ]; then
    echo "  📥 Code-to-Code Translation dataset..."
    mkdir -p data/Code-Code/code-to-code-trans
    # TODO: Add actual dataset download commands
fi

if [ ! -d "data/Text-Code/code-generation" ]; then
    echo "  📥 Code Generation dataset..."
    mkdir -p data/Text-Code/code-generation
    # TODO: Add actual dataset download commands
fi

if [ ! -d "data/Code-Text/code-summarization" ]; then
    echo "  📥 Code Summarization dataset..."
    mkdir -p data/Code-Text/code-summarization
    # TODO: Add actual dataset download commands
fi

# Create configuration files
echo "⚙️  Creating configuration files..."
cat > configs/config.json << EOF
{
    "tasks": {
        "code_to_code": {
            "enabled": true,
            "dataset_path": "data/Code-Code/code-to-code-trans",
            "model": "microsoft/codebert-base"
        },
        "code_generation": {
            "enabled": true,
            "dataset_path": "data/Text-Code/code-generation",
            "model": "microsoft/codet5-base"
        },
        "code_summarization": {
            "enabled": true,
            "dataset_path": "data/Code-Text/code-summarization",
            "model": "microsoft/unixcoder-base"
        },
        "defect_detection": {
            "enabled": true,
            "dataset_path": "data/Code-Code/defect-detection",
            "model": "microsoft/codebert-base"
        },
        "clone_detection": {
            "enabled": true,
            "dataset_path": "data/Code-Code/clone-detection",
            "model": "microsoft/codebert-base"
        }
    },
    "training": {
        "batch_size": 16,
        "learning_rate": 5e-5,
        "epochs": 10,
        "warmup_steps": 1000,
        "logging_steps": 100,
        "save_steps": 1000,
        "eval_steps": 500
    },
    "inference": {
        "max_length": 512,
        "num_beams": 10,
        "temperature": 0.8,
        "top_p": 0.95
    }
}
EOF

# Create task-specific setup scripts
for task in "code-to-code-trans" "code-generation" "code-summarization" "defect-detection" "clone-detection"; do
    task_dir="Code-Code/$task"
    if [ -d "$task_dir" ]; then
        echo "✅ $task already exists"
    else
        echo "📁 Creating $task directory..."
        mkdir -p "$task_dir"
    fi
done

# Create environment setup
if [ ! -f ".env" ]; then
    echo "📝 Creating environment file..."
    cat > .env << EOF
# Model Configuration
MODEL_PATH=./models/
CACHE_DIR=./.cache/
OUTPUT_DIR=./outputs/

# Training Configuration
BATCH_SIZE=16
LEARNING_RATE=5e-5
EPOCHS=10

# Hardware Configuration
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
EOF
fi

# Create task runner script
cat > ../run_task.sh << 'EOF'
#!/bin/bash
# Usage: ./run_task.sh <task_name> <model_name>

TASK=$1
MODEL=$2

if [ -z "$TASK" ] || [ -z "$MODEL" ]; then
    echo "Usage: $0 <task_name> <model_name>"
    echo "Tasks: code-to-code-trans, code-generation, code-summarization, defect-detection, clone-detection"
    echo "Models: codebert, codet5, unixcoder"
    exit 1
fi

source venv/bin/activate
cd CodeXGLUE
python3 ../CodeXGLUE.py --task $TASK --model $MODEL
EOF

chmod +x ../run_task.sh

# Create data download script
cat > download_data.sh << 'EOF'
#!/bin/bash
# Download CodeXGLUE datasets

echo "📥 Downloading CodeXGLUE datasets..."

# TODO: Add actual dataset download commands
# These would typically involve:
# 1. Downloading from official sources
# 2. Extracting archives
# 3. Preprocessing data

echo "✅ Dataset download complete (placeholder)"
EOF

chmod +x download_data.sh

cd ..

echo "✅ CodeXGLUE setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Activate environment: source CodeXGLUE/venv/bin/activate"
echo "2. Download datasets: ./CodeXGLUE/download_data.sh"
echo "3. Run benchmark: python3 CodeXGLUE.py --task code-generation --model codet5"
echo "4. Available tasks: code-to-code-trans, code-generation, code-summarization, defect-detection, clone-detection"