#!/bin/bash

# DA-Code Benchmark One-Click Setup Script
# Usage: ./da-code.sh

set -e

echo "🚀 DA-Code Benchmark Setup Starting..."

# Check Python version
if ! python3 --version | grep -E "Python 3\.[8-9]|Python 3\.[1-9][0-9]" > /dev/null; then
    echo "❌ Python 3.8+ required"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
cat > requirements.txt << EOF
openai>=1.0.0
anthropic>=0.3.0
docker>=6.0.0
requests>=2.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
jupyter>=1.0.0
ipykernel>=6.0.0
plotly>=5.0.0
nbformat>=5.0.0
EOF

pip install -r requirements.txt

# Create directory structure
echo "📁 Setting up directory structure..."
mkdir -p da_code/{source,gold,output}
mkdir -p config

# Download DA-Code dataset
echo "📥 Downloading DA-Code dataset..."
# TODO: Add actual download commands when links are available
# For now, create placeholder structure

# Create config file
echo "⚙️  Creating configuration..."
cat > config/config.json << EOF
{
    "models": {
        "openai": {
            "api_key": "${OPENAI_API_KEY:-}",
            "base_url": "https://api.openai.com/v1"
        },
        "anthropic": {
            "api_key": "${ANTHROPIC_API_KEY:-}",
            "base_url": "https://api.anthropic.com"
        }
    },
    "docker": {
        "image": "python:3.9-slim",
        "timeout": 300
    },
    "evaluation": {
        "max_steps": 20,
        "num_workers": 4,
        "output_dir": "output/"
    }
}
EOF

# Create environment template
if [ ! -f ".env" ]; then
    echo "📝 Creating environment template..."
    cat > .env << EOF
# API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
AZURE_API_KEY=your-azure-api-key-here

# Model Configuration
DEFAULT_MODEL=gpt-4
MAX_STEPS=20
EOF
fi

# Make da-code.py executable
chmod +x da-code.py

# Download sample data if not exists
if [ ! -f "da_code/sample_tasks.json" ]; then
    echo "🎯 Creating sample tasks..."
    cat > da_code/sample_tasks.json << EOF
{
    "tasks": [
        {
            "task_id": "data_analysis_001",
            "description": "Analyze a CSV file and create a visualization",
            "prompt": "Load the data.csv file and create a bar chart showing the top 10 values",
            "ground_truth": "# Expected solution\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\ndf = pd.read_csv('data.csv')\ntop_10 = df.value_counts().head(10)\nplt.figure(figsize=(10, 6))\nplt.bar(top_10.index, top_10.values)\nplt.title('Top 10 Values')\nplt.xlabel('Categories')\nplt.ylabel('Counts')\nplt.xticks(rotation=45)\nplt.tight_layout()\nplt.savefig('output.png')",
            "test_cases": [
                {
                    "type": "file_exists",
                    "path": "output.png"
                }
            ]
        }
    ]
}
EOF
fi

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: python3 da-code.py --model gpt-4"
echo "3. Results will be saved in output/ directory"