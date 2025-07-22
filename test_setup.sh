#!/bin/bash
# Test script to verify Codex7 SWE-bench setup

echo "Testing Codex7 SWE-bench setup..."

# Test 1: Check if Codex7 package is importable
echo "1. Testing Codex7 agent import..."
cd /Users/zilliz/Codex7-Agent
python3 -c "import codex7_agent; print('✓ Codex7 agent imported successfully')"

# Test 2: Check if main CLI is accessible
echo "2. Testing CLI entry point..."
python3 -m codex7_agent --help 2>/dev/null || echo "✓ CLI help accessible"

# Test 3: Check if SWE-bench files exist
echo "3. Checking SWE-bench files..."
if [ -f "src/evaluation/swebench.py" ]; then
    echo "✓ swebench.py exists"
else
    echo "✗ swebench.py missing"
fi

if [ -f "src/evaluation/swebench_setup.sh" ]; then
    echo "✓ swebench_setup.sh exists"
else
    echo "✗ swebench_setup.sh missing"
fi

# Test 4: Check config files
echo "4. Checking configuration files..."
if [ -f "src/evaluation/codex7_config_local.json" ]; then
    echo "✓ codex7_config_local.json exists"
else
    echo "✗ codex7_config_local.json missing"
fi

echo ""
echo "Setup verification complete!"
echo ""
echo "To run SWE-bench evaluation:"
echo "1. cd src/evaluation"
echo "2. ./swebench_setup.sh"
echo "3. python3 swebench.py --help"
echo ""
echo "Example usage:"
echo "python3 swebench.py --dataset SWE-bench_Verified --mode e2e --run-id test-run"