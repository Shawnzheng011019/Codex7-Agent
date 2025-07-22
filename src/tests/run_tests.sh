#!/bin/bash
# Simple test runner for retrieval end-to-end tests

set -e

echo "🧪 Running Retrieval E2E Tests..."

# Check if required services are running
echo "🔍 Checking required services..."

# Check Milvus
if ! curl -s http://localhost:19530/health > /dev/null 2>&1; then
    echo "❌ Milvus is not running. Please start Milvus:"
    echo "   docker run -d --name milvus-standalone \"
    echo "     -p 19530:19530 \"
    echo "     -p 9091:9091 \"
    echo "     milvusdb/milvus:v2.3.3"
    exit 1
fi

# Check Neo4j
if ! nc -z localhost 7687 > /dev/null 2>&1; then
    echo "❌ Neo4j is not running. Please start Neo4j:"
    echo "   docker run -d --name neo4j \"
    echo "     -p 7474:7474 -p 7687:7687 \"
    echo "     -e NEO4J_AUTH=neo4j/password \"
    echo "     neo4j:latest"
    exit 1
fi

echo "✅ All required services are running"

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install pytest sentence-transformers neo4j pymilvus

# Run the basic test
echo "🚀 Running basic functionality test..."
cd src
python tests/test_retrieval_e2e.py

echo "✅ Basic test completed successfully!"

echo ""
echo "📋 To run the full test suite:"
echo "   pytest tests/test_retrieval_e2e.py -v"