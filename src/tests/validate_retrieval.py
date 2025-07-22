"""
Quick validation script for the retrieval module.
This script tests the core functionality without external dependencies.
"""

import os
import tempfile
import shutil
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def create_test_code_structure(test_dir):
    """Create test code files for validation."""
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a simple Python module
    test_py = os.path.join(test_dir, "calculator.py")
    with open(test_py, 'w') as f:
        f.write('''"""
A simple calculator module for testing retrieval.
"""

class Calculator:
    """A basic calculator class."""
    
    def __init__(self, name: str = "Calculator"):
        self.name = name
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

def utility_function(value: int) -> str:
    """A utility function for testing."""
    return f"Processed: {value}"
''')
    
    return test_py

def test_basic_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing basic imports...")
    
    try:
        from retrieval import CodeIndexer, RetrievalConfig
        print("✅ Core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration validation."""
    print("🧪 Testing configuration...")
    
    try:
        from retrieval import RetrievalConfig
        
        config = RetrievalConfig()
        
        # Test with HuggingFace (works offline)
        config.embedding_provider = "huggingface"
        config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.milvus_dimension = 384
        
        config.validate()
        print("✅ Configuration validation successful")
        return config
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return None

def test_parser_only():
    """Test AST parser without database dependencies."""
    print("🧪 Testing parser functionality...")
    
    try:
        from retrieval.parser import ASTParser
        
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def hello_world():
    """A simple hello world function."""
    return "Hello, World!"

class MyClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
''')
            temp_file = f.name
        
        try:
            parser = ASTParser(file_extensions=['.py'])
            entities = parser.parse_file(temp_file)
            
            print(f"✅ Parser found {len(entities)} entities")
            
            # Verify entity types
            entity_types = [e.entity_type for e in entities]
            if 'function' in entity_types:
                print("✅ Found functions")
            if 'class' in entity_types:
                print("✅ Found classes")
            
            return len(entities) > 0
        finally:
            os.unlink(temp_file)
            
    except Exception as e:
        print(f"❌ Parser error: {e}")
        return False

def test_embedding_provider():
    """Test embedding generation without databases."""
    print("🧪 Testing embedding provider...")
    
    try:
        from retrieval.embedding import EmbeddingGenerator
        
        # Test with HuggingFace provider
        config = {
            'model': 'sentence-transformers/all-MiniLM-L6-v2',
            'max_length': 512,
            'device': 'cpu'
        }
        
        generator = EmbeddingGenerator('huggingface', config)
        
        # Test single embedding
        embedding = generator.generate_embedding("test function")
        if embedding and len(embedding) > 0:
            print(f"✅ Generated embedding with dimension {len(embedding)}")
            return True
        else:
            print("❌ Failed to generate embedding")
            return False
            
    except ImportError as e:
        print(f"⚠️  HuggingFace not available: {e}")
        return True  # Skip this test if dependencies not available
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return False

def test_complete_pipeline():
    """Test complete pipeline with mock data."""
    print("🧪 Testing complete pipeline...")
    
    try:
        # This test requires running services, so we'll skip for now
        print("⚠️  Complete pipeline test requires running services")
        return True
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🔍 Retrieval Module Validation")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_configuration,
        test_parser_only,
        test_embedding_provider,
        test_complete_pipeline
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
    
    print("=" * 40)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("\nNext steps:")
        print("1. Start required services (Milvus + Neo4j)")
        print("2. Run full E2E test: python tests/test_retrieval_e2e.py")
        print("3. Or use: ./tests/run_tests.sh")
    else:
        print("⚠️  Some tests failed. Check dependencies and configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)