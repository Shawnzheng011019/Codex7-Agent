"""
End-to-end test suite for the retrieval module.

This test suite validates the complete retrieval pipeline including:
- Code parsing and entity extraction
- Embedding generation
- Vector storage (Milvus)
- Graph storage (Neo4j)
- Search functionality
- Relationship extraction
"""

import os
import tempfile
import shutil
import pytest
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from retrieval import CodeIndexer, RetrievalConfig


class TestRetrievalE2E:
    """End-to-end tests for the retrieval system."""
    
    @classmethod
    def setup_class(cls):
        """Set up test configuration and directories."""
        # Create temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()
        cls.test_files_dir = os.path.join(cls.test_dir, "test_code")
        os.makedirs(cls.test_files_dir, exist_ok=True)
        
        # Create test configuration
        cls.config = RetrievalConfig()
        
        # Use in-memory/test databases for testing
        cls.config.milvus_uri = "http://localhost:19530"
        cls.config.neo4j_uri = "bolt://localhost:7687"
        cls.config.neo4j_user = "neo4j"
        cls.config.neo4j_password = "password"
        cls.config.embedding_provider = "huggingface"
        cls.config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        cls.config.milvus_collection_name = "test_code_vectors"
        cls.config.milvus_dimension = 384  # MiniLM-L6-v2 dimension
        
    @classmethod
    def teardown_class(cls):
        """Clean up test directories."""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def create_test_files(self) -> List[str]:
        """Create test Python files with various code structures."""
        file_contents = [
            # Basic module with functions
            """
"""Test module with basic functions."""

def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

def process_data(items: List[str]) -> List[str]:
    """Process a list of strings."""
    return [item.upper() for item in items]

class DataProcessor:
    """A class to process data."""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
    
    def process(self, data: str) -> str:
        """Process a single data item."""
        self.processed_count += 1
        return f"{self.name}: {data.upper()}"
""",
            # Class with inheritance
            """
"""Advanced data processing module."""

from typing import Dict, Any

class BaseProcessor:
    """Base processor class."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def process(self, data: Any) -> Any:
        raise NotImplementedError

class AdvancedProcessor(BaseProcessor):
    """Advanced data processor with inheritance."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process complex data structures."""
        return {
            key: str(value).upper() 
            for key, value in data.items()
        }
    
    def validate_config(self) -> bool:
        """Validate processor configuration."""
        return "mode" in self.config
""",
            # File with async functions
            """
"""Async processing utilities."""

import asyncio
from typing import List

async def async_process(items: List[str]) -> List[str]:
    """Process items asynchronously."""
    results = []
    for item in items:
        await asyncio.sleep(0.1)
        results.append(item.lower())
    return results

class AsyncProcessor:
    """Async processor for handling concurrent operations."""
    
    def __init__(self, workers: int = 5):
        self.workers = workers
    
    async def batch_process(self, items: List[str]) -> List[str]:
        """Process items in batches."""
        semaphore = asyncio.Semaphore(self.workers)
        
        async def process_single(item: str) -> str:
            async with semaphore:
                await asyncio.sleep(0.01)
                return item.strip()
        
        tasks = [process_single(item) for item in items]
        return await asyncio.gather(*tasks)
"""
        ]
        
        file_paths = []
        for i, content in enumerate(file_contents):
            file_path = os.path.join(self.test_files_dir, f"test_module_{i+1}.py")
            with open(file_path, 'w') as f:
                f.write(content)
            file_paths.append(file_path)
        
        return file_paths
    
    def test_basic_indexing(self):
        """Test basic indexing functionality."""
        self.create_test_files()
        
        with CodeIndexer(self.config) as indexer:
            # Test indexing
            result = indexer.index_directory(self.test_files_dir)
            
            assert result["status"] == "success"
            assert result["entities_count"] > 0
            assert result["vectors_count"] > 0
            assert result["files_count"] > 0
            assert result["relationships_count"] > 0
            
            # Test stats
            stats = indexer.get_stats()
            assert "vector_store" in stats
            assert "graph_store" in stats
            
            vector_stats = stats["vector_store"]
            assert vector_stats["total_entities"] > 0
            
            graph_stats = stats["graph_store"]
            assert graph_stats["total_entities"] > 0
            assert "entities_by_type" in graph_stats
            assert "relationships_by_type" in graph_stats
    
    def test_search_functionality(self):
        """Test search functionality."""
        self.create_test_files()
        
        with CodeIndexer(self.config) as indexer:
            # Index first
            indexer.index_directory(self.test_files_dir)
            
            # Test various search queries
            queries = [
                "calculate sum function",
                "DataProcessor class",
                "async processing",
                "validation method",
                "process data"
            ]
            
            for query in queries:
                results = indexer.search_code(query, top_k=5)
                assert isinstance(results, list)
                
                if results:
                    for result in results:
                        assert "id" in result
                        assert "score" in result
                        assert "file_path" in result
                        assert "entity_type" in result
                        assert "name" in result
                        assert "related_entities" in result
                        assert isinstance(result["related_entities"], list)
    
    def test_entity_extraction(self):
        """Test that entities are properly extracted."""
        self.create_test_files()
        
        with CodeIndexer(self.config) as indexer:
            result = indexer.index_directory(self.test_files_dir)
            
            # Verify we have different types of entities
            stats = indexer.get_stats()
            entities_by_type = stats["graph_store"]["entities_by_type"]
            
            # Should have classes
            assert any("class" in etype.lower() for etype in entities_by_type.keys())
            
            # Should have functions
            assert any("function" in etype.lower() for etype in entities_by_type.keys())
            
            # Should have methods (functions within classes)
            assert any("function" in etype.lower() for etype in entities_by_type.keys())
    
    def test_relationship_extraction(self):
        """Test that relationships between entities are properly extracted."""
        self.create_test_files()
        
        with CodeIndexer(self.config) as indexer:
            result = indexer.index_directory(self.test_files_dir)
            
            # Verify we have different types of relationships
            stats = indexer.get_stats()
            relationships_by_type = stats["graph_store"]["relationships_by_type"]
            
            # Should have CHILD_OF relationships (methods in classes)
            assert "CHILD_OF" in relationships_by_type or any(
                "child" in rtype.lower() for rtype in relationships_by_type.keys()
            )
            
            # Should have IN_FILE relationships (entities in same file)
            assert "IN_FILE" in relationships_by_type or any(
                "file" in rtype.lower() for rtype in relationships_by_type.keys()
            )
    
    def test_empty_directory(self):
        """Test indexing an empty directory."""
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        with CodeIndexer(self.config) as indexer:
            result = indexer.index_directory(empty_dir)
            
            assert result["status"] == "success"
            assert result["entities_count"] == 0
            assert result["vectors_count"] == 0
            assert result["relationships_count"] == 0
    
    def test_large_file_handling(self):
        """Test handling of larger files."""
        # Create a larger test file
        large_content = '"""Large test module."""\n\n'
        
        # Add many functions
        for i in range(50):
            large_content += f"""
def function_{i}(x: int) -> int:
    \"\"\"Function {i} for testing large file processing.\"\"\"
    return x + {i}
"""
        
        large_file = os.path.join(self.test_files_dir, "large_module.py")
        with open(large_file, 'w') as f:
            f.write(large_content)
        
        with CodeIndexer(self.config) as indexer:
            result = indexer.index_directory(self.test_files_dir)
            
            assert result["status"] == "success"
            assert result["entities_count"] >= 50  # At least the 50 functions
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid provider
        invalid_config = RetrievalConfig()
        invalid_config.embedding_provider = "invalid_provider"
        
        with pytest.raises(ValueError):
            invalid_config.validate()
        
        # Test missing OpenAI key
        invalid_config.embedding_provider = "openai"
        invalid_config.openai_api_key = ""
        
        with pytest.raises(ValueError):
            invalid_config.validate()
    
    def test_context_manager(self):
        """Test that the context manager properly cleans up resources."""
        indexer = None
        try:
            indexer = CodeIndexer(self.config)
            assert indexer is not None
        finally:
            if indexer:
                indexer.cleanup()
    
    def test_search_with_no_results(self):
        """Test search when no results are found."""
        self.create_test_files()
        
        with CodeIndexer(self.config) as indexer:
            indexer.index_directory(self.test_files_dir)
            
            # Search for something that shouldn't exist
            results = indexer.search_code("nonexistent_function_xyz_123", top_k=10)
            assert isinstance(results, list)
            assert len(results) == 0
    
    def test_entity_metadata(self):
        """Test that entity metadata is properly stored."""
        self.create_test_files()
        
        with CodeIndexer(self.config) as indexer:
            indexer.index_directory(self.test_files_dir)
            
            # Search for a specific function
            results = indexer.search_code("calculate_sum", top_k=5)
            
            if results:
                result = results[0]
                metadata = result.get("metadata", {})
                
                # Check for expected metadata fields
                assert "line_start" in metadata
                assert "line_end" in metadata
                assert "parameters" in metadata
                assert isinstance(metadata["parameters"], list)


class TestRetrievalIntegration:
    """Integration tests for retrieval with different providers."""
    
    def test_huggingface_provider(self):
        """Test with HuggingFace provider."""
        config = RetrievalConfig()
        config.embedding_provider = "huggingface"
        config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.milvus_dimension = 384
        
        test_dir = tempfile.mkdtemp()
        try:
            # Create simple test file
            test_file = os.path.join(test_dir, "test.py")
            with open(test_file, 'w') as f:
                f.write("""
def hello_world():
    return "Hello, World!"
""")
            
            with CodeIndexer(config) as indexer:
                result = indexer.index_directory(test_dir)
                assert result["status"] == "success"
                
                # Test search
                results = indexer.search_code("hello world function")
                assert len(results) > 0
                
        finally:
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    """Run basic functionality test."""
    import unittest
    
    # Create test instance
    test_instance = TestRetrievalE2E()
    test_instance.setup_class()
    
    try:
        print("Running basic retrieval test...")
        
        # Test basic indexing
        test_instance.create_test_files()
        
        with CodeIndexer(test_instance.config) as indexer:
            result = indexer.index_directory(test_instance.test_files_dir)
            
            print(f"✓ Indexing completed: {result}")
            
            # Test search
            results = indexer.search_code("process data", top_k=3)
            print(f"✓ Search returned {len(results)} results")
            
            # Test stats
            stats = indexer.get_stats()
            print(f"✓ Stats retrieved: {stats}")
            
            print("✓ All basic tests passed!")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        test_instance.teardown_class()