# Retrieval Module

A comprehensive code indexing and retrieval system that combines AST parsing, vector embeddings, and graph databases to enable intelligent code search and analysis.

## Features

- **AST-based Code Parsing**: Extracts functions, classes, methods, and variables with full context
- **Vector Embeddings**: Generates semantic embeddings using configurable providers (HuggingFace, OpenAI, Ollama)
- **Vector Storage**: Stores embeddings in Milvus for fast similarity search
- **Graph Storage**: Stores code relationships in Neo4j for complex queries
- **Relationship Extraction**: Automatically discovers inheritance, dependencies, and file scope relationships
- **Intelligent Search**: Combine vector similarity with graph relationships

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   AST Parser    │───▶│  Embedding Gen  │───▶│  Vector Store   │
│                 │    │                 │    │    (Milvus)     │
│  • Functions    │    │  • HuggingFace  │    │                 │
│  • Classes      │    │  • OpenAI       │    │  • Similarity   │
│  • Methods      │    │  • Ollama       │    │  • Filtering    │
│  • Variables    │    │                 │    │  • Metadata     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Graph Store    │◀───│  Relationship   │◀───│   Search API    │
│    (Neo4j)      │    │   Extraction    │    │                 │
│                 │    │                 │    │  • Vector +     │
│  • Entities     │    │  • CHILD_OF     │    │    Graph        │
│  • Relationships│    │  • DEPENDS_ON   │    │  • Filtering    │
│  • Queries      │    │  • IN_FILE      │    │  • Ranking      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Required Services

```bash
# Start Milvus (vector database)
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:v2.3.3

# Start Neo4j (graph database)
docker run -d --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 3. Basic Usage

```python
from retrieval import CodeIndexer

# Create indexer
indexer = CodeIndexer()

# Index a directory
result = indexer.index_directory("/path/to/code")
print(f"Indexed {result['entities_count']} entities")

# Search for code
results = indexer.search_code("user authentication", top_k=5)
for result in results:
    print(f"Found: {result['name']} in {result['file_path']}")
    print(f"Similarity: {result['score']}")
    print(f"Related entities: {len(result['related_entities'])}")
```

## Configuration

### Environment Variables

```bash
# Vector Database (Milvus)
export MILVUS_URI=http://localhost:19530
export MILVUS_COLLECTION_NAME=code_vectors
export MILVUS_DIMENSION=768

# Graph Database (Neo4j)
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

# Embedding Provider
export EMBEDDING_PROVIDER=huggingface
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export EMBEDDING_MAX_LENGTH=512
```

### Provider-specific Configuration

#### HuggingFace (Default)
```bash
export EMBEDDING_PROVIDER=huggingface
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export EMBEDDING_DEVICE=cpu  # or cuda
```

#### OpenAI
```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL=text-embedding-3-small
```

#### Ollama
```bash
export EMBEDDING_PROVIDER=ollama
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=nomic-embed-text
```

## API Reference

### CodeIndexer

Main orchestrator for the retrieval system.

#### Methods

- `index_directory(directory: str, max_workers: int = 4) -> Dict[str, Any]`
  - Index an entire directory of code files
  - Returns statistics about the indexing process

- `search_code(query: str, top_k: int = 10) -> List[Dict[str, Any]]`
  - Search for similar code entities
  - Returns ranked results with metadata and relationships

- `get_stats() -> Dict[str, Any]`
  - Get statistics about indexed data

- `cleanup() -> None`
  - Clean up resources and connections

#### Context Manager

The indexer supports context management:

```python
with CodeIndexer() as indexer:
    result = indexer.index_directory("/path/to/code")
    # Resources automatically cleaned up
```

### Search Results Format

```json
[
  {
    "id": "file.py:ClassName:42",
    "score": 0.95,
    "file_path": "/path/to/file.py",
    "entity_type": "class",
    "name": "ClassName",
    "code_snippet": "class ClassName:\n    pass",
    "metadata": {
      "line_start": 42,
      "line_end": 45,
      "parent": null,
      "parameters": [],
      "return_type": null,
      "dependencies": []
    },
    "related_entities": [
      {
        "name": "ParentClass",
        "entity_type": "class",
        "relationship_type": "CHILD_OF"
      }
    ]
  }
]
```

## Supported Languages

- Python (`.py`)
- JavaScript (`.js`)
- TypeScript (`.ts`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`)
- Go (`.go`)
- Rust (`.rs`)

## Testing

### Running Tests

```bash
# Make test runner executable
chmod +x src/tests/run_tests.sh

# Run tests (requires Milvus and Neo4j)
./src/tests/run_tests.sh

# Or run specific test
python src/tests/test_retrieval_e2e.py
```

### Test Coverage

- Basic indexing functionality
- Search with various queries
- Entity extraction (functions, classes, methods)
- Relationship extraction (inheritance, dependencies)
- Empty directory handling
- Large file processing
- Configuration validation
- Context manager cleanup

## Examples

### Basic Indexing and Search

```python
from retrieval import CodeIndexer, RetrievalConfig

# Custom configuration
config = RetrievalConfig()
config.embedding_provider = "openai"
config.openai_api_key = "your-key"

indexer = CodeIndexer(config)

# Index code
result = indexer.index_directory("./my_project")
print(f"Indexed {result['entities_count']} entities from {result['files_count']} files")

# Search
results = indexer.search_code("database connection", top_k=3)
for result in results:
    print(f"{result['name']} ({result['entity_type']}) - Score: {result['score']}")
    print(f"File: {result['file_path']}:{result['metadata']['line_start']}")
    print(f"Code: {result['code_snippet'][:100]}...")
    print()
```

### Advanced Usage with Filtering

```python
from retrieval import CodeIndexer

with CodeIndexer() as indexer:
    # Index with progress tracking
    result = indexer.index_directory("./large_project", max_workers=8)
    
    # Get comprehensive stats
    stats = indexer.get_stats()
    print(f"Total entities: {stats['graph_store']['total_entities']}")
    
    # Find all classes that inherit from a specific base class
    classes = indexer.search_code("BaseProcessor class", top_k=10)
    for cls in classes:
        if cls['entity_type'] == 'class':
            related = cls['related_entities']
            parents = [r for r in related if r['relationship_type'] == 'CHILD_OF']
            print(f"{cls['name']} inherits from: {[p['name'] for p in parents]}")
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Ensure Milvus is running on the configured port
   - Check Neo4j credentials and connection

2. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - For HuggingFace: `pip install sentence-transformers`
   - For OpenAI: `pip install openai`

3. **Memory Issues**
   - Reduce `max_workers` parameter
   - Use smaller embedding models
   - Increase Milvus memory limits

4. **Empty Results**
   - Check file extensions in configuration
   - Verify code files contain parseable content
   - Check embedding provider connectivity

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

indexer = CodeIndexer()
# All operations will now log detailed information
```

## Performance Tips

1. **Batch Processing**: Use `max_workers` parameter for parallel processing
2. **Model Selection**: Use smaller models (MiniLM) for faster processing
3. **Database Tuning**: Configure Milvus and Neo4j for your workload
4. **Incremental Indexing**: Index only changed files for updates
5. **Memory Management**: Use context managers for automatic cleanup