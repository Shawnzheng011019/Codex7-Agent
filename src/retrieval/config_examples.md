# Environment Configuration Examples

## HuggingFace (Default)
```bash
export EMBEDDING_PROVIDER=huggingface
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export EMBEDDING_MAX_LENGTH=512
export EMBEDDING_DEVICE=cpu
```

## Ollama
```bash
export EMBEDDING_PROVIDER=ollama
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=nomic-embed-text
export OLLAMA_TIMEOUT=30
```

## OpenAI
```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your-api-key-here
export OPENAI_MODEL=text-embedding-3-small
export OPENAI_TIMEOUT=30
```

## Additional Configuration
```bash
export MILVUS_URI=http://localhost:19530
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
export MAX_FILE_SIZE=1048576
export SUPPORTED_EXTENSIONS=.py,.js,.ts,.java,.cpp,.c,.go,.rs
```