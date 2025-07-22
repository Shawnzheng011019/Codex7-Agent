"""Configuration management for retrieval module."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    milvus_collection_name: str = os.getenv("MILVUS_COLLECTION_NAME", "code_vectors")
    milvus_dimension: int = int(os.getenv("MILVUS_DIMENSION", "768"))
    
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # Embedding provider configuration
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_max_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "512"))
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", None)  # auto-detect
    
    # Ollama-specific configuration
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    
    # OpenAI-specific configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_timeout: int = int(os.getenv("OPENAI_TIMEOUT", "30"))
    
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "1048576"))  # 1MB
    supported_extensions: str = os.getenv("SUPPORTED_EXTENSIONS", ".py,.js,.ts,.java,.cpp,.c,.go,.rs")
    
    batch_size: int = int(os.getenv("BATCH_SIZE", "100"))
    index_name: str = os.getenv("INDEX_NAME", "code_index")
    
    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create configuration from environment variables."""
        return cls()
    
    def validate(self) -> None:
        """Validate configuration values."""
        if not self.milvus_uri:
            raise ValueError("MILVUS_URI is required")
        if not self.neo4j_uri:
            raise ValueError("NEO4J_URI is required")
        if not self.neo4j_user or not self.neo4j_password:
            raise ValueError("NEO4J_USER and NEO4J_PASSWORD are required")
        
        # Validate embedding provider
        valid_providers = {'ollama', 'openai', 'huggingface', 'sentence-transformers'}
        if self.embedding_provider.lower() not in valid_providers:
            raise ValueError(
                f"Invalid embedding provider '{self.embedding_provider}'. "
                f"Supported providers: {valid_providers}"
            )
        
        # Validate provider-specific requirements
        if self.embedding_provider.lower() == 'openai' and not self.openai_api_key:
            raise ValueError(
                "OpenAI provider requires OPENAI_API_KEY environment variable"
            )