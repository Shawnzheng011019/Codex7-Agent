"""Configuration management for retrieval module."""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class RetrievalConfig:
    """Configuration for retrieval operations."""
    
    # Load from embedding_config.json if available
    config_path: str = field(default="src/retrieval/embedding_config.json")
    
    # Default values that can be overridden by config file
    milvus_uri: str = "http://localhost:19530"
    milvus_collection_name: str = "codex7_embeddings"
    milvus_dimension: int = 1536
    
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_max_length: int = 1000
    embedding_device: str = None
    
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout: int = 30
    
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"
    ollama_timeout: int = 30
    
    huggingface_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    max_file_size: int = 10485760  # 10MB
    supported_extensions: str = ".py,.js,.ts,.java,.cpp,.c,.h,.hpp,.go,.rs,.php,.rb,.scala,.kt,.swift,.md,.txt,.json,.yaml,.yml,.xml,.html,.css,.scss,.sql,.sh,.bat,.dockerfile"
    
    batch_size: int = 100
    
    # Proxy configuration
    proxy_enabled: bool = False
    http_proxy: str = ""
    https_proxy: str = ""
    socks_proxy: str = ""
    
    def __post_init__(self):
        """Load configuration from file if it exists."""
        self.load_from_config()
    
    def load_from_config(self) -> None:
        """Load configuration from embedding_config.json."""
        config_path = Path(self.config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Load embedding configuration
                embedding_config = config.get('embedding', {})
                self.embedding_provider = embedding_config.get('provider', self.embedding_provider)
                self.embedding_model = embedding_config.get('model', self.embedding_model)
                self.milvus_dimension = embedding_config.get('dimensions', self.milvus_dimension)
                self.embedding_max_length = embedding_config.get('chunk_size', self.embedding_max_length)
                
                # Load API configuration
                self.openai_api_key = embedding_config.get('api_key', self.openai_api_key)
                self.openai_base_url = embedding_config.get('base_url', self.openai_base_url)
                self.openai_timeout = embedding_config.get('timeout', self.openai_timeout)
                
                # Load vector store configuration
                vector_config = config.get('vector_store', {})
                self.milvus_uri = f"{vector_config.get('host', 'localhost')}:{vector_config.get('port', 19530)}"
                self.milvus_collection_name = vector_config.get('collection_name', self.milvus_collection_name)
                
                # Load proxy configuration
                proxy_config = embedding_config.get('proxy', {})
                self.proxy_enabled = proxy_config.get('enabled', self.proxy_enabled)
                self.http_proxy = proxy_config.get('http_proxy', self.http_proxy)
                self.https_proxy = proxy_config.get('https_proxy', self.https_proxy)
                self.socks_proxy = proxy_config.get('socks_proxy', self.socks_proxy)
                
            except Exception as e:
                print(f"Warning: Failed to load embedding config from {config_path}: {e}")
    
    @classmethod
    def from_config_file(cls, config_path: str = "src/retrieval/embedding_config.json") -> "RetrievalConfig":
        """Create configuration from config file."""
        return cls(config_path=config_path)
    
    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create configuration from environment variables (legacy support)."""
        config = cls()
        
        # Override with environment variables if they exist
        config.milvus_uri = os.getenv("MILVUS_URI", config.milvus_uri)
        config.milvus_collection_name = os.getenv("MILVUS_COLLECTION_NAME", config.milvus_collection_name)
        config.milvus_dimension = int(os.getenv("MILVUS_DIMENSION", str(config.milvus_dimension)))
        
        config.neo4j_uri = os.getenv("NEO4J_URI", config.neo4j_uri)
        config.neo4j_user = os.getenv("NEO4J_USER", config.neo4j_user)
        config.neo4j_password = os.getenv("NEO4J_PASSWORD", config.neo4j_password)
        
        config.embedding_provider = os.getenv("EMBEDDING_PROVIDER", config.embedding_provider)
        config.embedding_model = os.getenv("EMBEDDING_MODEL", config.embedding_model)
        config.openai_api_key = os.getenv("OPENAI_API_KEY", config.openai_api_key)
        config.openai_base_url = os.getenv("OPENAI_BASE_URL", config.openai_base_url)
        
        return config
    
    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create configuration from environment variables."""
        return cls()