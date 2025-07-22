from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import RetrievalConfig
from .parser.ast_parser import ASTParser
from .parser.entities import CodeEntity
from .embedding.embeddings import EmbeddingGenerator
from .embedding.milvus_store import MilvusVectorStore
from .embedding.models import VectorEntity
from .graph.neo4j_store import Neo4jGraphStore


class RetrievalFactory:
    @staticmethod
    def create_ast_parser(config: RetrievalConfig) -> ASTParser:
        return ASTParser(file_extensions=config.supported_extensions.split(","))

    @staticmethod
    def create_embedding_generator(config: RetrievalConfig) -> EmbeddingGenerator:
        provider_name = config.embedding_provider.lower()
        provider_config = RetrievalFactory._get_provider_config(config)
        return EmbeddingGenerator(provider_name, provider_config)

    @staticmethod
    def create_vector_store(config: RetrievalConfig) -> MilvusVectorStore:
        return MilvusVectorStore(config)

    @staticmethod
    def create_graph_store(config: RetrievalConfig) -> Neo4jGraphStore:
        return Neo4jGraphStore(config)
    
    @staticmethod
    def _get_provider_config(config: RetrievalConfig) -> Dict[str, Any]:
        """Get provider-specific configuration based on the selected provider."""
        provider = config.embedding_provider.lower()
        
        if provider == 'ollama':
            return {
                'host': config.ollama_host,
                'model': config.ollama_model,
                'timeout': config.ollama_timeout,
                'max_length': config.embedding_max_length
            }
        elif provider == 'openai':
            return {
                'api_key': config.openai_api_key,
                'model': config.openai_model,
                'base_url': config.openai_base_url,
                'timeout': config.openai_timeout,
                'max_length': config.embedding_max_length
            }
        elif provider in ['huggingface', 'sentence-transformers']:
            return {
                'model': config.embedding_model,
                'max_length': config.embedding_max_length,
                'device': config.embedding_device
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")