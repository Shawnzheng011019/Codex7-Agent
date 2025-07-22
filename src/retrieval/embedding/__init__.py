"""
Embedding module for vector storage and similarity search.

This module handles text embeddings and vector database operations.
"""

from .models import VectorEntity
from .embeddings import EmbeddingGenerator
from .milvus_store import MilvusVectorStore

__all__ = ["VectorEntity", "EmbeddingGenerator", "MilvusVectorStore"]