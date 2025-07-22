"""
Abstract base class for embedding providers.

This module defines the interface for different embedding backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass
    
    def validate_config(self) -> bool:
        """Validate provider configuration."""
        return True