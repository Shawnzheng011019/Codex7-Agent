"""
Factory for creating embedding providers.

Provides a unified interface for creating different embedding backends.
"""

import logging
from typing import Dict, Any, Optional
from .base import BaseEmbeddingProvider


class EmbeddingProviderFactory:
    """Factory class for creating embedding providers."""
    
    _providers = {}
    
    @classmethod
    def register(cls, name: str, provider_class):
        """Register a new provider."""
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(cls, provider_name: str, config: Dict[str, Any]) -> BaseEmbeddingProvider:
        """Create an embedding provider instance."""
        provider_name = provider_name.lower()
        
        if provider_name not in cls._providers:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available providers: {list(cls._providers.keys())}"
            )
        
        provider_class = cls._providers[provider_name]
        return provider_class(config)
    
    @classmethod
    def available_providers(cls) -> list:
        """Get list of available providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def _register_default_providers(cls):
        """Register default providers."""
        try:
            from .ollama import OllamaEmbeddingProvider
            cls.register('ollama', OllamaEmbeddingProvider)
        except ImportError as e:
            logging.getLogger(__name__).warning(f"Ollama provider not available: {e}")
        
        try:
            from .openai import OpenAIEmbeddingProvider
            cls.register('openai', OpenAIEmbeddingProvider)
        except ImportError as e:
            logging.getLogger(__name__).warning(f"OpenAI provider not available: {e}")
        
        try:
            from .huggingface import HuggingFaceEmbeddingProvider
            cls.register('huggingface', HuggingFaceEmbeddingProvider)
            cls.register('sentence-transformers', HuggingFaceEmbeddingProvider)
        except ImportError as e:
            logging.getLogger(__name__).warning(f"HuggingFace provider not available: {e}")


# Register default providers when module is imported
EmbeddingProviderFactory._register_default_providers()