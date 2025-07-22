"""
Ollama embedding provider implementation.

Provides embeddings using Ollama's local embedding models.
"""

import logging
import requests
from typing import List, Dict, Any
from .base import BaseEmbeddingProvider


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local model inference."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get('host', 'http://localhost:11434')
        self.model = config.get('model', 'nomic-embed-text')
        self.timeout = config.get('timeout', 30)
        self.logger = logging.getLogger(__name__)
        
        # Validate Ollama is available
        self._validate_connection()
    
    def _validate_connection(self) -> None:
        """Validate Ollama connection and model availability."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            available_models = [m['name'] for m in models]
            
            if self.model not in available_models:
                raise ValueError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models: {available_models}"
                )
            
            self.logger.info(f"Connected to Ollama with model: {self.model}")
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.host}. "
                "Please ensure Ollama is running with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to validate Ollama connection: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using Ollama."""
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return []
        
        try:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text.strip()
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get('embedding', [])
            
            if not embedding:
                raise ValueError("No embedding returned from Ollama")
            
            return embedding
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP error generating Ollama embedding: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error generating Ollama embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using Ollama."""
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Failed to generate embedding for text: {str(e)[:100]}...")
                embeddings.append([])
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension for the configured model."""
        # Common dimensions for Ollama models
        model_dimensions = {
            'nomic-embed-text': 768,
            'mxbai-embed-large': 1024,
            'snowflake-arctic-embed': 768,
            'all-minilm': 384,
            'all-minilm-l6-v2': 384,
            'all-mpnet-base-v2': 768
        }
        
        for key, dim in model_dimensions.items():
            if key in self.model.lower():
                return dim
        
        # Default fallback - query Ollama for dimension
        try:
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        except Exception:
            self.logger.warning("Could not determine embedding dimension, using 768 as fallback")
            return 768
    
    def validate_config(self) -> bool:
        """Validate Ollama configuration."""
        try:
            self._validate_connection()
            return True
        except Exception as e:
            self.logger.error(f"Ollama configuration invalid: {e}")
            return False