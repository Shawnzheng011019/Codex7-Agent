"""
OpenAI embedding provider implementation.

Provides embeddings using OpenAI's API services.
"""

import logging
from typing import List, Dict, Any
import os

from .base import BaseEmbeddingProvider


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider for API-based model inference."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.model = config.get('model', 'text-embedding-3-small')
        self.base_url = config.get('base_url', 'https://api.openai.com/v1')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide 'api_key' in configuration."
            )
        
        self._validate_config()
        self._setup_client()
    
    def _setup_client(self) -> None:
        """Setup OpenAI client with error handling."""
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )
            self.logger.info(f"Initialized OpenAI client with model: {self.model}")
        except ImportError:
            raise ImportError(
                "OpenAI provider requires 'openai' package. Install with: pip install openai"
            )
    
    def _validate_config(self) -> None:
        """Validate OpenAI configuration."""
        supported_models = {
            'text-embedding-3-small',
            'text-embedding-3-large',
            'text-embedding-ada-002'
        }
        
        if self.model not in supported_models:
            self.logger.warning(
                f"Model '{self.model}' may not be supported. "
                f"Supported models: {supported_models}"
            )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI."""
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return []
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text.strip()
            )
            
            if not response.data or not response.data[0].embedding:
                raise ValueError("No embedding returned from OpenAI API")
            
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"Error generating OpenAI embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI."""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            return [[] for _ in texts]
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=valid_texts
            )
            
            embeddings = []
            text_index = 0
            
            for text in texts:
                if text and text.strip():
                    embeddings.append(response.data[text_index].embedding)
                    text_index += 1
                else:
                    embeddings.append([])
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension for the configured model."""
        model_dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }
        
        return model_dimensions.get(self.model, 1536)
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        try:
            # Test with a simple embedding
            self.generate_embedding("test")
            return True
        except Exception as e:
            self.logger.error(f"OpenAI configuration invalid: {e}")
            return False
    
    def get_usage_info(self) -> Dict[str, Any]:
        """Get usage information from last API call."""
        return {
            'model': self.model,
            'provider': 'openai',
            'dimension': self.dimension
        }