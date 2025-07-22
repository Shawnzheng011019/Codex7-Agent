"""
HuggingFace/Sentence-Transformers embedding provider implementation.

Provides embeddings using local HuggingFace models.
"""

import logging
from typing import List, Dict, Any
import os

from .base import BaseEmbeddingProvider


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """HuggingFace embedding provider for local model inference."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.max_length = config.get('max_length', 512)
        self.device = config.get('device', None)  # auto-detect
        self.logger = logging.getLogger(__name__)
        
        self._setup_model()
    
    def _setup_model(self) -> None:
        """Setup the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            self.logger.info(f"Loaded HuggingFace model: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "HuggingFace provider requires 'sentence-transformers' package. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using HuggingFace."""
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return []
        
        try:
            if len(text) > self.max_length:
                text = text[:self.max_length]
            
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating HuggingFace embedding: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using HuggingFace."""
        if not texts:
            return []
        
        try:
            valid_texts = [t[:self.max_length] if t and len(t) > self.max_length else t or "" for t in texts]
            
            embeddings = self.model.encode(valid_texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
            
        except Exception as e:
            self.logger.error(f"Error generating HuggingFace embeddings: {e}")
            raise
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension for the configured model."""
        try:
            # Get dimension from model
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback to common dimensions
            model_name_lower = self.model_name.lower()
            if 'mini' in model_name_lower:
                return 384
            elif 'base' in model_name_lower:
                return 768
            elif 'large' in model_name_lower:
                return 1024
            else:
                # Try to get dimension from actual embedding
                try:
                    test_embedding = self.generate_embedding("test")
                    return len(test_embedding)
                except Exception:
                    self.logger.warning(
                        "Could not determine embedding dimension, using 768 as fallback"
                    )
                    return 768
    
    def validate_config(self) -> bool:
        """Validate HuggingFace configuration."""
        try:
            # Test with a simple embedding
            self.generate_embedding("test")
            return True
        except Exception as e:
            self.logger.error(f"HuggingFace configuration invalid: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'max_length': self.max_length,
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown'
        }