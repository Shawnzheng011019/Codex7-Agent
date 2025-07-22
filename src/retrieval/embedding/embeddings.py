import logging
from typing import List, Dict, Any
from .models import VectorEntity
from .providers.factory import EmbeddingProviderFactory


class EmbeddingGenerator:
    """Embedding generator using configurable providers."""
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name.lower()
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.provider = None
        self._setup_provider()
    
    def _setup_provider(self) -> None:
        """Initialize the embedding provider."""
        try:
            self.provider = EmbeddingProviderFactory.create(
                self.provider_name,
                self.config
            )
            self.logger.info(f"Initialized embedding provider: {self.provider_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding provider: {e}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured provider."""
        if not text or not text.strip():
            self.logger.warning("Empty text provided for embedding")
            return []
        
        try:
            return self.provider.generate_embedding(text)
        except Exception as e:
            self.logger.error(f"Failed to generate embedding with {self.provider_name}: {e}")
            return []
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        try:
            return self.provider.generate_embeddings(texts)
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings with {self.provider_name}: {e}")
            return [[] for _ in texts]
    
    def generate_entity_embedding(self, entity) -> VectorEntity:
        """Generate embedding for a code entity."""
        text_parts = [entity.name, entity.docstring or "", entity.code_snippet]
        text = " ".join(filter(None, text_parts))
        vector = self.generate_embedding(text)
        
        return VectorEntity(
            entity_id=f"{entity.file_path}:{entity.name}:{entity.line_start}",
            file_path=entity.file_path,
            entity_type=entity.entity_type,
            name=entity.name,
            vector=vector,
            metadata={
                "line_start": entity.line_start,
                "line_end": entity.line_end,
                "parent": entity.parent,
                "parameters": entity.parameters,
                "return_type": entity.return_type,
                "dependencies": entity.dependencies
            },
            code_snippet=entity.code_snippet
        )
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.provider.dimension
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            'provider': self.provider_name,
            'dimension': self.dimension,
            'config': self.config
        }