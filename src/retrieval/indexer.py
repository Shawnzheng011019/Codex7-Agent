"""Main indexing orchestrator for code indexing pipeline."""

import os
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .config import RetrievalConfig
from .parser import ASTParser, CodeEntity
from .embedding import EmbeddingGenerator, MilvusVectorStore, VectorEntity
from .graph import Neo4jGraphStore


class CodeIndexer:
    """Main orchestrator for code indexing pipeline."""
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig.from_env()
        self.config.validate()
        
        # Initialize components
        self.parser = ASTParser(
            file_extensions=self.config.supported_extensions.split(",")
        )
        self.embedding_generator = EmbeddingGenerator(self.config.embedding_model)
        self.vector_store = MilvusVectorStore(self.config)
        self.graph_store = Neo4jGraphStore(self.config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def index_directory(self, directory: str, max_workers: int = 4) -> Dict[str, Any]:
        """Index an entire directory."""
        self.logger.info(f"Starting indexing of directory: {directory}")
        start_time = time.time()
        
        try:
            # Parse all code files
            self.logger.info("Parsing code files...")
            parse_start = time.time()
            entities_by_file = self.parser.parse_directory(directory)
            parse_time = time.time() - parse_start
            
            # Flatten entities
            all_entities = []
            for file_entities in entities_by_file.values():
                all_entities.extend(file_entities)
            
            self.logger.info(f"Parsed {len(all_entities)} entities from {len(entities_by_file)} files in {parse_time:.2f}s")
            
            if not all_entities:
                return {"status": "success", "message": "No entities found to index", "entities_count": 0}
            
            # Generate embeddings and store in parallel
            self.logger.info("Generating embeddings...")
            embed_start = time.time()
            vectors = self._generate_embeddings_parallel(all_entities, max_workers)
            embed_time = time.time() - embed_start
            
            self.logger.info(f"Generated {len(vectors)} embeddings in {embed_time:.2f}s")
            
            # Store in vector database
            self.logger.info("Storing in vector database...")
            vector_start = time.time()
            self.vector_store.insert_vectors(vectors)
            vector_time = time.time() - vector_start
            
            # Store in graph database
            self.logger.info("Storing in graph database...")
            graph_start = time.time()
            self.graph_store.store_entities(all_entities)
            
            # Extract and store relationships
            relationships = self.graph_store.extract_relationships(all_entities)
            self.graph_store.store_relationships(relationships)
            graph_time = time.time() - graph_start
            
            total_time = time.time() - start_time
            
            stats = {
                "status": "success",
                "entities_count": len(all_entities),
                "vectors_count": len(vectors),
                "relationships_count": len(relationships),
                "files_count": len(entities_by_file),
                "parse_time": parse_time,
                "embed_time": embed_time,
                "vector_time": vector_time,
                "graph_time": graph_time,
                "total_time": total_time
            }
            
            self.logger.info(f"Indexing completed successfully in {total_time:.2f}s")
            self.logger.info(f"Stats: {stats}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Indexing failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def _generate_embeddings_parallel(self, entities: List[CodeEntity], max_workers: int) -> List[VectorEntity]:
        """Generate embeddings in parallel."""
        vectors = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_entity = {
                executor.submit(self.embedding_generator.generate_entity_embedding, entity): entity
                for entity in entities
            }
            
            # Collect results
            for future in as_completed(future_to_entity):
                try:
                    vector = future.result()
                    if vector.vector:  # Only include valid vectors
                        vectors.append(vector)
                except Exception as e:
                    entity = future_to_entity[future]
                    self.logger.error(f"Failed to generate embedding for {entity.name}: {e}")
        
        return vectors
    
    def search_code(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar code entities."""
        try:
            # Generate query embedding
            query_vector = self.embedding_generator.generate_embedding(query)
            if not query_vector:
                return []
            
            # Search in vector store
            results = self.vector_store.search_similar(query_vector, top_k)
            
            # Enhance results with graph relationships
            enhanced_results = []
            for result in results:
                entity_id = result["id"]
                related = self.graph_store.query_related_entities(entity_id)
                
                enhanced_result = result.copy()
                enhanced_result["related_entities"] = related
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            graph_stats = self.graph_store.get_graph_stats()
            
            return {
                "vector_store": vector_stats,
                "graph_store": graph_stats
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.vector_store.close()
            self.graph_store.close()
            self.logger.info("Cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()