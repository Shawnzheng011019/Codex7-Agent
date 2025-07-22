import logging
from typing import List, Dict, Any
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema,
    DataType, utility
)
from ..config import RetrievalConfig
from .models import VectorEntity


class MilvusVectorStore:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.collection = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection and collection
        self.connect()
        self.create_collection()

    def connect(self) -> None:
        connections.connect(alias="default", uri=self.config.milvus_uri)
        self.logger.info("Connected to Milvus")

    def create_collection(self) -> None:
        if utility.has_collection(self.config.milvus_collection_name):
            utility.drop_collection(self.config.milvus_collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.milvus_dimension),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="code_snippet", dtype=DataType.VARCHAR, max_length=8192)
        ]

        schema = CollectionSchema(fields, "Code entity vectors")
        self.collection = Collection(name=self.config.milvus_collection_name, schema=schema)

        index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        self.collection.create_index("vector", index_params)
        self.logger.info(f"Created collection: {self.config.milvus_collection_name}")

    def insert_vectors(self, vectors: List[VectorEntity]) -> None:
        if not vectors:
            return

        data = [
            [v.entity_id for v in vectors],
            [v.vector for v in vectors],
            [v.file_path for v in vectors],
            [v.entity_type for v in vectors],
            [v.name for v in vectors],
            [v.metadata for v in vectors],
            [v.code_snippet for v in vectors]
        ]

        self.collection.insert(data)
        self.collection.flush()
        self.logger.info(f"Inserted {len(vectors)} vectors")

    def search_similar(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        self.collection.load()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["file_path", "entity_type", "name", "metadata", "code_snippet"]
        )

        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "file_path": hit.entity.get("file_path"),
                    "entity_type": hit.entity.get("entity_type"),
                    "name": hit.entity.get("name"),
                    "metadata": hit.entity.get("metadata"),
                    "code_snippet": hit.entity.get("code_snippet")
                })
        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get comprehensive collection statistics."""
        try:
            self.collection.load()
            total_entities = self.collection.num_entities
            
            # Get collection schema info
            schema = self.collection.schema
            vector_dim = None
            for field in schema.fields:
                if field.name == "vector" and field.dtype == DataType.FLOAT_VECTOR:
                    vector_dim = field.params.get('dim')
                    break
            
            return {
                "collection_name": self.config.milvus_collection_name,
                "total_entities": total_entities,
                "vector_dimension": vector_dim,
                "status": "loaded" if self.collection.is_empty is False else "empty"
            }
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    def close(self) -> None:
        from pymilvus import connections
        connections.disconnect("default")