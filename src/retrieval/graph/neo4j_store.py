import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from ..config import RetrievalConfig
from .models import Relationship


class Neo4jGraphStore:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection and constraints
        self.connect()
        self.create_constraints()

    def connect(self) -> None:
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )
        self.driver.verify_connectivity()
        self.logger.info("Connected to Neo4j")

    def create_constraints(self) -> None:
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.file_path)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)"
            ]
            for constraint in constraints:
                session.run(constraint)

    def store_entities(self, entities: List[Any]) -> None:
        with self.driver.session() as session:
            for entity in entities:
                entity_id = f"{entity.file_path}:{entity.name}:{entity.line_start}"
                session.run("""
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.entity_type = $entity_type,
                        e.file_path = $file_path,
                        e.line_start = $line_start,
                        e.line_end = $line_end,
                        e.code_snippet = $code_snippet,
                        e.docstring = $docstring,
                        e.parent = $parent,
                        e.parameters = $parameters,
                        e.return_type = $return_type,
                        e.dependencies = $dependencies
                """, {
                    "id": entity_id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "file_path": entity.file_path,
                    "line_start": entity.line_start,
                    "line_end": entity.line_end,
                    "code_snippet": entity.code_snippet,
                    "docstring": entity.docstring or "",
                    "parent": entity.parent or "",
                    "parameters": entity.parameters or [],
                    "return_type": entity.return_type or "",
                    "dependencies": entity.dependencies or []
                })

    def store_relationships(self, relationships: List[Relationship]) -> None:
        with self.driver.session() as session:
            for rel in relationships:
                session.run(f"""
                    MATCH (source:Entity {{id: $source_id}})
                    MATCH (target:Entity {{id: $target_id}})
                    MERGE (source)-[r:{rel.relationship_type}]-(target)
                    SET r += $properties
                """, {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "properties": rel.properties
                })

    def extract_relationships(self, entities: List[Any]) -> List[Relationship]:
        relationships = []
        entity_lookup = {}
        for entity in entities:
            entity_id = f"{entity.file_path}:{entity.name}:{entity.line_start}"
            entity_lookup[entity.name] = entity_id

        for entity in entities:
            entity_id = f"{entity.file_path}:{entity.name}:{entity.line_start}"
            
            if entity.parent and entity.parent in entity_lookup:
                relationships.append(Relationship(
                    source_id=entity_id,
                    target_id=entity_lookup[entity.parent],
                    relationship_type="CHILD_OF",
                    properties={"relationship_type": "hierarchical"}
                ))

            for dep in entity.dependencies:
                if dep in entity_lookup:
                    relationships.append(Relationship(
                        source_id=entity_id,
                        target_id=entity_lookup[dep],
                        relationship_type="DEPENDS_ON",
                        properties={"relationship_type": "dependency"}
                    ))

            for other_entity in entities:
                if other_entity != entity and other_entity.file_path == entity.file_path:
                    other_id = f"{other_entity.file_path}:{other_entity.name}:{other_entity.line_start}"
                    relationships.append(Relationship(
                        source_id=entity_id,
                        target_id=other_id,
                        relationship_type="IN_FILE",
                        properties={"relationship_type": "file_scope"}
                    ))
        return relationships

    def query_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {id: $id})
                RETURN e
            """, {"id": entity_id})
            record = result.single()
            return dict(record["e"]) if record else None

    def query_related_entities(self, entity_id: str, relationship_type: str = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            query = "MATCH (e:Entity {id: $id})-[r]-(related:Entity)"
            if relationship_type:
                query += f" WHERE type(r) = '{relationship_type}'"
            query += " RETURN related, type(r) as relationship_type, r"
            
            result = session.run(query, {"id": entity_id})
            entities = []
            for record in result:
                entity_data = dict(record["related"])
                entity_data["relationship_type"] = record["relationship_type"]
                entity_data["relationship_properties"] = dict(record["r"])
                entities.append(entity_data)
            return entities

    def get_graph_stats(self) -> Dict[str, Any]:
        with self.driver.session() as session:
            entities_result = session.run("""
                MATCH (e:Entity)
                RETURN e.entity_type as type, count(e) as count
            """)
            entities_by_type = {record["type"]: record["count"] for record in entities_result}

            relationships_result = session.run("""
                MATCH ()-[r]-()
                RETURN type(r) as type, count(r) as count
            """)
            relationships_by_type = {record["type"]: record["count"] for record in relationships_result}

            total_entities = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
            total_relationships = session.run("MATCH ()-[r]-() RETURN count(r) as count").single()["count"]

            return {
                "entities_by_type": entities_by_type,
                "relationships_by_type": relationships_by_type,
                "total_entities": total_entities,
                "total_relationships": total_relationships
            }

    def close(self) -> None:
        if self.driver:
            self.driver.close()