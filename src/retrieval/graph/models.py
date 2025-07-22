from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Relationship:
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]