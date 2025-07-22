from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class VectorEntity:
    entity_id: str
    file_path: str
    entity_type: str
    name: str
    vector: List[float]
    metadata: Dict[str, Any]
    code_snippet: str