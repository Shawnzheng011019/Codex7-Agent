from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CodeEntity:
    name: str
    entity_type: str
    file_path: str
    line_start: int
    line_end: int
    code_snippet: str
    docstring: Optional[str] = None
    parent: Optional[str] = None
    parameters: Optional[List[str]] = None
    return_type: Optional[str] = None
    dependencies: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []