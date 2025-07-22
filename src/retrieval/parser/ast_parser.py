import ast
import os
import logging
from typing import List, Dict
from .entities import CodeEntity
from .visitor import EntityVisitor


class ASTParser:
    def __init__(self, file_extensions: List[str] = None):
        self.file_extensions = file_extensions or [".py"]
        self.logger = logging.getLogger(__name__)

    def parse_file(self, file_path: str) -> List[CodeEntity]:
        if not any(file_path.endswith(ext) for ext in self.file_extensions):
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=file_path)
            visitor = EntityVisitor(file_path, content)
            visitor.visit(tree)
            return visitor.entities

        except Exception as e:
            self.logger.error(f"Failed to parse {file_path}: {e}")
            return []

    def parse_directory(self, directory: str) -> Dict[str, List[CodeEntity]]:
        results = {}
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".") 
                      and d not in ["__pycache__", "node_modules", ".git", ".venv", "venv"]]

            for file in files:
                if any(file.endswith(ext) for ext in self.file_extensions):
                    file_path = os.path.join(root, file)
                    entities = self.parse_file(file_path)
                    if entities:
                        results[file_path] = entities
        return results