import ast
import os
from typing import List, Dict
from .entities import CodeEntity


class EntityVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, source: str):
        self.file_path = file_path
        self.source = source.splitlines()
        self.entities = []
        self.current_class = None
        self.imports = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            for alias in node.names:
                self.imports.append(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_entity = CodeEntity(
            name=node.name,
            entity_type="class",
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            code_snippet=self._get_code_snippet(node),
            docstring=ast.get_docstring(node),
            dependencies=self._extract_dependencies(node)
        )
        self.entities.append(class_entity)

        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node, "async_function")

    def _process_function(self, node: ast.AST, func_type: str) -> None:
        params = []
        if hasattr(node, 'args') and node.args:
            for arg in node.args.args:
                if hasattr(arg, 'arg') and arg.arg:
                    params.append(arg.arg)

        return_type = None
        if hasattr(node, 'returns') and node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

        func_entity = CodeEntity(
            name=node.name,
            entity_type=func_type,
            file_path=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            code_snippet=self._get_code_snippet(node),
            docstring=ast.get_docstring(node),
            parent=self.current_class,
            parameters=params,
            return_type=return_type,
            dependencies=self._extract_dependencies(node)
        )
        self.entities.append(func_entity)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_entity = CodeEntity(
                    name=target.id,
                    entity_type="variable",
                    file_path=self.file_path,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    code_snippet=self._get_code_snippet(node),
                    parent=self.current_class,
                    dependencies=self._extract_dependencies(node)
                )
                self.entities.append(var_entity)
        self.generic_visit(node)

    def _get_code_snippet(self, node: ast.AST) -> str:
        try:
            start_line = node.lineno - 1
            end_line = node.end_lineno or node.lineno
            return "\n".join(self.source[start_line:end_line])
        except (IndexError, AttributeError):
            return ""

    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in self.imports:
                dependencies.append(child.id)
            elif isinstance(child, ast.Attribute):
                attr_chain = self._get_attribute_chain(child)
                if attr_chain and attr_chain in self.imports:
                    dependencies.append(attr_chain)
        return dependencies

    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return ""