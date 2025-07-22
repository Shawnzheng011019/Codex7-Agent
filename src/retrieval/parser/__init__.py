"""
Parser module for AST-based code analysis.

This module provides functionality for parsing code into structured entities.
"""

from .entities import CodeEntity
from .ast_parser import ASTParser
from .visitor import EntityVisitor

__all__ = ["CodeEntity", "ASTParser", "EntityVisitor"]