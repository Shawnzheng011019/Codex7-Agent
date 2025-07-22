"""
Tool management and orchestration system.
Manages tool registration, execution, and intelligent scheduling.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import inspect

from ..agent.loop import ToolCall, ToolResult


class ToolCategory(Enum):
    """Categories of tools for organization."""
    FILESYSTEM = "filesystem"
    CODE = "code"
    SEARCH = "search"
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    UTIL = "util"


@dataclass
class ToolDefinition:
    """Definition of a tool including metadata and execution info."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    is_concurrency_safe: bool
    priority: int
    timeout: int = 30
    retry_count: int = 1
    requires_confirmation: bool = False


class ToolExecutor:
    """Base class for tool executors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with given parameters."""
        raise NotImplementedError
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate tool parameters."""
        return True, ""
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {}


class FileSystemExecutor(ToolExecutor):
    """Executor for filesystem-related tools."""
    
    def __init__(self):
        super().__init__("filesystem")
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute filesystem operations."""
        operation = parameters.get("operation")
        
        try:
            if operation == "read":
                return await self._read_file(parameters)
            elif operation == "write":
                return await self._write_file(parameters)
            elif operation == "list":
                return await self._list_directory(parameters)
            else:
                return ToolResult(
                    success=False,
                    error_message=f"Unknown operation: {operation}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    async def _read_file(self, parameters: Dict[str, Any]) -> ToolResult:
        """Read file content."""
        file_path = parameters.get("file_path")
        if not file_path:
            return ToolResult(
                success=False,
                error_message="file_path parameter required"
            )
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            return ToolResult(
                success=True,
                content={
                    "file_path": file_path,
                    "content": content,
                    "size": len(content)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to read {file_path}: {str(e)}"
            )
    
    async def _write_file(self, parameters: Dict[str, Any]) -> ToolResult:
        """Write content to file."""
        file_path = parameters.get("file_path")
        content = parameters.get("content")
        
        if not file_path or content is None:
            return ToolResult(
                success=False,
                error_message="file_path and content parameters required"
            )
        
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                content={
                    "file_path": file_path,
                    "bytes_written": len(content)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=f"Failed to write {file_path}: {str(e)}"
            )
    
    async def _list_directory(self, parameters: Dict[str, Any]) -> ToolResult:
        """List directory contents."""
        directory = parameters.get("directory", ".")
        
        try:
            import os
            items = os.listdir(directory)
            
            files = []
            directories = []
            
            for item in items:
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    files.append(item)
                elif os.path.isdir(item_path):
                    directories.append(item)
            
            return ToolResult(
                success=True,
                content={
                    "directory": directory,
                    "files": files,
                    "directories": directories,
                    "total_items": len(items)
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )


class CodeExecutor(ToolExecutor):
    """Executor for code-related tools."""
    
    def __init__(self):
        super().__init__("code")
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute code operations."""
        operation = parameters.get("operation")
        
        try:
            if operation == "analyze":
                return await self._analyze_code(parameters)
            elif operation == "format":
                return await self._format_code(parameters)
            else:
                return ToolResult(
                    success=False,
                    error_message=f"Unknown operation: {operation}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    async def _analyze_code(self, parameters: Dict[str, Any]) -> ToolResult:
        """Analyze code structure."""
        file_path = parameters.get("file_path")
        
        if not file_path:
            return ToolResult(
                success=False,
                error_message="file_path parameter required"
            )
        
        try:
            # TODO: Implement actual code analysis
            return ToolResult(
                success=True,
                content={
                    "file_path": file_path,
                    "lines": 0,
                    "classes": 0,
                    "functions": 0,
                    "analysis": "Code analysis not yet implemented"
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    async def _format_code(self, parameters: Dict[str, Any]) -> ToolResult:
        """Format code."""
        code = parameters.get("code")
        
        if not code:
            return ToolResult(
                success=False,
                error_message="code parameter required"
            )
        
        try:
            # TODO: Implement actual code formatting
            return ToolResult(
                success=True,
                content={
                    "formatted_code": code,
                    "changes_made": 0
                }
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )


class ToolRegistry:
    """Registry for managing all available tools."""
    
    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}
        self.executors: Dict[str, ToolExecutor] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_tool(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        parameters: Dict[str, Any],
        executor: ToolExecutor,
        is_concurrency_safe: bool = False,
        priority: int = 5,
        **kwargs
    ) -> None:
        """Register a new tool."""
        tool_def = ToolDefinition(
            name=name,
            description=description,
            category=category,
            parameters=parameters,
            is_concurrency_safe=is_concurrency_safe,
            priority=priority,
            **kwargs
        )
        
        self.tools[name] = tool_def
        self.executors[name] = executor
        self.logger.info(f"Registered tool: {name}")
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get tool definition by name."""
        return self.tools.get(name)
    
    def get_executor(self, name: str) -> Optional[ToolExecutor]:
        """Get tool executor by name."""
        return self.executors.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolDefinition]:
        """List all tools, optionally filtered by category."""
        tools = list(self.tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools
    
    def get_concurrency_info(self, tool_name: str) -> Tuple[bool, int]:
        """Get concurrency safety and priority for a tool."""
        tool = self.get_tool(tool_name)
        if tool:
            return tool.is_concurrency_safe, tool.priority
        return False, 5  # Default values


class ToolOrchestrator:
    """Orchestrates tool execution with intelligent scheduling."""
    
    def __init__(self, registry: ToolRegistry, max_concurrency: int = 10):
        self.registry = registry
        self.max_concurrency = max_concurrency
        self.logger = logging.getLogger(__name__)
    
    async def execute_tools(
        self, 
        tool_calls: List[ToolCall], 
        context: Dict[str, Any] = None
    ) -> List[ToolResult]:
        """
        Execute tools with intelligent scheduling and concurrency control.
        
        Args:
            tool_calls: List of tool calls to execute
            context: Additional context for tool execution
            
        Returns:
            List of tool execution results
        """
        if context is None:
            context = {}
        
        self.logger.info(f"Executing {len(tool_calls)} tools")
        
        # Categorize tools by concurrency safety
        concurrent_tools = []
        sequential_tools = []
        
        for call in tool_calls:
            is_safe, priority = self.registry.get_concurrency_info(call.tool_name)
            call.is_concurrency_safe = is_safe
            call.priority = priority
            
            if is_safe:
                concurrent_tools.append(call)
            else:
                sequential_tools.append(call)
        
        # Execute concurrent tools
        concurrent_results = []
        if concurrent_tools:
            concurrent_results = await self._execute_concurrent_tools(concurrent_tools, context)
        
        # Execute sequential tools in priority order
        sequential_results = []
        if sequential_tools:
            sequential_results = await self._execute_sequential_tools(sequential_tools, context)
        
        # Combine results
        all_results = concurrent_results + sequential_results
        
        # Sort by original order (tool_call id)
        all_results.sort(key=lambda x: next(
            call.id for call in tool_calls if call.id == x.tool_call_id
        ))
        
        return all_results
    
    async def _execute_concurrent_tools(
        self, 
        tool_calls: List[ToolCall], 
        context: Dict[str, Any]
    ) -> List[ToolResult]:
        """Execute concurrent-safe tools in parallel."""
        semaphore = asyncio.Semaphore(self.max_concurrency)
        
        async def execute_single_tool(tool_call: ToolCall) -> ToolResult:
            async with semaphore:
                return await self._execute_single_tool(tool_call, context)
        
        tasks = [execute_single_tool(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for result, tool_call in zip(results, tool_calls):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _execute_sequential_tools(
        self, 
        tool_calls: List[ToolCall], 
        context: Dict[str, Any]
    ) -> List[ToolResult]:
        """Execute sequential tools in priority order."""
        # Sort by priority (higher priority first)
        sorted_calls = sorted(tool_calls, key=lambda x: x.priority, reverse=True)
        
        results = []
        for tool_call in sorted_calls:
            try:
                result = await self._execute_single_tool(tool_call, context)
                results.append(result)
                
                # Check for prevent continuation
                if result.prevent_continuation:
                    break
                    
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error_message=str(e)
                ))
                break
        
        return results
    
    async def _execute_single_tool(
        self, 
        tool_call: ToolCall, 
        context: Dict[str, Any]
    ) -> ToolResult:
        """Execute a single tool call."""
        executor = self.registry.get_executor(tool_call.tool_name)
        if not executor:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error_message=f"Unknown tool: {tool_call.tool_name}"
            )
        
        # Validate parameters
        is_valid, error_msg = executor.validate_parameters(tool_call.parameters)
        if not is_valid:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error_message=error_msg
            )
        
        # Execute tool
        result = await executor.execute(tool_call.parameters)
        result.tool_call_id = tool_call.id
        
        return result
    
    def calculate_priority(
        self, 
        tool_call: ToolCall, 
        context: Dict[str, Any]
    ) -> int:
        """Calculate tool priority based on context and preferences."""
        base_priority = tool_call.priority
        
        # TODO: Implement sophisticated priority calculation
        # - Context relevance bonus
        # - User preference bonus
        # - Tool dependency bonus
        
        return base_priority
    
    def get_tools_for_task(self, task_type: str) -> List[ToolDefinition]:
        """Get appropriate tools for a given task type."""
        # TODO: Implement intelligent tool selection based on task type
        return self.registry.list_tools()


# Factory function to create default registry with built-in tools
def create_default_registry() -> ToolRegistry:
    """Create a registry with default tools."""
    registry = ToolRegistry()
    
    # Register filesystem tools
    filesystem_executor = FileSystemExecutor()
    registry.register_tool(
        name="read_file",
        description="Read content from a file",
        category=ToolCategory.FILESYSTEM,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["file_path"]
        },
        executor=filesystem_executor,
        is_concurrency_safe=True,
        priority=5
    )
    
    registry.register_tool(
        name="write_file",
        description="Write content to a file",
        category=ToolCategory.FILESYSTEM,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "required": ["file_path", "content"]
        },
        executor=filesystem_executor,
        is_concurrency_safe=False,
        priority=7,
        requires_confirmation=True
    )
    
    registry.register_tool(
        name="list_directory",
        description="List contents of a directory",
        category=ToolCategory.FILESYSTEM,
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory path", "default": "."}
            }
        },
        executor=filesystem_executor,
        is_concurrency_safe=True,
        priority=3
    )
    
    # Register code tools
    code_executor = CodeExecutor()
    registry.register_tool(
        name="analyze_code",
        description="Analyze code structure and complexity",
        category=ToolCategory.CODE,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to code file"}
            },
            "required": ["file_path"]
        },
        executor=code_executor,
        is_concurrency_safe=True,
        priority=6
    )
    
    return registry