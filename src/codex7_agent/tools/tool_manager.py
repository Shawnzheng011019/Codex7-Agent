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
from .base import Tool, ToolExecResult, ToolCallArguments


class ToolCategory(Enum):
    """7-category classification system for Agent Loop."""
    FILESYSTEM = "filesystem"      # File operations (read, write, list)
    CODE = "code"                  # Code analysis and manipulation
    SEARCH = "search"              # Search and discovery operations
    SYSTEM = "system"              # System-level operations (bash, env)
    NETWORK = "network"            # Network operations (web fetch)
    DATABASE = "database"          # Database operations
    UTIL = "util"                  # Utility operations (tasks, etc)


@dataclass
class ToolDefinition:
    """Definition of a tool including metadata and execution info for Agent Loop."""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any]
    is_concurrency_safe: bool
    priority: int
    timeout: int = 30
    retry_count: int = 1
    requires_confirmation: bool = False
    complexity_score: int = 5  # 1-10 scale for complexity assessment
    force_pattern: Optional[str] = None  # Special pattern to force tool usage


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


class SystemExecutor(ToolExecutor):
    """Executor for system-level tools including bash operations."""
    
    def __init__(self):
        super().__init__("system")
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute system operations."""
        operation = parameters.get("operation", "bash")
        
        try:
            if operation == "bash":
                return await self._execute_bash(parameters)
            elif operation == "env":
                return await self._get_env(parameters)
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
    
    async def _execute_bash(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute bash commands."""
        command = parameters.get("command")
        
        if not command:
            return ToolResult(
                success=False,
                error_message="command parameter required"
            )
        
        try:
            import subprocess
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                timeout=parameters.get("timeout", 30)
            )
            
            return ToolResult(
                success=result.returncode == 0,
                content={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                },
                error_message=result.stderr if result.returncode != 0 else None
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error_message="Command timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    async def _get_env(self, parameters: Dict[str, Any]) -> ToolResult:
        """Get environment variables."""
        var_name = parameters.get("var_name")
        
        try:
            import os
            if var_name:
                value = os.environ.get(var_name)
                return ToolResult(
                    success=True,
                    content={"var_name": var_name, "value": value}
                )
            else:
                return ToolResult(
                    success=True,
                    content=dict(os.environ)
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
            import ast
            with open(file_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Basic analysis
            class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            import_count = len([node for node in ast.walk(tree) if isinstance(node, ast.Import)])
            
            return ToolResult(
                success=True,
                content={
                    "file_path": file_path,
                    "lines": len(content.splitlines()),
                    "classes": class_count,
                    "functions": function_count,
                    "imports": import_count,
                    "analysis": f"File contains {class_count} classes, {function_count} functions, {import_count} imports"
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
            # Basic formatting (indentation and spacing)
            import re
            formatted = re.sub(r'\s+\n', '\n', code)
            formatted = re.sub(r'\n{3,}', '\n\n', formatted)
            
            changes_made = 1 if formatted != code else 0
            
            return ToolResult(
                success=True,
                content={
                    "formatted_code": formatted,
                    "changes_made": changes_made
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
        self._registered_tools: List[Tool] = []  # Store actual Tool instances
    
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
    
    def register_adapted_tool(
        self,
        tool: Tool,
        category: ToolCategory,
        is_concurrency_safe: bool = False,
        priority: int = 5,
        **kwargs
    ) -> None:
        """Register an existing Tool implementation with adapter."""
        # Import here to avoid circular imports
        from .tool_adapter import ToolAdapter
        
        # Store the tool for later access
        self._registered_tools.append(tool)
        
        # Create adapter
        adapter = ToolAdapter(tool)
        
        # Register with the registry
        self.tools[tool.name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            category=category,
            parameters=tool.get_input_schema(),
            is_concurrency_safe=is_concurrency_safe,
            priority=priority,
            **kwargs
        )
        self.executors[tool.name] = adapter
        self.logger.info(f"Registered adapted tool: {tool.name}")
    
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
    """Agent Loop compatible orchestrator with intelligent scheduling."""
    
    def __init__(self, registry: ToolRegistry, max_concurrency: int = 10):
        self.registry = registry
        self.max_concurrency = max_concurrency
        self.logger = logging.getLogger(__name__)
        self.interruption_flag = False
        self.complexity_patterns = self._load_complexity_patterns()
    
    def _load_complexity_patterns(self) -> Dict[str, int]:
        """Load patterns for complexity assessment."""
        return {
            r'\b(large|complex|system|architecture)\b': 8,
            r'\b(simple|small|basic|easy)\b': 2,
            r'\b(optimization|performance|refactor)\b': 7,
            r'\b(bug|fix|error|issue)\b': 6,
            r'\b(search|find|explore|research)\b': 4,
            r'\b(implement|create|build|develop)\b': 5,
            r'\b(test|validate|verify|check)\b': 3
        }
    
    async def execute_tools(
        self, 
        tool_calls: List[ToolCall], 
        context: Dict[str, Any] = None
    ) -> List[ToolResult]:
        """
        Execute tools with Agent Loop intelligent scheduling.
        
        Args:
            tool_calls: List of tool calls to execute
            context: Additional context including task description for complexity assessment
            
        Returns:
            List of tool execution results
        """
        if context is None:
            context = {}
        
        # Check for force tool selection patterns
        forced_tools = self._check_force_patterns(context.get("task_description", ""))
        if forced_tools:
            tool_calls = self._apply_force_patterns(tool_calls, forced_tools)
        
        # Assess complexity and determine execution strategy
        complexity = self._assess_complexity(context.get("task_description", ""))
        if complexity >= 7 and len(tool_calls) > 1:
            return await self._execute_high_complexity_tools(tool_calls, context, complexity)
        
        self.logger.info(f"Executing {len(tool_calls)} tools (complexity: {complexity})")
        
        # Categorize tools by concurrency safety and micro-optimize
        concurrent_tools, sequential_tools = self._categorize_and_optimize(tool_calls, context)
        
        # Execute with real-time interruption checking
        all_results = await self._execute_with_interrupt_checking(
            concurrent_tools, sequential_tools, context
        )
        
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
    
    def _check_force_patterns(self, task_description: str) -> List[str]:
        """Check for special patterns that force tool selection."""
        force_patterns = {
            r'^#read\s+': ['read_file'],
            r'^#write\s+': ['write_file'],
            r'^#list\s+': ['list_directory'],
            r'^#search\s+': ['grep', 'find'],
            r'^#analyze\s+': ['analyze_code'],
            r'^#edit\s+': ['edit_file'],
            r'^#task\s+': ['task'],
            r'^#bash\s+': ['bash']
        }
        
        forced_tools = []
        import re
        for pattern, tools in force_patterns.items():
            if re.match(pattern, task_description, re.I):
                forced_tools.extend(tools)
        
        return forced_tools
    
    def _apply_force_patterns(self, tool_calls: List[ToolCall], forced_tools: List[str]) -> List[ToolCall]:
        """Apply force patterns to prioritize specific tools."""
        prioritized_calls = []
        remaining_calls = []
        
        for call in tool_calls:
            if call.tool_name in forced_tools:
                call.priority = 10  # Highest priority
                prioritized_calls.append(call)
            else:
                remaining_calls.append(call)
        
        return prioritized_calls + remaining_calls
    
    def _assess_complexity(self, task_description: str) -> int:
        """Intelligent complexity assessment using 2025 patterns."""
        if not task_description:
            return 5
        
        complexity = 5  # Default medium complexity
        import re
        
        for pattern, score in self.complexity_patterns.items():
            if re.search(pattern, task_description, re.I):
                complexity = max(complexity, score)
        
        # Additional complexity factors
        word_count = len(task_description.split())
        if word_count > 50:
            complexity += 1
        if word_count > 100:
            complexity += 2
        
        return min(10, complexity)
    
    def _categorize_and_optimize(self, tool_calls: List[ToolCall], context: Dict[str, Any]) -> Tuple[List[ToolCall], List[ToolCall]]:
        """Categorize tools by concurrency safety and apply micro-optimizations."""
        concurrent_tools = []
        sequential_tools = []
        
        for call in tool_calls:
            is_safe, priority = self.registry.get_concurrency_info(call.tool_name)
            tool_def = self.registry.get_tool(call.tool_name)
            
            # Micro-optimization: adjust priority based on context
            adjusted_priority = self._micro_optimize_priority(call, tool_def, context)
            
            call.is_concurrency_safe = is_safe
            call.priority = adjusted_priority
            
            if is_safe:
                concurrent_tools.append(call)
            else:
                sequential_tools.append(call)
        
        return concurrent_tools, sequential_tools
    
    def _micro_optimize_priority(self, tool_call: ToolCall, tool_def: Optional[ToolDefinition], context: Dict[str, Any]) -> int:
        """Micro-optimize tool priority based on context."""
        if not tool_def:
            return tool_call.priority
        
        base_priority = tool_def.priority
        
        # Context relevance bonus
        task_description = context.get("task_description", "").lower()
        tool_description = tool_def.description.lower()
        
        # Direct relevance bonus
        if any(word in task_description for word in tool_description.split()):
            base_priority += 2
        
        # Category relevance bonus
        task_category = self._determine_task_category(task_description)
        if str(tool_def.category.value) in task_description:
            base_priority += 1
        
        # User preference bonus (from context)
        preferences = context.get("tool_preferences", {})
        if tool_call.tool_name in preferences:
            base_priority += preferences[tool_call.tool_name]
        
        return min(10, max(1, base_priority))
    
    def _determine_task_category(self, task_description: str) -> str:
        """Determine the primary category for a task."""
        description = task_description.lower()
        
        if any(word in description for word in ['file', 'read', 'write', 'directory']):
            return "filesystem"
        elif any(word in description for word in ['code', 'analyze', 'refactor', 'function']):
            return "code"
        elif any(word in description for word in ['search', 'find', 'grep']):
            return "search"
        elif any(word in description for word in ['bash', 'command', 'system', 'env']):
            return "system"
        elif any(word in description for word in ['url', 'web', 'http', 'fetch']):
            return "network"
        elif any(word in description for word in ['database', 'sql', 'query']):
            return "database"
        else:
            return "util"
    
    async def _execute_high_complexity_tools(
        self, 
        tool_calls: List[ToolCall], 
        context: Dict[str, Any], 
        complexity: int
    ) -> List[ToolResult]:
        """Execute tools using Task tool for high complexity scenarios."""
        # Delegate to task tool for complex scenarios
        task_description = context.get("task_description", "Complex task execution")
        
        # Group tools by category for better organization
        categorized_tools = {}
        for call in tool_calls:
            tool_def = self.registry.get_tool(call.tool_name)
            if tool_def:
                category = str(tool_def.category.value)
                if category not in categorized_tools:
                    categorized_tools[category] = []
                categorized_tools[category].append(call.tool_name)
        
        # Use task tool for orchestration
        from .task_tool import TaskTool
        task_tool = TaskTool(
            registry=self.registry,
            context_manager=None,  # Will be provided
            intent_recognizer=None  # Will be provided
        )
        
        result = await task_tool.execute_task(
            description=task_description,
            tools=[call.tool_name for call in tool_calls],
            max_agents=min(3, len(tool_calls)),
            context={"complexity": complexity, "categorized_tools": categorized_tools}
        )
        
        # Convert task result to tool results
        return [ToolResult(
            success=result["success"],
            content=result,
            error_message=result.get("error", "")
        )]
    
    async def _execute_with_interrupt_checking(
        self, 
        concurrent_tools: List[ToolCall], 
        sequential_tools: List[ToolCall], 
        context: Dict[str, Any]
    ) -> List[ToolResult]:
        """Execute tools with real-time interruption checking."""
        all_results = []
        
        # Execute concurrent tools with interruption checking
        if concurrent_tools:
            semaphore = asyncio.Semaphore(self.max_concurrency)
            
            async def execute_with_interrupt(tool_call: ToolCall) -> ToolResult:
                if self.interruption_flag:
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        success=False,
                        error_message="Execution interrupted by user"
                    )
                
                async with semaphore:
                    return await self._execute_single_tool(tool_call, context)
            
            tasks = [execute_with_interrupt(call) for call in concurrent_tools]
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            for result, tool_call in zip(concurrent_results, concurrent_tools):
                if isinstance(result, Exception):
                    all_results.append(ToolResult(
                        tool_call_id=tool_call.id,
                        success=False,
                        error_message=str(result)
                    ))
                else:
                    all_results.append(result)
        
        # Execute sequential tools with interruption checking
        if sequential_tools:
            # Sort by priority (higher priority first)
            sorted_sequential = sorted(sequential_tools, key=lambda x: x.priority, reverse=True)
            
            for tool_call in sorted_sequential:
                if self.interruption_flag:
                    all_results.append(ToolResult(
                        tool_call_id=tool_call.id,
                        success=False,
                        error_message="Execution interrupted by user"
                    ))
                    break
                
                try:
                    result = await self._execute_single_tool(tool_call, context)
                    all_results.append(result)
                    
                    # Check for prevent continuation
                    if result.prevent_continuation:
                        break
                        
                except Exception as e:
                    all_results.append(ToolResult(
                        tool_call_id=tool_call.id,
                        success=False,
                        error_message=str(e)
                    ))
                    break
        
        return all_results
    
    def interrupt_execution(self) -> None:
        """Signal to interrupt current execution."""
        self.interruption_flag = True
        self.logger.info("Execution interruption requested")
    
    def reset_interruption(self) -> None:
        """Reset interruption flag."""
        self.interruption_flag = False
    
    def calculate_priority(
        self, 
        tool_call: ToolCall, 
        context: Dict[str, Any]
    ) -> int:
        """Calculate tool priority based on context and preferences."""
        return self._micro_optimize_priority(tool_call, self.registry.get_tool(tool_call.tool_name), context)
    
    def get_tools_for_task(self, task_type: str) -> List[ToolDefinition]:
        """Get appropriate tools for a given task type using intelligent selection."""
        all_tools = self.registry.list_tools()
        
        # Filter by task type and complexity
        relevant_tools = []
        task_category = self._determine_task_category(task_type)
        
        for tool in all_tools:
            if str(tool.category.value) == task_category:
                relevant_tools.append(tool)
            elif task_category == "util" and tool.complexity_score <= 5:
                relevant_tools.append(tool)
        
        # Sort by relevance and priority
        return sorted(relevant_tools, key=lambda t: (t.complexity_score, -t.priority))


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
        priority=5,
        complexity_score=2
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
        complexity_score=4,
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
        priority=3,
        complexity_score=1
    )
    
    # Register system tools
    system_executor = SystemExecutor()
    registry.register_tool(
        name="bash",
        description="Execute bash commands",
        category=ToolCategory.SYSTEM,
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {"type": "integer", "description": "Command timeout in seconds", "default": 30}
            },
            "required": ["command"]
        },
        executor=system_executor,
        is_concurrency_safe=False,
        priority=6,
        complexity_score=4,
        requires_confirmation=True
    )
    
    registry.register_tool(
        name="env",
        description="Get environment variables",
        category=ToolCategory.SYSTEM,
        parameters={
            "type": "object",
            "properties": {
                "var_name": {"type": "string", "description": "Environment variable name (optional)"}
            }
        },
        executor=system_executor,
        is_concurrency_safe=True,
        priority=3,
        complexity_score=1
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
        priority=6,
        complexity_score=6
    )
    
    # Additional tools for complete Agent Loop compatibility
    registry.register_tool(
        name="edit_file",
        description="Edit file content with string replacements",
        category=ToolCategory.FILESYSTEM,
        parameters={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "old_string": {"type": "string", "description": "Text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "default": False}
            },
            "required": ["file_path", "old_string", "new_string"]
        },
        executor=filesystem_executor,
        is_concurrency_safe=False,
        priority=8,
        complexity_score=5,
        requires_confirmation=True
    )
    
    registry.register_tool(
        name="grep",
        description="Search for patterns in files",
        category=ToolCategory.SEARCH,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern"},
                "path": {"type": "string", "description": "Search path", "default": "."},
                "file_pattern": {"type": "string", "description": "File pattern to search in"}
            },
            "required": ["pattern"]
        },
        executor=filesystem_executor,
        is_concurrency_safe=True,
        priority=4,
        complexity_score=3
    )
    
    registry.register_tool(
        name="find",
        description="Find files by name or pattern",
        category=ToolCategory.SEARCH,
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "File name pattern"},
                "path": {"type": "string", "description": "Search path", "default": "."}
            },
            "required": ["pattern"]
        },
        executor=filesystem_executor,
        is_concurrency_safe=True,
        priority=4,
        complexity_score=2
    )
    
    registry.register_tool(
        name="task",
        description="Execute complex tasks with SubAgent orchestration",
        category=ToolCategory.UTIL,
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Task description"},
                "tools": {"type": "array", "items": {"type": "string"}, "description": "Tools to use"},
                "max_agents": {"type": "integer", "default": 3},
                "context": {"type": "object", "description": "Additional context"}
            },
            "required": ["description"]
        },
        executor=None,  # Special handling in orchestrator
        is_concurrency_safe=False,
        priority=9,
        complexity_score=8
    )
    
    return registry