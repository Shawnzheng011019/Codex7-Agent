"""
Tool adapter to bridge existing Tool implementations with the ToolRegistry system.
"""

import logging
from typing import Dict, Any, Optional, Tuple

from .base import Tool, ToolExecResult, ToolCallArguments
from .tool_manager import ToolExecutor, ToolResult


class ToolAdapter(ToolExecutor):
    """Adapter to convert existing Tool implementations to ToolExecutor interface."""
    
    def __init__(self, tool: Tool):
        super().__init__(tool.name)
        self.tool = tool
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the adapted tool."""
        try:
            # Convert parameters to the format expected by the tool
            tool_args: ToolCallArguments = parameters
            
            # Execute the tool
            result: ToolExecResult = await self.tool.execute(tool_args)
            
            # Convert result to ToolResult
            return ToolResult(
                success=result.error_code == 0,
                content=result.output,
                error_message=result.error,
                # The following fields will be filled by the caller
                tool_call_id=None,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error_message=str(e)
            )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate tool parameters."""
        # This is a simplified validation - in a real implementation, we'd want to
        # validate against the schema defined in the tool
        return True, ""