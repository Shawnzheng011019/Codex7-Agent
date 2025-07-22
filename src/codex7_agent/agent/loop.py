"""
Core Agent Loop framework implementation.
This module provides the foundational structure for the dynamic agent loop system.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging


class LoopStatus(Enum):
    """Agent loop execution status."""
    INIT = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    INTERRUPTED = "interrupted"


class MessageType(Enum):
    """Types of messages in agent communication."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_use"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"


@dataclass
class Message:
    """Represents a single message in the conversation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.USER
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    parent_uuid: Optional[str] = None


@dataclass
class Context:
    """Represents the current context for agent execution."""
    messages: List[Message] = field(default_factory=list)
    current_request: str = ""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "default"
    workspace_path: str = ""
    prevent_continuation: bool = False
    loop_count: int = 0
    max_loops: int = 100


@dataclass
class ToolCall:
    """Represents a tool call request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    is_concurrency_safe: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    tool_call_id: str = ""
    success: bool = True
    content: Any = None
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    prevent_continuation: bool = False


class AgentLoopError(Exception):
    """Base exception for agent loop errors."""
    pass


class ModelDowngradeError(AgentLoopError):
    """Raised when model needs to be downgraded."""
    pass


class ToolExecutionError(AgentLoopError):
    """Raised when tool execution fails."""
    pass


class ContextOverflowError(AgentLoopError):
    """Raised when context exceeds limits."""
    pass