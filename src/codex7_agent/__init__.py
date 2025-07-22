"""
Claude Code Agent - Main package initialization.
Provides a complete agent loop framework for software engineering tasks.
"""

from .agent.orchestrator import AgentLoopOrchestrator, AgentConfig, create_agent

__version__ = "1.0.0"
__all__ = ["AgentLoopOrchestrator", "AgentConfig", "create_agent"]