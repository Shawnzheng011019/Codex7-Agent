"""
Main Agent Loop Orchestrator.
Implements the dynamic Agent Loop with all integrated components.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import uuid

from .loop import AgentLoopError, Context, Message, MessageType, LoopStatus, ToolCall
from ..utils.intent_recognition import IntentRecognizer, IntentRecognitionResult
from .context_manager import ContextManager
from ..tools.tool_manager import ToolRegistry, ToolOrchestrator, create_default_registry
from ..tools.task_tool import TaskTool
from .state_manager import StateManager
from ..prompt.prompt_generator import PromptGenerator
from ..config import get_config_manager


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""
    max_loops: int = 100
    max_context_size: int = 50
    workspace_path: str = "."
    user_id: str = "default"
    enable_task_tool: bool = True
    enable_cache: bool = True
    log_level: str = "INFO"
    config_path: Optional[str] = None
    llm_config: Dict[str, Any] = field(default_factory=dict)


class AgentLoopOrchestrator:
    """Main orchestrator for the dynamic agent loop."""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config_manager = get_config_manager()
        if self.config.config_path:
            self.config_manager = get_config_manager()(self.config.config_path)
        
        # Apply configuration overrides
        self.llm_config = self.config.llm_config or self.config_manager.get_llm_config()
        self.embedding_config = self.config_manager.get_embedding_config()
        
        # Setup logging based on configuration
        log_level = self.llm_config.get('logging', {}).get('level', self.config.log_level)
        logging.basicConfig(level=getattr(logging, log_level))
        
        # Initialize core components
        self.context_manager = ContextManager()
        self.state_manager = StateManager()
        self.prompt_generator = PromptGenerator()
        self.intent_recognizer = IntentRecognizer(self.llm_config)
        self.registry = create_default_registry()
        self.tool_orchestrator = ToolOrchestrator(self.registry)
        self.task_tool = TaskTool(
            self.registry, 
            self.context_manager, 
            self.intent_recognizer
        )
        
        # Current loop state
        self.current_context = Context(
            workspace_path=self.config.workspace_path,
            user_id=self.config.user_id,
            max_loops=self.config.max_loops
        )
        self.status = LoopStatus.INIT
        
        # Event listeners
        self._listeners: List[Callable] = []
        
        # Log configuration loaded
        self.logger.info("Agent initialized with LLM configuration")
    
    async def start_loop(self, initial_message: str = "") -> Dict[str, Any]:
        """Start the agent loop with optional initial message."""
        self.logger.info("Starting agent loop")
        self.status = LoopStatus.RUNNING
        
        if initial_message:
            return await self.process_message(initial_message)
        
        return {"status": "ready", "message": "Agent loop initialized"}
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a single message through the agent loop.
        
        Args:
            message: User input message
            
        Returns:
            Processing result with actions taken
        """
        if self.status != LoopStatus.RUNNING:
            raise AgentLoopError(f"Agent is not running (status: {self.status})")
        
        if self.current_context.loop_count >= self.config.max_loops:
            self.status = LoopStatus.COMPLETED
            return {"status": "completed", "reason": "max_loops_reached"}
        
        try:
            # Step 1: Add message to context
            self._add_message_to_context(message, MessageType.USER)
            
            # Step 2: Intent recognition
            intent_result = self._recognize_intent(message)
            
            # Step 3: Generate system prompt
            system_prompt = self._generate_system_prompt(intent_result)
            
            # Step 4: Process intent and execute actions
            result = await self._process_intent(intent_result, system_prompt)
            
            # Step 5: Update context and state
            await self._update_context_and_state(result)
            
            # Step 6: Check for loop continuation
            should_continue = not result.get("prevent_continuation", False)
            
            if not should_continue:
                self.status = LoopStatus.COMPLETED
                result["status"] = "completed"
                result["reason"] = "prevent_continuation"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent loop error: {e}")
            self.status = LoopStatus.ERROR
            return {
                "status": "error",
                "error": str(e),
                "actions": []
            }
    
    def _add_message_to_context(self, content: str, message_type: MessageType) -> None:
        """Add a message to the current context."""
        message = Message(
            type=message_type,
            content=content,
            metadata={"loop_count": self.current_context.loop_count}
        )
        self.context_manager.add_message(message)
        self.current_context.messages.append(message)
    
    def _recognize_intent(self, message: str) -> IntentRecognitionResult:
        """Recognize user intent from message."""
        context = {
            "messages": [msg.content for msg in self.current_context.messages[-5:]],
            "workspace": self.config.workspace_path,
            "loop_count": self.current_context.loop_count
        }
        
        return self.intent_recognizer.recognize_intent(message, context)
    
    def _generate_system_prompt(
        self, 
        intent_result: IntentRecognitionResult
    ) -> str:
        """Generate system prompt based on current context."""
        session_context = {
            "workspace_path": self.config.workspace_path,
            "user_id": self.config.user_id,
            "loop_count": self.current_context.loop_count,
            "git_repo": self._check_git_repo(),
            "use_todo": True,
            "run_tests": True
        }
        
        return self.prompt_generator.generate_system_prompt(
            workspace_path=self.config.workspace_path,
            session_context=session_context,
            available_tools=[tool.name for tool in self.registry.list_tools()]
        )
    
    async def _process_intent(
        self, 
        intent_result: IntentRecognitionResult,
        system_prompt: str
    ) -> Dict[str, Any]:
        """Process the recognized intent and execute appropriate actions."""
        
        self.logger.info(f"Processing intent: {intent_result.task_type}")
        
        # Handle special commands
        if intent_result.task_type.value == "exit":
            return {"prevent_continuation": True, "actions": ["exit"]}
        
        if intent_result.task_type.value == "clear":
            self.context_manager.clear_short_term()
            return {"actions": ["clear_memory"], "message": "Context cleared"}
        
        # Handle task tool for complex operations
        if intent_result.complexity.value == "highest" and self.config.enable_task_tool:
            return await self._handle_complex_task(intent_result)
        
        # Handle regular tool execution
        return await self._handle_regular_task(intent_result)
    
    async def _handle_complex_task(
        self, 
        intent_result: IntentRecognitionResult
    ) -> Dict[str, Any]:
        """Handle complex tasks using Task tool and SubAgent mechanism."""
        try:
            task_result = await self.task_tool.execute_task(
                description=intent_result.original_message,
                tools=intent_result.recommended_tools,
                context=intent_result.parameters
            )
            
            # Add assistant response
            response = self._format_task_response(task_result)
            self._add_message_to_context(response, MessageType.ASSISTANT)
            
            return {
                "actions": ["task_execution"],
                "result": task_result,
                "assistant_response": response
            }
            
        except Exception as e:
            self.logger.error(f"Task tool execution failed: {e}")
            return {
                "actions": ["task_error"],
                "error": str(e)
            }
    
    async def _handle_regular_task(
        self, 
        intent_result: IntentRecognitionResult
    ) -> Dict[str, Any]:
        """Handle regular tasks with direct tool execution."""
        
        # Create tool calls based on intent
        tool_calls = self._create_tool_calls(intent_result)
        
        if not tool_calls:
            # No tools needed, provide direct response
            response = self._generate_direct_response(intent_result)
            self._add_message_to_context(response, MessageType.ASSISTANT)
            return {"actions": ["direct_response"], "response": response}
        
        # Execute tools
        results = await self.tool_orchestrator.execute_tools(
            tool_calls, 
            {"intent": intent_result}
        )
        
        # Process results
        response = self._process_tool_results(results, intent_result, tool_calls)
        self._add_message_to_context(response, MessageType.ASSISTANT)
        
        return {
            "actions": ["tool_execution"],
            "tool_results": results,
            "assistant_response": response
        }
    
    def _create_tool_calls(self, intent_result: IntentRecognitionResult) -> List[ToolCall]:
        """Create tool calls based on intent recognition."""
        tool_calls = []
        
        # Map intent to specific tools
        tool_mapping = {
            "search": ["list_directory", "read_file"],
            "analyze": ["read_file", "analyze_code"],
            "create": ["write_file"],
            "modify": ["read_file", "write_file"],
            "debug": ["read_file"]
        }
        
        recommended_tools = tool_mapping.get(intent_result.task_type.value, [])
        
        for tool_name in recommended_tools:
            if self.registry.get_tool(tool_name):
                tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    parameters=intent_result.parameters
                ))
        
        return tool_calls
    
    def _format_task_response(self, task_result: Dict[str, Any]) -> str:
        """Format response from task execution."""
        if task_result.get("success"):
            findings = task_result.get("findings", [])
            recommendations = task_result.get("recommendations", [])
            
            response_parts = []
            if findings:
                response_parts.append("**Findings:**")
                response_parts.extend(f"- {finding}" for finding in findings)
            
            if recommendations:
                response_parts.append("**Recommendations:**")
                response_parts.extend(f"- {rec}" for rec in recommendations)
            
            return "\n".join(response_parts)
        else:
            return f"Task execution failed: {task_result.get('error', 'Unknown error')}"
    
    def _process_tool_results(
        self, 
        results: List[Any], 
        intent_result: IntentRecognitionResult,
        tool_calls: List[ToolCall]
    ) -> str:
        """Process tool execution results into response."""
        if not results:
            return "No tools executed."
        
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        response_parts = []
        
        if successful_results:
            response_parts.append("**Execution Results:**")
            for result in successful_results:
                tool_name = next((call.tool_name for call in tool_calls if call.id == result.tool_call_id), "Unknown")
                response_parts.append(f"- {tool_name}: Success")
        
        if failed_results:
            response_parts.append("**Errors:**")
            for result in failed_results:
                tool_name = next((call.tool_name for call in tool_calls if call.id == result.tool_call_id), "Unknown")
                response_parts.append(f"- {tool_name}: {result.error_message}")
        
        return "\n".join(response_parts)
    
    def _generate_direct_response(self, intent_result: IntentRecognitionResult) -> str:
        """Generate direct response when no tools are needed."""
        return f"I understand you want to {intent_result.intent}. I'll help you with that."
    
    async def _update_context_and_state(self, result: Dict[str, Any]) -> None:
        """Update context and state based on processing result."""
        self.current_context.loop_count += 1
        
        # Update state manager
        self.state_manager.update_global_state(
            "loop_count", 
            self.current_context.loop_count,
            "agent_loop"
        )
        
        # Save session if needed
        if self.current_context.loop_count % 10 == 0:
            self.context_manager.save_session(self.current_context.session_id)
    
    def _check_git_repo(self) -> bool:
        """Check if current workspace is a git repository."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.config.workspace_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def pause(self) -> None:
        """Pause the agent loop."""
        if self.status == LoopStatus.RUNNING:
            self.status = LoopStatus.PAUSED
            self.logger.info("Agent loop paused")
    
    def resume(self) -> None:
        """Resume the agent loop."""
        if self.status == LoopStatus.PAUSED:
            self.status = LoopStatus.RUNNING
            self.logger.info("Agent loop resumed")
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self.status = LoopStatus.COMPLETED
        self.context_manager.save_session(self.current_context.session_id)
        self.logger.info("Agent loop stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "status": self.status.value,
            "loop_count": self.current_context.loop_count,
            "max_loops": self.config.max_loops,
            "session_id": self.current_context.session_id,
            "messages_count": len(self.current_context.messages),
            "workspace": self.config.workspace_path
        }
    
    def add_listener(self, callback: Callable) -> None:
        """Add event listener for agent events."""
        self._listeners.append(callback)
    
    def remove_listener(self, callback: Callable) -> None:
        """Remove event listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def _notify_listeners(self, event: Dict[str, Any]) -> None:
        """Notify all listeners of events."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"Listener error: {e}")


# Convenience factory functions
def create_agent(config: AgentConfig = None) -> AgentLoopOrchestrator:
    """Create a new agent instance with default configuration."""
    return AgentLoopOrchestrator(config)


async def run_agent_demo():
    """Demo function to show agent capabilities."""
    import tempfile
    import os
    
    # Create temporary directory for demo
    demo_dir = tempfile.mkdtemp()
    
    # Create test file
    test_file = os.path.join(demo_dir, "hello.py")
    with open(test_file, 'w') as f:
        f.write('print("Hello from Claude Agent!")')
    
    # Create agent
    config = AgentConfig(workspace_path=demo_dir, max_loops=1)
    agent = create_agent(config)
    
    # Start and process message
    await agent.start_loop()
    result = await agent.process_message("Analyze the hello.py file")
    
    print("Agent Demo Result:")
    print(json.dumps(result, indent=2))
    
    # Cleanup
    import shutil
    shutil.rmtree(demo_dir)


if __name__ == "__main__":
    import json
    asyncio.run(run_agent_demo())