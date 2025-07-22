"""
Task tool implementation - SubAgent mechanism.
Provides stateless SubAgent system for complex task handling.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

from ..agent.loop import Context, Message, MessageType, ToolCall, ToolResult
from ..agent.context_manager import ContextManager
from .tool_manager import ToolRegistry, ToolOrchestrator
from ..utils.intent_recognition import IntentRecognizer, TaskComplexity


@dataclass
class SubAgentTask:
    """Represents a task for a SubAgent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    tools: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 10
    timeout: int = 300


@dataclass
class SubAgentResult:
    """Result from SubAgent execution."""
    task_id: str
    success: bool
    findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubAgent:
    """Stateless SubAgent for parallel task execution."""
    
    def __init__(
        self,
        agent_id: str,
        registry: ToolRegistry,
        context_manager: ContextManager,
        intent_recognizer: IntentRecognizer
    ):
        self.agent_id = agent_id
        self.registry = registry
        self.context_manager = context_manager
        self.intent_recognizer = intent_recognizer
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
    
    async def execute_task(self, task: SubAgentTask) -> SubAgentResult:
        """
        Execute a task as a stateless SubAgent.
        
        Args:
            task: The task to execute
            
        Returns:
            SubAgentResult with findings and recommendations
        """
        self.logger.info(f"SubAgent {self.agent_id} executing task: {task.description}")
        
        # Create isolated context for this SubAgent
        isolated_context = self._create_isolated_context(task)
        
        try:
            # Decompose task into subtasks
            subtasks = await self._decompose_task(task)
            
            # Execute subtasks
            subtask_results = []
            for subtask in subtasks:
                result = await self._execute_subtask(subtask, isolated_context)
                subtask_results.append(result)
            
            # Aggregate results
            final_result = await self._aggregate_results(task, subtask_results)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"SubAgent {self.agent_id} failed: {e}")
            return SubAgentResult(
                task_id=task.id,
                success=False,
                findings=[f"Error: {str(e)}"]
            )
    
    def _create_isolated_context(self, task: SubAgentTask) -> Dict[str, Any]:
        """Create an isolated context for this SubAgent."""
        return {
            "agent_id": self.agent_id,
            "task_id": task.id,
            "description": task.description,
            "tools": task.tools,
            "timeout": task.timeout,
            "max_iterations": task.max_iterations,
            **task.context
        }
    
    async def _decompose_task(self, task: SubAgentTask) -> List[SubAgentTask]:
        """Decompose a complex task into smaller subtasks."""
        # TODO: Implement intelligent task decomposition
        
        # Simple decomposition based on task complexity
        if len(task.description) < 100:
            return [task]
        
        # Split into research, analysis, and implementation phases
        subtasks = [
            SubAgentTask(
                description=f"Research phase: {task.description}",
                tools=["read_file", "list_directory", "analyze_code"],
                max_iterations=3,
                timeout=60
            ),
            SubAgentTask(
                description=f"Analysis phase: {task.description}",
                tools=["analyze_code"],
                max_iterations=2,
                timeout=60
            ),
            SubAgentTask(
                description=f"Implementation phase: {task.description}",
                tools=["write_file", "read_file"],
                max_iterations=5,
                timeout=120
            )
        ]
        
        return subtasks
    
    async def _execute_subtask(
        self, 
        subtask: SubAgentTask, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single subtask."""
        self.logger.debug(f"Executing subtask: {subtask.description}")
        
        # Create tool calls for this subtask
        tool_calls = []
        for tool_name in subtask.tools:
            if self.registry.get_tool(tool_name):
                tool_calls.append(ToolCall(
                    tool_name=tool_name,
                    parameters={"context": context}
                ))
        
        # Initialize isolated context manager
        isolated_manager = ContextManager()
        
        # Execute tools
        orchestrator = ToolOrchestrator(self.registry)
        results = await orchestrator.execute_tools(tool_calls, context)
        
        return {
            "subtask_id": subtask.id,
            "description": subtask.description,
            "tool_results": results,
            "success": all(r.success for r in results)
        }
    
    async def _aggregate_results(
        self, 
        original_task: SubAgentTask,
        subtask_results: List[Dict[str, Any]]
    ) -> SubAgentResult:
        """Aggregate results from all subtasks."""
        findings = []
        recommendations = []
        code_examples = []
        conflicts = []
        
        # Extract findings from each subtask
        for result in subtask_results:
            if result["success"]:
                findings.extend(self._extract_findings(result))
                recommendations.extend(self._extract_recommendations(result))
                code_examples.extend(self._extract_code_examples(result))
            else:
                conflicts.append(f"Subtask {result['subtask_id']} failed")
        
        # TODO: Use LLM for intelligent result aggregation
        # For now, use simple concatenation
        
        return SubAgentResult(
            task_id=original_task.id,
            success=len(conflicts) == 0,
            findings=findings,
            recommendations=recommendations,
            code_examples=code_examples,
            conflicts=conflicts,
            metadata={
                "subtask_count": len(subtask_results),
                "successful_subtasks": len([r for r in subtask_results if r["success"]]),
                "total_findings": len(findings)
            }
        )
    
    def _extract_findings(self, result: Dict[str, Any]) -> List[str]:
        """Extract key findings from subtask results."""
        findings = []
        
        for tool_result in result.get("tool_results", []):
            if tool_result.success and tool_result.content:
                findings.append(str(tool_result.content))
        
        return findings
    
    def _extract_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Extract recommendations from subtask results."""
        recommendations = []
        
        # TODO: Implement intelligent recommendation generation
        recommendations.append("Review findings and implement necessary changes")
        
        return recommendations
    
    def _extract_code_examples(self, result: Dict[str, Any]) -> List[str]:
        """Extract code examples from subtask results."""
        code_examples = []
        
        # TODO: Implement code example extraction
        code_examples.append("// Code examples will be generated based on findings")
        
        return code_examples


class TaskTool:
    """Main Task tool that manages SubAgent execution."""
    
    def __init__(
        self,
        registry: ToolRegistry,
        context_manager: ContextManager,
        intent_recognizer: IntentRecognizer
    ):
        self.registry = registry
        self.context_manager = context_manager
        self.intent_recognizer = intent_recognizer
        self.logger = logging.getLogger(__name__)
    
    async def execute_task(
        self,
        description: str,
        tools: List[str] = None,
        max_agents: int = 3,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a complex task using SubAgent mechanism.
        
        Args:
            description: Task description
            tools: List of tools to use
            max_agents: Maximum number of concurrent SubAgents
            context: Additional context
            
        Returns:
            Aggregated results from all SubAgents
        """
        if context is None:
            context = {}
        
        if tools is None:
            tools = ["read_file", "write_file", "analyze_code"]
        
        self.logger.info(f"Starting Task tool execution: {description}")
        
        try:
            # Analyze task complexity
            intent_result = self.intent_recognizer.recognize_intent(description)
            
            # Create SubAgent task
            task = SubAgentTask(
                description=description,
                tools=tools,
                context=context
            )
            
            # Determine number of SubAgents needed
            agent_count = self._determine_agent_count(intent_result, max_agents)
            
            # Create and execute SubAgents
            subagents = []
            for i in range(agent_count):
                agent_id = f"subagent_{i+1}_{uuid.uuid4().hex[:8]}"
                subagent = SubAgent(
                    agent_id=agent_id,
                    registry=self.registry,
                    context_manager=self.context_manager,
                    intent_recognizer=self.intent_recognizer
                )
                subagents.append(subagent)
            
            # Execute with parallel SubAgents
            if len(subagents) == 1:
                result = await subagents[0].execute_task(task)
                return self._format_task_result(result)
            else:
                # Split task among multiple SubAgents
                subtasks = self._split_task_for_parallel_execution(task, subagents)
                
                # Execute in parallel
                results = await asyncio.gather(*[
                    subagent.execute_task(subtask)
                    for subagent, subtask in zip(subagents, subtasks)
                ])
                
                # Aggregate parallel results
                final_result = await self._aggregate_parallel_results(results)
                return self._format_task_result(final_result)
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "findings": [],
                "recommendations": []
            }
    
    def _determine_agent_count(
        self, 
        intent_result: Any, 
        max_agents: int
    ) -> int:
        """Determine optimal number of SubAgents for a task."""
        # TODO: Implement intelligent agent count determination
        
        if intent_result.complexity.value == "highest":
            return min(max_agents, 3)
        elif intent_result.complexity.value == "middle":
            return min(max_agents, 2)
        else:
            return 1
    
    def _split_task_for_parallel_execution(
        self, 
        task: SubAgentTask, 
        subagents: List[SubAgent]
    ) -> List[SubAgentTask]:
        """Split a task for parallel execution among multiple SubAgents."""
        # TODO: Implement intelligent task splitting
        
        # Simple splitting based on file types or directories
        descriptions = [
            f"{task.description} - Research phase",
            f"{task.description} - Analysis phase",
            f"{task.description} - Implementation phase"
        ]
        
        subtasks = []
        for i, subagent in enumerate(subagents):
            subtask = SubAgentTask(
                description=descriptions[i % len(descriptions)],
                tools=task.tools,
                max_iterations=task.max_iterations // len(subagents),
                timeout=task.timeout // len(subagents)
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def _aggregate_parallel_results(
        self, 
        results: List[SubAgentResult]
    ) -> SubAgentResult:
        """Aggregate results from parallel SubAgent execution."""
        # TODO: Use LLM for intelligent result aggregation
        
        all_findings = []
        all_recommendations = []
        all_code_examples = []
        all_conflicts = []
        
        for result in results:
            all_findings.extend(result.findings)
            all_recommendations.extend(result.recommendations)
            all_code_examples.extend(result.code_examples)
            all_conflicts.extend(result.conflicts)
        
        # Remove duplicates
        all_findings = list(set(all_findings))
        all_recommendations = list(set(all_recommendations))
        all_code_examples = list(set(all_code_examples))
        all_conflicts = list(set(all_conflicts))
        
        return SubAgentResult(
            task_id="aggregated",
            success=all(not result.success for result in results),
            findings=all_findings,
            recommendations=all_recommendations,
            code_examples=all_code_examples,
            conflicts=all_conflicts,
            metadata={
                "subagent_count": len(results),
                "successful_subagents": len([r for r in results if r.success])
            }
        )
    
    def _format_task_result(self, result: SubAgentResult) -> Dict[str, Any]:
        """Format the final task result."""
        return {
            "success": result.success,
            "findings": result.findings,
            "recommendations": result.recommendations,
            "code_examples": result.code_examples,
            "conflicts": result.conflicts,
            "metadata": result.metadata
        }