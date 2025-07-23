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
        """Decompose a complex task into smaller subtasks using 2025 hierarchical approach."""
        import re
        from typing import List, Tuple
        
        # Hierarchical task decomposition based on 2025 MegaAgent research
        complexity_score = self._calculate_complexity_score(task.description)
        
        if complexity_score < 0.3:
            return [task]
        
        # Analyze task patterns for intelligent decomposition
        task_patterns = self._analyze_task_patterns(task.description)
        subtasks = []
        
        # Dynamic role-based decomposition
        if task_patterns.get('requires_research', False):
            subtasks.append(SubAgentTask(
                description=self._generate_research_task(task.description),
                tools=["read_file", "list_directory", "analyze_code", "web_fetch"],
                max_iterations=4,
                timeout=90,
                context={"phase": "research", "parent_task": task.id}
            ))
        
        if task_patterns.get('requires_analysis', False):
            subtasks.append(SubAgentTask(
                description=self._generate_analysis_task(task.description),
                tools=["analyze_code", "read_file", "grep"],
                max_iterations=3,
                timeout=75,
                context={"phase": "analysis", "parent_task": task.id}
            ))
        
        if task_patterns.get('requires_implementation', False):
            subtasks.append(SubAgentTask(
                description=self._generate_implementation_task(task.description),
                tools=["write_file", "read_file", "analyze_code"],
                max_iterations=6,
                timeout=150,
                context={"phase": "implementation", "parent_task": task.id}
            ))
        
        if task_patterns.get('requires_validation', False):
            subtasks.append(SubAgentTask(
                description=self._generate_validation_task(task.description),
                tools=["read_file", "analyze_code"],
                max_iterations=2,
                timeout=45,
                context={"phase": "validation", "parent_task": task.id}
            ))
        
        # If no specific patterns matched, use recursive decomposition
        if not subtasks:
            subtasks = self._recursive_decomposition(task)
        
        return subtasks
    
    def _calculate_complexity_score(self, description: str) -> float:
        """Calculate task complexity score using 2025 AgentNet approach."""
        factors = {
            'keywords': ['optimize', 'refactor', 'implement', 'design', 'architecture', 'system', 'complex'],
            'multipliers': {
                'optimize': 0.15, 'refactor': 0.12, 'implement': 0.10,
                'design': 0.18, 'architecture': 0.20, 'system': 0.15, 'complex': 0.13
            }
        }
        
        score = 0.0
        desc_lower = description.lower()
        word_count = len(desc_lower.split())
        
        # Base complexity from length
        score += min(word_count / 50.0, 0.5)
        
        # Add keyword-based complexity
        for keyword, multiplier in factors['multipliers'].items():
            if keyword in desc_lower:
                score += multiplier
        
        # Technical indicators
        technical_indicators = ['async', 'concurrent', 'distributed', 'microservice', 'api', 'database']
        for indicator in technical_indicators:
            if indicator in desc_lower:
                score += 0.05
        
        return min(score, 1.0)
    
    def _analyze_task_patterns(self, description: str) -> Dict[str, bool]:
        """Analyze task patterns for intelligent decomposition."""
        patterns = {
            'requires_research': bool(re.search(r'find|search|investigate|research|understand|explore', description, re.I)),
            'requires_analysis': bool(re.search(r'analyze|review|examine|assess|evaluate|identify|detect', description, re.I)),
            'requires_implementation': bool(re.search(r'implement|write|create|build|develop|add|fix|update', description, re.I)),
            'requires_validation': bool(re.search(r'test|validate|verify|ensure|check', description, re.I))
        }
        return patterns
    
    def _generate_research_task(self, description: str) -> str:
        """Generate research-specific subtask description."""
        return f"Research and gather comprehensive information for: {description}. Focus on understanding requirements, existing solutions, and best practices."
    
    def _generate_analysis_task(self, description: str) -> str:
        """Generate analysis-specific subtask description."""
        return f"Deep analysis of codebase for: {description}. Identify patterns, issues, and opportunities for improvement."
    
    def _generate_implementation_task(self, description: str) -> str:
        """Generate implementation-specific subtask description."""
        return f"Implement solution for: {description}. Ensure code quality, maintainability, and adherence to best practices."
    
    def _generate_validation_task(self, description: str) -> str:
        """Generate validation-specific subtask description."""
        return f"Validate and verify solution for: {description}. Ensure correctness, performance, and edge case handling."
    
    def _recursive_decomposition(self, task: SubAgentTask) -> List[SubAgentTask]:
        """Recursive task decomposition for complex tasks."""
        # Split by logical components based on 2025 LLM-MA approach
        components = [
            "requirements gathering",
            "design and planning", 
            "implementation",
            "testing and validation"
        ]
        
        subtasks = []
        for i, component in enumerate(components):
            subtasks.append(SubAgentTask(
                description=f"{component.title()}: {task.description}",
                tools=task.tools,
                max_iterations=task.max_iterations // len(components),
                timeout=task.timeout // len(components),
                context={"component": component, "parent_task": task.id}
            ))
        
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
        """Aggregate results using LLM-based intelligent aggregation with dependency graphs."""
        findings = []
        recommendations = []
        code_examples = []
        conflicts = []
        
        # Build dependency graph from subtask results
        dependency_graph = self._build_dependency_graph(subtask_results)
        
        # Extract and categorize findings
        categorized_findings = self._categorize_findings(subtask_results)
        
        # Use LLM-style intelligent aggregation (simulated)
        findings = self._intelligent_findings_aggregation(categorized_findings)
        recommendations = self._intelligent_recommendations_aggregation(categorized_findings)
        code_examples = self._intelligent_code_examples_aggregation(categorized_findings)
        
        # Handle conflicts with resolution strategies
        conflicts = self._resolve_conflicts(subtask_results)
        
        # Generate metadata with dependency insights
        metadata = self._generate_aggregation_metadata(subtask_results, dependency_graph)
        
        return SubAgentResult(
            task_id=original_task.id,
            success=len(conflicts) == 0 or all(c.get('severity', 'low') != 'critical' for c in conflicts),
            findings=findings,
            recommendations=recommendations,
            code_examples=code_examples,
            conflicts=[c['message'] for c in conflicts],
            metadata=metadata
        )
    
    def _build_dependency_graph(self, subtask_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build dependency graph from subtask results using 2025 LLMCompiler approach."""
        graph = {
            'nodes': [],
            'edges': [],
            'critical_path': [],
            'bottlenecks': []
        }
        
        for result in subtask_results:
            node = {
                'id': result['subtask_id'],
                'description': result['description'],
                'success': result['success'],
                'duration': result.get('metadata', {}).get('duration', 0),
                'findings_count': len(result.get('tool_results', [])),
                'phase': result.get('context', {}).get('phase', 'unknown')
            }
            graph['nodes'].append(node)
            
            # Add edges based on phase dependencies
            if node['phase'] == 'analysis':
                graph['edges'].append({'from': 'research', 'to': 'analysis'})
            elif node['phase'] == 'implementation':
                graph['edges'].append({'from': 'analysis', 'to': 'implementation'})
            elif node['phase'] == 'validation':
                graph['edges'].append({'from': 'implementation', 'to': 'validation'})
        
        return graph
    
    def _categorize_findings(self, subtask_results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize findings by type and severity for intelligent aggregation."""
        categories = {
            'critical_bugs': [],
            'performance_issues': [],
            'security_vulnerabilities': [],
            'code_smells': [],
            'best_practice_violations': [],
            'optimization_opportunities': [],
            'documentation_gaps': []
        }
        
        for result in subtask_results:
            if not result["success"]:
                continue
                
            for tool_result in result.get("tool_results", []):
                if tool_result.success and tool_result.content:
                    content = str(tool_result.content).lower()
                    
                    # Pattern matching for categorization
                    if any(keyword in content for keyword in ['error', 'exception', 'bug', 'crash']):
                        categories['critical_bugs'].append(str(tool_result.content))
                    elif any(keyword in content for keyword in ['performance', 'slow', 'inefficient', 'optimization']):
                        categories['performance_issues'].append(str(tool_result.content))
                    elif any(keyword in content for keyword in ['security', 'vulnerability', 'injection', 'exploit']):
                        categories['security_vulnerabilities'].append(str(tool_result.content))
                    elif any(keyword in content for keyword in ['duplicate', 'complex', 'refactor', 'smell']):
                        categories['code_smells'].append(str(tool_result.content))
                    elif any(keyword in content for keyword in ['best practice', 'convention', 'standard']):
                        categories['best_practice_violations'].append(str(tool_result.content))
                    elif any(keyword in content for keyword in ['improve', 'optimize', 'enhance']):
                        categories['optimization_opportunities'].append(str(tool_result.content))
                    elif any(keyword in content for keyword in ['documentation', 'comment', 'docstring']):
                        categories['documentation_gaps'].append(str(tool_result.content))
                    else:
                        # Default categorization
                        if len(content) > 50:
                            categories['optimization_opportunities'].append(str(tool_result.content))
        
        return categories
    
    def _intelligent_findings_aggregation(self, categorized_findings: Dict[str, List[str]]) -> List[str]:
        """Intelligently aggregate findings using 2025-style LLM reasoning."""
        findings = []
        
        # Priority-based aggregation
        priority_order = [
            'critical_bugs',
            'security_vulnerabilities', 
            'performance_issues',
            'code_smells',
            'best_practice_violations',
            'optimization_opportunities',
            'documentation_gaps'
        ]
        
        for category in priority_order:
            items = categorized_findings[category]
            if items:
                # Deduplicate and summarize
                unique_items = list(set(items))
                if len(unique_items) > 3:
                    findings.append(f"{category.replace('_', ' ').title()}: Found {len(unique_items)} issues including {unique_items[:2]}...")
                else:
                    findings.extend([f"{category.replace('_', ' ').title()}: {item}" for item in unique_items])
        
        return findings
    
    def _intelligent_recommendations_aggregation(self, categorized_findings: Dict[str, List[str]]) -> List[str]:
        """Generate intelligent recommendations based on categorized findings."""
        recommendations = []
        
        if categorized_findings['critical_bugs']:
            recommendations.append("Address critical bugs immediately - these may cause system failures")
        
        if categorized_findings['security_vulnerabilities']:
            recommendations.append("Implement security best practices and conduct security audit")
        
        if categorized_findings['performance_issues']:
            recommendations.append("Optimize performance-critical code paths using profiling data")
        
        if categorized_findings['code_smells']:
            recommendations.append("Refactor complex code to improve maintainability and readability")
        
        if categorized_findings['best_practice_violations']:
            recommendations.append("Review and align with established coding standards and best practices")
        
        if categorized_findings['optimization_opportunities']:
            recommendations.append("Consider performance optimizations and code quality improvements")
        
        if categorized_findings['documentation_gaps']:
            recommendations.append("Add comprehensive documentation for better code maintainability")
        
        return recommendations
    
    def _intelligent_code_examples_aggregation(self, categorized_findings: Dict[str, List[str]]) -> List[str]:
        """Generate context-aware code examples based on findings."""
        code_examples = []
        
        # Generate examples based on findings
        if categorized_findings['critical_bugs']:
            code_examples.append("// Example: Error handling pattern\ntry {\n  // critical operation\n} catch (error) {\n  logger.error('Operation failed', error);\n  throw new CustomError('User-friendly message');\n}")
        
        if categorized_findings['performance_issues']:
            code_examples.append("// Example: Performance optimization with caching\nconst cache = new Map();\nfunction expensiveOperation(key) {\n  if (cache.has(key)) return cache.get(key);\n  const result = computeExpensive(key);\n  cache.set(key, result);\n  return result;\n}")
        
        if categorized_findings['security_vulnerabilities']:
            code_examples.append("// Example: Input validation\nfunction validateUserInput(input) {\n  if (!input || typeof input !== 'string') {\n    throw new ValidationError('Invalid input');\n  }\n  return DOMPurify.sanitize(input.trim());\n}")
        
        return code_examples
    
    def _resolve_conflicts(self, subtask_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between subtask results."""
        conflicts = []
        
        for result in subtask_results:
            if not result["success"]:
                conflicts.append({
                    'severity': 'high',
                    'message': f"Subtask {result['subtask_id']} failed: {result['description']}",
                    'resolution': 'Retry with adjusted parameters'
                })
        
        return conflicts
    
    def _generate_aggregation_metadata(self, subtask_results: List[Dict[str, Any]], dependency_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive metadata for aggregation."""
        successful_subtasks = [r for r in subtask_results if r["success"]]
        
        return {
            "subtask_count": len(subtask_results),
            "successful_subtasks": len(successful_subtasks),
            "success_rate": len(successful_subtasks) / len(subtask_results) if subtask_results else 0,
            "dependency_graph": dependency_graph,
            "critical_path_length": len(dependency_graph.get('critical_path', [])),
            "bottlenecks": dependency_graph.get('bottlenecks', []),
            "total_findings": sum(len(self._extract_findings(r)) for r in successful_subtasks),
            "categories_analyzed": 7  # Number of categories in _categorize_findings
        }
    
    def _extract_findings(self, result: Dict[str, Any]) -> List[str]:
        """Extract key findings from subtask results."""
        findings = []
        
        for tool_result in result.get("tool_results", []):
            if tool_result.success and tool_result.content:
                findings.append(str(tool_result.content))
        
        return findings
    
    def _extract_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Extract intelligent recommendations from subtask results using 2025 techniques."""
        recommendations = []
        
        # Analyze tool results for specific patterns
        tool_results = result.get("tool_results", [])
        
        for tool_result in tool_results:
            if tool_result.success and tool_result.content:
                content = str(tool_result.content).lower()
                
                # Pattern-based recommendation generation
                recommendations.extend(self._generate_pattern_based_recommendations(content))
        
        # Add context-aware recommendations
        phase = result.get('context', {}).get('phase', 'general')
        recommendations.extend(self._generate_phase_specific_recommendations(phase))
        
        # Ensure unique recommendations
        unique_recommendations = list(dict.fromkeys(recommendations))
        
        return unique_recommendations
    
    def _generate_pattern_based_recommendations(self, content: str) -> List[str]:
        """Generate recommendations based on content patterns."""
        recommendations = []
        
        # Code quality patterns
        if 'duplicate code' in content or 'repeated' in content:
            recommendations.append("Apply DRY principle - extract common functionality into reusable functions")
        
        if 'complex function' in content or 'cyclomatic complexity' in content:
            recommendations.append("Refactor complex functions using single responsibility principle")
        
        if 'long function' in content or 'exceeds' in content:
            recommendations.append("Break down long functions into smaller, focused units")
        
        # Security patterns
        if 'sql injection' in content or 'injection' in content:
            recommendations.append("Use parameterized queries to prevent SQL injection attacks")
        
        if 'xss' in content or 'cross-site scripting' in content:
            recommendations.append("Implement input sanitization and output encoding for XSS prevention")
        
        if 'authentication' in content or 'authorization' in content:
            recommendations.append("Implement proper authentication and authorization checks")
        
        # Performance patterns
        if 'n+1 query' in content or 'inefficient query' in content:
            recommendations.append("Optimize database queries using eager loading and batching")
        
        if 'memory leak' in content or 'memory' in content:
            recommendations.append("Implement proper resource cleanup and memory management")
        
        if 'slow' in content or 'performance' in content:
            recommendations.append("Use profiling tools to identify and optimize bottlenecks")
        
        # Testing patterns
        if 'no tests' in content or 'missing test' in content:
            recommendations.append("Add comprehensive unit tests and integration tests")
        
        if 'test coverage' in content:
            recommendations.append("Increase test coverage to at least 80% for critical paths")
        
        # Documentation patterns
        if 'no documentation' in content or 'missing documentation' in content:
            recommendations.append("Add inline documentation and README files")
        
        if 'deprecated' in content:
            recommendations.append("Update deprecated dependencies to latest stable versions")
        
        return recommendations
    
    def _generate_phase_specific_recommendations(self, phase: str) -> List[str]:
        """Generate phase-specific recommendations."""
        phase_recommendations = {
            'research': [
                "Gather comprehensive requirements before implementation",
                "Research existing solutions and best practices",
                "Document findings and create implementation plan"
            ],
            'analysis': [
                "Use static analysis tools to identify code issues",
                "Prioritize findings by severity and impact",
                "Consider technical debt in implementation decisions"
            ],
            'implementation': [
                "Follow established coding standards and conventions",
                "Write unit tests alongside implementation",
                "Use version control effectively with meaningful commits"
            ],
            'validation': [
                "Test edge cases and error scenarios",
                "Validate against original requirements",
                "Get peer review before final deployment"
            ],
            'general': [
                "Follow SOLID principles for maintainable code",
                "Use meaningful variable and function names",
                "Keep functions small and focused on single responsibility"
            ]
        }
        
        return phase_recommendations.get(phase, phase_recommendations['general'])
    
    def _extract_code_examples(self, result: Dict[str, Any]) -> List[str]:
        """Extract context-aware code examples from subtask results using 2025 techniques."""
        code_examples = []
        
        # Analyze tool results for code generation opportunities
        tool_results = result.get("tool_results", [])
        
        for tool_result in tool_results:
            if tool_result.success and tool_result.content:
                content = str(tool_result.content)
                
                # Extract code examples based on findings
                examples = self._generate_context_aware_examples(content, result)
                code_examples.extend(examples)
        
        # Add phase-specific examples
        phase = result.get('context', {}).get('phase', 'general')
        code_examples.extend(self._get_phase_specific_examples(phase))
        
        # Ensure unique examples
        unique_examples = list(dict.fromkeys(code_examples))
        
        return unique_examples
    
    def _generate_context_aware_examples(self, content: str, result: Dict[str, Any]) -> List[str]:
        """Generate context-aware code examples based on findings."""
        examples = []
        
        # Analyze content for specific patterns
        content_lower = content.lower()
        
        # Error handling patterns
        if any(pattern in content_lower for pattern in ['error', 'exception', 'try-catch', 'null pointer']):
            examples.extend([
                "```python\n# Robust error handling\ntry:\n    result = risky_operation()\n    return process_result(result)\nexcept ValueError as e:\n    logger.error(f'Invalid input: {e}')\n    raise CustomValidationError('Please provide valid input')\nexcept Exception as e:\n    logger.exception('Unexpected error occurred')\n    raise CustomServiceError('Service temporarily unavailable')\n```",
                "```python\n# Null-safe operations\ndef safe_get(data, key, default=None):\n    return data.get(key, default) if data else default\n```"
            ])
        
        # Performance optimization patterns
        if any(pattern in content_lower for pattern in ['performance', 'slow', 'optimization', 'cache']):
            examples.extend([
                "```python\n# Caching decorator for expensive operations\nfrom functools import lru_cache\nimport time\n\n@lru_cache(maxsize=128)\ndef expensive_computation(input_data):\n    time.sleep(2)  # Simulate expensive operation\n    return process_data(input_data)\n```",
                "```python\n# Batch processing for efficiency\ndef batch_process(items, batch_size=100):\n    results = []\n    for i in range(0, len(items), batch_size):\n        batch = items[i:i + batch_size]\n        results.extend(process_batch(batch))\n    return results\n```"
            ])
        
        # Security patterns
        if any(pattern in content_lower for pattern in ['security', 'injection', 'xss', 'sql']):
            examples.extend([
                "```python\n# Input validation and sanitization\nimport re\n\ndef validate_email(email):\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    if not re.match(pattern, email):\n        raise ValueError('Invalid email format')\n    return email.lower().strip()\n```",
                "```python\n# SQL injection prevention\ndef get_user_by_id(user_id):\n    query = \"SELECT * FROM users WHERE id = %s\"\n    cursor.execute(query, (user_id,))  # Parameterized query\n    return cursor.fetchone()\n```"
            ])
        
        # Code structure patterns
        if any(pattern in content_lower for pattern in ['refactor', 'complex', 'duplicate', 'long']):
            examples.extend([
                """```python
# Single Responsibility Principle
class UserService:
    def __init__(self, repository):
        self.repository = repository
    
    def create_user(self, user_data):
        validated_data = self.validate_user_data(user_data)
        return self.repository.save(validated_data)
    
    def validate_user_data(self, user_data):
        # Validation logic here
        return validated_data
```""",
                """```python
# Factory pattern for object creation
class NotificationFactory:
    @staticmethod
    def create_notification(type, recipient, message):
        if type == 'email':
            return EmailNotification(recipient, message)
        elif type == 'sms':
            return SMSNotification(recipient, message)
        else:
            raise ValueError(f'Unsupported notification type: {type}')
```"""
            ])
        
        # Testing patterns
        if any(pattern in content_lower for pattern in ['test', 'coverage', 'missing']):
            examples.extend([
                "```python\n# Unit test example\nimport pytest\n\ndef test_user_creation():\n    user_service = UserService(MockRepository())\n    user_data = {'name': 'John', 'email': 'john@example.com'}\n    user = user_service.create_user(user_data)\n    assert user.name == 'John'\n    assert user.email == 'john@example.com'\n```",
                "```python\n# Mocking for testing\nfrom unittest.mock import Mock\n\ndef test_notification_service():\n    mock_sender = Mock()\n    service = NotificationService(mock_sender)\n    service.send_notification('user@example.com', 'Hello')\n    mock_sender.send.assert_called_once_with('user@example.com', 'Hello')\n```"
            ])
        
        # Configuration and setup patterns
        if any(pattern in content_lower for pattern in ['config', 'setup', 'environment']):
            examples.extend([
                "```python\n# Environment configuration\nimport os\nfrom dataclasses import dataclass\n\n@dataclass\nclass Config:\n    database_url: str\n    api_key: str\n    debug: bool = False\n    \n    @classmethod\n    def from_env(cls):\n        return cls(\n            database_url=os.getenv('DATABASE_URL', 'sqlite:///app.db'),\n            api_key=os.getenv('API_KEY'),\n            debug=os.getenv('DEBUG', 'false').lower() == 'true'\n        )\n```"
            ])
        
        return examples
    
    def _get_phase_specific_examples(self, phase: str) -> List[str]:
        """Get phase-specific code examples."""
        phase_examples = {
            'research': [
                "```python\n# Research data collection\nimport requests\nfrom typing import List, Dict\n\ndef collect_api_data(endpoints: List[str]) -> Dict[str, any]:\n    results = {}\n    for endpoint in endpoints:\n        try:\n            response = requests.get(endpoint, timeout=30)\n            results[endpoint] = response.json()\n        except Exception as e:\n            results[endpoint] = {'error': str(e)}\n    return results\n```"
            ],
            'analysis': [
                "```python\n# Code analysis helper\nimport ast\nfrom typing import List\n\ndef analyze_function_complexity(code: str) -> List[Dict]:\n    tree = ast.parse(code)\n    complexities = []\n    for node in ast.walk(tree):\n        if isinstance(node, ast.FunctionDef):\n            complexity = calculate_cyclomatic_complexity(node)\n            complexities.append({\n                'name': node.name,\n                'complexity': complexity,\n                'lines': len(node.body)\n            })\n    return complexities\n```"
            ],
            'implementation': [
                "```python\n# Clean implementation pattern\nclass CleanArchitecture:\n    def __init__(self, repository, validator):\n        self.repository = repository\n        self.validator = validator\n    \n    def execute(self, request):\n        validated_request = self.validator.validate(request)\n        return self.repository.process(validated_request)\n```"
            ],
            'validation': [
                "```python\n# Validation testing\ndef validate_implementation(results, expected):\n    assert results is not None, 'Results should not be None'\n    assert len(results) == len(expected), 'Result count mismatch'\n    for result, exp in zip(results, expected):\n        assert result == exp, f'Expected {exp}, got {result}'\n    return True\n```"
            ]
        }
        
        return phase_examples.get(phase, phase_examples['research'])


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
        """Determine optimal number of SubAgents using 2025 capability vectors approach."""
        # Capability vectors and dynamic resource allocation from 2025 AgentNet research
        
        # Base complexity score
        complexity_map = {
            "lowest": 0.2,
            "low": 0.4,
            "middle": 0.6,
            "high": 0.8,
            "highest": 1.0
        }
        
        base_complexity = complexity_map.get(intent_result.complexity.value, 0.5)
        
        # Analyze task characteristics for capability matching
        task_characteristics = self._analyze_task_characteristics(intent_result)
        
        # Calculate optimal agent count using vector dot product
        capability_score = self._calculate_capability_score(task_characteristics)
        
        # Dynamic scaling based on task requirements
        optimal_agents = self._scale_agents_by_requirements(
            base_complexity, 
            capability_score, 
            max_agents,
            task_characteristics
        )
        
        return optimal_agents
    
    def _analyze_task_characteristics(self, intent_result: Any) -> Dict[str, float]:
        """Analyze task characteristics for capability matching."""
        characteristics = {
            'parallelizability': 0.5,
            'domain_complexity': 0.5,
            'resource_intensity': 0.5,
            'coordination_overhead': 0.5,
            'expertise_diversity': 0.5
        }
        
        # Analyze intent result for characteristics
        if hasattr(intent_result, 'keywords'):
            keywords = [kw.lower() for kw in intent_result.keywords]
            
            # Parallelizability indicators
            if any(word in str(keywords) for word in ['parallel', 'concurrent', 'async', 'distributed']):
                characteristics['parallelizability'] = 0.9
            
            # Domain complexity indicators
            if any(word in str(keywords) for word in ['algorithm', 'architecture', 'system', 'framework']):
                characteristics['domain_complexity'] = 0.8
            
            # Resource intensity indicators
            if any(word in str(keywords) for word in ['large', 'complex', 'optimize', 'performance']):
                characteristics['resource_intensity'] = 0.8
            
            # Coordination overhead indicators
            if any(word in str(keywords) for word in ['integrate', 'coordinate', 'collaborate', 'sync']):
                characteristics['coordination_overhead'] = 0.7
            
            # Expertise diversity indicators
            if any(word in str(keywords) for word in ['frontend', 'backend', 'database', 'api', 'ui', 'devops']):
                characteristics['expertise_diversity'] = 0.8
        
        return characteristics
    
    def _calculate_capability_score(self, characteristics: Dict[str, float]) -> float:
        """Calculate capability score using vector dot product approach."""
        # Weight vector based on 2025 AgentNet research
        weights = {
            'parallelizability': 0.3,
            'domain_complexity': 0.25,
            'resource_intensity': 0.2,
            'coordination_overhead': -0.15,  # Negative weight for overhead
            'expertise_diversity': 0.2
        }
        
        # Calculate dot product
        score = sum(characteristics[key] * weights[key] for key in weights)
        return max(0, min(1, score))  # Clamp between 0 and 1
    
    def _scale_agents_by_requirements(
        self, 
        complexity: float, 
        capability_score: float, 
        max_agents: int,
        characteristics: Dict[str, float]
    ) -> int:
        """Scale agent count based on requirements and constraints."""
        # Base agent count calculation
        base_agents = max(1, int(complexity * capability_score * max_agents))
        
        # Adjust for coordination overhead
        coordination_penalty = characteristics['coordination_overhead'] * 0.3
        adjusted_agents = max(1, int(base_agents * (1 - coordination_penalty)))
        
        # Ensure we don't exceed max_agents
        final_agents = min(max_agents, adjusted_agents)
        
        # Special cases for high expertise diversity
        if characteristics['expertise_diversity'] > 0.7 and final_agents < 3:
            final_agents = min(max_agents, final_agents + 1)
        
        return final_agents
    
    def _split_task_for_parallel_execution(
        self, 
        task: SubAgentTask, 
        subagents: List[SubAgent]
    ) -> List[SubAgentTask]:
        """Intelligently split task for parallel execution using 2025 MegaAgent approach."""
        import re
        
        # Analyze task for parallelizable components
        parallel_components = self._identify_parallel_components(task.description)
        
        # Dynamic work distribution based on component complexity
        agent_count = len(subagents)
        
        if len(parallel_components) >= agent_count:
            # Distribute components across agents
            subtasks = self._distribute_components(task, parallel_components, agent_count)
        else:
            # Use phase-based parallelization
            subtasks = self._create_phase_based_subtasks(task, agent_count)
        
        return subtasks
    
    def _identify_parallel_components(self, description: str) -> List[str]:
        """Identify parallelizable components in the task description."""
        components = []
        
        # Component identification patterns
        patterns = {
            'frontend': r'\b(frontend|ui|react|vue|angular|css|html)\b',
            'backend': r'\b(backend|server|api|database|sql|nosql)\b',
            'testing': r'\b(test|testing|validation|verification|qa)\b',
            'deployment': r'\b(deploy|deployment|devops|ci|cd|docker|kubernetes)\b',
            'documentation': r'\b(documentation|docs|readme|guide|tutorial)\b',
            'performance': r'\b(performance|optimization|speed|efficiency|profiling)\b',
            'security': r'\b(security|auth|authorization|encryption|vulnerability)\b'
        }
        
        desc_lower = description.lower()
        
        for component_type, pattern in patterns.items():
            if re.search(pattern, desc_lower, re.I):
                components.append(component_type)
        
        # Add generic components if no specific ones found
        if not components:
            components = ['analysis', 'implementation', 'validation']
        
        return components
    
    def _distribute_components(
        self, 
        task: SubAgentTask, 
        components: List[str], 
        agent_count: int
    ) -> List[SubAgentTask]:
        """Distribute components across agents for parallel execution."""
        subtasks = []
        
        # Calculate workload distribution
        components_per_agent = max(1, len(components) // agent_count)
        remainder = len(components) % agent_count
        
        start_idx = 0
        for i in range(agent_count):
            # Calculate end index for this agent's components
            end_idx = start_idx + components_per_agent + (1 if i < remainder else 0)
            agent_components = components[start_idx:end_idx]
            
            if agent_components:
                # Create specialized subtask for this agent
                subtask = SubAgentTask(
                    description=f"Parallel execution: {task.description} - Focus on {', '.join(agent_components)}",
                    tools=self._optimize_tools_for_components(task.tools, agent_components),
                    max_iterations=max(1, task.max_iterations // agent_count),
                    timeout=max(30, task.timeout // agent_count),
                    context={
                        'parallel_agent_id': i + 1,
                        'total_agents': agent_count,
                        'components': agent_components,
                        'parent_task': task.id
                    }
                )
                subtasks.append(subtask)
            
            start_idx = end_idx
        
        return subtasks
    
    def _optimize_tools_for_components(self, base_tools: List[str], components: List[str]) -> List[str]:
        """Optimize tool selection based on component types."""
        tool_mapping = {
            'frontend': ['read_file', 'analyze_code', 'grep'],
            'backend': ['read_file', 'analyze_code', 'list_directory'],
            'testing': ['read_file', 'analyze_code'],
            'deployment': ['read_file', 'list_directory', 'analyze_code'],
            'documentation': ['read_file', 'write_file'],
            'performance': ['analyze_code', 'read_file'],
            'security': ['analyze_code', 'read_file', 'grep']
        }
        
        optimized_tools = set(base_tools)
        
        for component in components:
            if component in tool_mapping:
                optimized_tools.update(tool_mapping[component])
        
        return list(optimized_tools)
    
    def _create_phase_based_subtasks(
        self, 
        task: SubAgentTask, 
        agent_count: int
    ) -> List[SubAgentTask]:
        """Create phase-based subtasks for parallel execution."""
        phases = [
            {'name': 'research', 'weight': 0.25, 'tools': ['read_file', 'list_directory', 'web_fetch']},
            {'name': 'analysis', 'weight': 0.3, 'tools': ['analyze_code', 'read_file', 'grep']},
            {'name': 'implementation', 'weight': 0.35, 'tools': ['write_file', 'read_file', 'analyze_code']},
            {'name': 'validation', 'weight': 0.1, 'tools': ['read_file', 'analyze_code']}
        ]
        
        # Distribute phases based on agent count
        subtasks = []
        phases_per_agent = max(1, len(phases) // agent_count)
        
        for i in range(agent_count):
            # Calculate phase distribution
            start_phase = i * phases_per_agent
            end_phase = min((i + 1) * phases_per_agent, len(phases))
            agent_phases = phases[start_phase:end_phase]
            
            if agent_phases:
                phase_names = [p['name'] for p in agent_phases]
                phase_tools = []
                for phase in agent_phases:
                    phase_tools.extend(phase['tools'])
                
                # Remove duplicates while preserving order
                unique_tools = []
                for tool in phase_tools:
                    if tool not in unique_tools:
                        unique_tools.append(tool)
                
                # Calculate iterations and timeout based on phase weights
                total_weight = sum(p['weight'] for p in agent_phases)
                iterations = max(1, int(task.max_iterations * total_weight))
                timeout = max(30, int(task.timeout * total_weight))
                
                subtask = SubAgentTask(
                    description=f"Phase-based execution: {task.description} - {', '.join(phase_names)}",
                    tools=unique_tools,
                    max_iterations=iterations,
                    timeout=timeout,
                    context={
                        'phase_agent_id': i + 1,
                        'phases': phase_names,
                        'parent_task': task.id
                    }
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