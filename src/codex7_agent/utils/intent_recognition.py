"""
Intent recognition system with multi-level semantic analysis.
This module implements the sophisticated intent recognition mechanism described in the technical framework.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class TaskComplexity(Enum):
    """Task complexity levels."""
    HIGHEST = "highest"
    MIDDLE = "middle"
    BASIC = "basic"
    NONE = "none"


class TaskType(Enum):
    """Types of tasks the agent can handle."""
    SEARCH = "search"
    ANALYZE = "analyze"
    CREATE = "create"
    MODIFY = "modify"
    DEBUG = "debug"
    EXECUTE = "execute"
    HELP = "help"
    EXIT = "exit"
    CLEAR = "clear"
    MEMORY_SAVE = "memory_save"


@dataclass
class IntentRecognitionResult:
    """Result of intent recognition process."""
    task_type: TaskType
    intent: str
    complexity: TaskComplexity
    parameters: Dict[str, Any]
    recommended_tools: List[str]
    execution_strategy: str
    confidence: float
    original_message: str


class IntentRecognizer:
    """Multi-level intent recognition system with LLM integration."""
    
    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.llm_config = llm_config or {}
        self.complexity_patterns = self._load_complexity_patterns()
        self.command_patterns = self._load_command_patterns()
        
    def _load_complexity_patterns(self) -> Dict[TaskComplexity, List[str]]:
        """Load complexity assessment patterns."""
        return {
            TaskComplexity.HIGHEST: [
                r'\b(complex|advanced|multi-file|architecture|refactor|redesign)\b',
                r'\b(system|framework|library|module)\b',
                r'\b(integration|deployment|pipeline)\b',
                r'\b(performance|optimization|scalability)\b'
            ],
            TaskComplexity.MIDDLE: [
                r'\b(analyze|understand|explain|document)\b',
                r'\b(refactor|improve|enhance)\b',
                r'\b(implementation|feature|function)\b',
                r'\b(test|validate|verify)\b'
            ],
            TaskComplexity.BASIC: [
                r'\b(simple|quick|basic|minor)\b',
                r'\b(fix|correct|update)\b',
                r'\b(add|remove|change)\b',
                r'\b(check|view|list)\b'
            ]
        }
    
    def _load_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load command recognition patterns."""
        return {
            "exit": {
                "pattern": r'^(exit|quit|bye)$',
                "task_type": TaskType.EXIT,
                "complexity": TaskComplexity.NONE
            },
            "help": {
                "pattern": r'^(help|\?|assist)$',
                "task_type": TaskType.HELP,
                "complexity": TaskComplexity.NONE
            },
            "clear": {
                "pattern": r'^(clear|cls)$',
                "task_type": TaskType.CLEAR,
                "complexity": TaskComplexity.NONE
            },
            "memory_save": {
                "pattern": r'^#',
                "task_type": TaskType.MEMORY_SAVE,
                "complexity": TaskComplexity.BASIC
            },
            "search": {
                "pattern": r'\b(search|find|lookup)\b',
                "task_type": TaskType.SEARCH,
                "complexity": TaskComplexity.BASIC
            },
            "analyze": {
                "pattern": r'\b(analyze|understand|explain|document)\b',
                "task_type": TaskType.ANALYZE,
                "complexity": TaskComplexity.MIDDLE
            },
            "create": {
                "pattern": r'\b(create|implement|build|develop)\b',
                "task_type": TaskType.CREATE,
                "complexity": TaskComplexity.MIDDLE
            },
            "modify": {
                "pattern": r'\b(modify|edit|change|update|refactor)\b',
                "task_type": TaskType.MODIFY,
                "complexity": TaskComplexity.MIDDLE
            },
            "debug": {
                "pattern": r'\b(debug|fix|error|issue|problem)\b',
                "task_type": TaskType.DEBUG,
                "complexity": TaskComplexity.MIDDLE
            },
            "execute": {
                "pattern": r'\b(run|execute|test|deploy)\b',
                "task_type": TaskType.EXECUTE,
                "complexity": TaskComplexity.BASIC
            }
        }
    
    def recognize_intent(self, message: str, context: Dict[str, Any] = None) -> IntentRecognitionResult:
        """
        Main intent recognition function implementing multi-level semantic analysis.
        
        Args:
            message: The user's input message
            context: Additional context including conversation history
            
        Returns:
            IntentRecognitionResult with structured task understanding
        """
        if context is None:
            context = {}
            
        self.logger.debug(f"Recognizing intent for: {message}")
        
        # Step 1: Content extraction and normalization
        normalized_message = self._normalize_message(message)
        
        # Step 2: Command and pattern recognition
        command_result = self._recognize_command(normalized_message)
        
        # Step 3: LLM-driven semantic understanding (if LLM config available)
        semantic_result = self._semantic_understanding(normalized_message, context)
        
        # Step 4: Task complexity assessment
        complexity = self._assess_complexity(normalized_message, context)
        
        # Step 5: Generate structured understanding result
        return self._generate_structured_result(
            message, command_result, semantic_result, complexity, context
        )
    
    def _normalize_message(self, message: str) -> str:
        """Normalize message content for consistent processing."""
        # Remove extra whitespace and normalize case
        normalized = re.sub(r'\s+', ' ', message.strip().lower())
        return normalized
    
    def _recognize_command(self, message: str) -> Optional[Dict[str, Any]]:
        """Recognize special commands and patterns."""
        for command_name, command_info in self.command_patterns.items():
            if re.search(command_info["pattern"], message, re.IGNORECASE):
                return {
                    "task_type": command_info["task_type"],
                    "complexity": command_info["complexity"],
                    "command": command_name
                }
        return None
    
    def _semantic_understanding(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM-driven semantic understanding."""
        # For now, return basic semantic analysis without LLM
        # In future, integrate with LLM for better understanding
        
        semantic_analysis = {
            "keywords": self._extract_keywords(message),
            "intent_summary": self._generate_intent_summary(message),
            "context_relevance": self._assess_context_relevance(message, context),
            "task_dependencies": self._identify_dependencies(message, context)
        }
        
        return semantic_analysis
    
    def _assess_complexity(self, message: str, context: Dict[str, Any]) -> TaskComplexity:
        """Assess task complexity using pattern matching."""
        complexity_scores = {}
        
        for complexity, patterns in self.complexity_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 1
            complexity_scores[complexity] = score
        
        # Find the highest complexity level with matches
        for complexity in [TaskComplexity.HIGHEST, TaskComplexity.MIDDLE, TaskComplexity.BASIC]:
            if complexity_scores.get(complexity, 0) > 0:
                return complexity
        
        return TaskComplexity.BASIC
    
    def _extract_keywords(self, message: str) -> List[str]:
        """Extract key technical terms from the message."""
        technical_terms = [
            "function", "class", "method", "variable", "file", "directory",
            "import", "export", "return", "async", "await", "test", "debug",
            "refactor", "optimize", "search", "analyze", "create", "modify"
        ]
        
        found_terms = []
        for term in technical_terms:
            if term in message.lower():
                found_terms.append(term)
        
        return found_terms
    
    def _generate_intent_summary(self, message: str) -> str:
        """Generate a brief summary of the user's intent."""
        if len(message) < 50:
            return message
        else:
            return message[:100] + "..."
    
    def _assess_context_relevance(self, message: str, context: Dict[str, Any]) -> float:
        """Assess relevance to current context."""
        if not context or 'messages' not in context:
            return 0.0
        
        # Simple keyword overlap scoring
        message_words = set(message.lower().split())
        context_words = set()
        
        for msg in context.get('messages', []):
            if isinstance(msg, dict) and 'content' in msg:
                context_words.update(msg['content'].lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = len(message_words.intersection(context_words))
        return min(overlap / len(context_words), 1.0)
    
    def _identify_dependencies(self, message: str, context: Dict[str, Any]) -> List[str]:
        """Identify task dependencies from context."""
        dependencies = []
        
        # Look for file references
        file_pattern = r'\b(\w+\.\w+)\b'
        files = re.findall(file_pattern, message)
        dependencies.extend(files)
        
        return dependencies
    
    def _generate_structured_result(
        self, 
        original_message: str,
        command_result: Optional[Dict[str, Any]],
        semantic_result: Dict[str, Any],
        complexity: TaskComplexity,
        context: Dict[str, Any]
    ) -> IntentRecognitionResult:
        """Generate the final structured understanding result."""
        
        # Determine task type
        if command_result:
            task_type = command_result["task_type"]
            complexity = command_result["complexity"]
        else:
            # Infer from semantic analysis
            keywords = semantic_result["keywords"]
            if "search" in keywords or "find" in original_message.lower():
                task_type = TaskType.SEARCH
            elif "create" in keywords or "implement" in original_message.lower():
                task_type = TaskType.CREATE
            elif "debug" in keywords or "fix" in original_message.lower():
                task_type = TaskType.DEBUG
            elif "analyze" in keywords:
                task_type = TaskType.ANALYZE
            else:
                task_type = TaskType.SEARCH  # Default
        
        # Map task type to recommended tools
        tool_mapping = {
            TaskType.SEARCH: ["LS", "Glob", "Grep", "Read"],
            TaskType.ANALYZE: ["Read", "Grep", "Task"],
            TaskType.CREATE: ["Write", "Edit", "Bash"],
            TaskType.MODIFY: ["Read", "Edit", "Bash"],
            TaskType.DEBUG: ["Read", "Bash", "Task"],
            TaskType.EXECUTE: ["Bash", "Task"],
            TaskType.HELP: ["Read"],
            TaskType.CLEAR: [],
            TaskType.EXIT: [],
            TaskType.MEMORY_SAVE: ["Edit"]
        }
        
        # Generate execution strategy
        if complexity == TaskComplexity.HIGHEST:
            strategy = "use_subagent_with_parallel_processing"
        elif complexity == TaskComplexity.MIDDLE:
            strategy = "sequential_tool_execution_with_validation"
        else:
            strategy = "direct_tool_execution"
        
        return IntentRecognitionResult(
            task_type=task_type,
            intent=semantic_result["intent_summary"],
            complexity=complexity,
            parameters={
                "keywords": semantic_result["keywords"],
                "context_relevance": semantic_result["context_relevance"],
                "dependencies": semantic_result["task_dependencies"]
            },
            recommended_tools=tool_mapping[task_type],
            execution_strategy=strategy,
            confidence=0.8,
            original_message=original_message
        )
        
    def _load_complexity_patterns(self) -> Dict[TaskComplexity, List[str]]:
        """Load complexity assessment patterns."""
        return {
            TaskComplexity.HIGHEST: [
                r'\b(complex|advanced|multi-file|architecture|refactor|redesign)\b',
                r'\b(system|framework|library|module)\b',
                r'\b(integration|deployment|pipeline)\b',
                r'\b(performance|optimization|scalability)\b'
            ],
            TaskComplexity.MIDDLE: [
                r'\b(analyze|understand|explain|document)\b',
                r'\b(refactor|improve|enhance)\b',
                r'\b(implementation|feature|function)\b',
                r'\b(test|validate|verify)\b'
            ],
            TaskComplexity.BASIC: [
                r'\b(simple|quick|basic|minor)\b',
                r'\b(fix|correct|update)\b',
                r'\b(add|remove|change)\b',
                r'\b(check|view|list)\b'
            ]
        }
    
    def _load_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load command recognition patterns."""
        return {
            "exit": {
                "pattern": r'^(exit|quit|bye)$',
                "task_type": TaskType.EXIT,
                "complexity": TaskComplexity.NONE
            },
            "help": {
                "pattern": r'^(help|\?|assist)$',
                "task_type": TaskType.HELP,
                "complexity": TaskComplexity.NONE
            },
            "clear": {
                "pattern": r'^(clear|cls)$',
                "task_type": TaskType.CLEAR,
                "complexity": TaskComplexity.NONE
            },
            "memory_save": {
                "pattern": r'^#',
                "task_type": TaskType.MEMORY_SAVE,
                "complexity": TaskComplexity.BASIC
            },
            "search": {
                "pattern": r'\b(search|find|lookup)\b',
                "task_type": TaskType.SEARCH,
                "complexity": TaskComplexity.BASIC
            },
            "analyze": {
                "pattern": r'\b(analyze|understand|explain|document)\b',
                "task_type": TaskType.ANALYZE,
                "complexity": TaskComplexity.MIDDLE
            },
            "create": {
                "pattern": r'\b(create|implement|build|develop)\b',
                "task_type": TaskType.CREATE,
                "complexity": TaskComplexity.MIDDLE
            },
            "modify": {
                "pattern": r'\b(modify|edit|change|update|refactor)\b',
                "task_type": TaskType.MODIFY,
                "complexity": TaskComplexity.MIDDLE
            },
            "debug": {
                "pattern": r'\b(debug|fix|error|issue|problem)\b',
                "task_type": TaskType.DEBUG,
                "complexity": TaskComplexity.MIDDLE
            },
            "execute": {
                "pattern": r'\b(run|execute|test|deploy)\b',
                "task_type": TaskType.EXECUTE,
                "complexity": TaskComplexity.BASIC
            }
        }
    
    def recognize_intent(self, message: str, context: Dict[str, Any] = None) -> IntentRecognitionResult:
        """
        Main intent recognition function implementing multi-level semantic analysis.
        
        Args:
            message: The user's input message
            context: Additional context including conversation history
            
        Returns:
            IntentRecognitionResult with structured task understanding
        """
        if context is None:
            context = {}
            
        self.logger.debug(f"Recognizing intent for: {message}")
        
        # Step 1: Content extraction and normalization
        normalized_message = self._normalize_message(message)
        
        # Step 2: Command and pattern recognition
        command_result = self._recognize_command(normalized_message)
        
        # Step 3: LLM-driven semantic understanding
        semantic_result = self._semantic_understanding(normalized_message, context)
        
        # Step 4: Task complexity assessment
        complexity = self._assess_complexity(normalized_message, context)
        
        # Step 5: Generate structured understanding result
        return self._generate_structured_result(
            message, command_result, semantic_result, complexity, context
        )
    
    def _normalize_message(self, message: str) -> str:
        """Normalize message content for consistent processing."""
        # Remove extra whitespace and normalize case
        normalized = re.sub(r'\s+', ' ', message.strip().lower())
        return normalized
    
    def _recognize_command(self, message: str) -> Optional[Dict[str, Any]]:
        """Recognize special commands and patterns."""
        for command_name, command_info in self.command_patterns.items():
            if re.search(command_info["pattern"], message, re.IGNORECASE):
                return {
                    "task_type": command_info["task_type"],
                    "complexity": command_info["complexity"],
                    "command": command_name
                }
        return None
    
    def _semantic_understanding(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """LLM-driven semantic understanding."""
        # TODO: Implement actual LLM integration for semantic understanding
        # For now, return basic semantic analysis
        
        semantic_analysis = {
            "keywords": self._extract_keywords(message),
            "intent_summary": self._generate_intent_summary(message),
            "context_relevance": self._assess_context_relevance(message, context),
            "task_dependencies": self._identify_dependencies(message, context)
        }
        
        return semantic_analysis
    
    def _assess_complexity(self, message: str, context: Dict[str, Any]) -> TaskComplexity:
        """Assess task complexity using pattern matching."""
        complexity_scores = {}
        
        for complexity, patterns in self.complexity_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    score += 1
            complexity_scores[complexity] = score
        
        # Find the highest complexity level with matches
        for complexity in [TaskComplexity.HIGHEST, TaskComplexity.MIDDLE, TaskComplexity.BASIC]:
            if complexity_scores.get(complexity, 0) > 0:
                return complexity
        
        return TaskComplexity.BASIC
    
    def _extract_keywords(self, message: str) -> List[str]:
        """Extract key technical terms from the message."""
        # TODO: Use more sophisticated NLP for keyword extraction
        technical_terms = [
            "function", "class", "method", "variable", "file", "directory",
            "import", "export", "return", "async", "await", "test", "debug",
            "refactor", "optimize", "search", "analyze", "create", "modify"
        ]
        
        found_terms = []
        for term in technical_terms:
            if term in message.lower():
                found_terms.append(term)
        
        return found_terms
    
    def _generate_intent_summary(self, message: str) -> str:
        """Generate a brief summary of the user's intent."""
        # TODO: Use LLM for better intent summarization
        if len(message) < 50:
            return message
        else:
            return message[:100] + "..."
    
    def _assess_context_relevance(self, message: str, context: Dict[str, Any]) -> float:
        """Assess relevance to current context."""
        # TODO: Implement sophisticated context relevance scoring
        if not context or 'messages' not in context:
            return 0.0
        
        # Simple keyword overlap scoring
        message_words = set(message.lower().split())
        context_words = set()
        
        for msg in context.get('messages', []):
            if isinstance(msg, dict) and 'content' in msg:
                context_words.update(msg['content'].lower().split())
        
        if not context_words:
            return 0.0
        
        overlap = len(message_words.intersection(context_words))
        return min(overlap / len(context_words), 1.0)
    
    def _identify_dependencies(self, message: str, context: Dict[str, Any]) -> List[str]:
        """Identify task dependencies from context."""
        # TODO: Implement sophisticated dependency identification
        dependencies = []
        
        # Look for file references
        file_pattern = r'\b(\w+\.\w+)\b'
        files = re.findall(file_pattern, message)
        dependencies.extend(files)
        
        return dependencies
    
    def _generate_structured_result(
        self, 
        original_message: str,
        command_result: Optional[Dict[str, Any]],
        semantic_result: Dict[str, Any],
        complexity: TaskComplexity,
        context: Dict[str, Any]
    ) -> IntentRecognitionResult:
        """Generate the final structured understanding result."""
        
        # Determine task type
        if command_result:
            task_type = command_result["task_type"]
            complexity = command_result["complexity"]
        else:
            # Infer from semantic analysis
            keywords = semantic_result["keywords"]
            if "search" in keywords or "find" in original_message.lower():
                task_type = TaskType.SEARCH
            elif "create" in keywords or "implement" in original_message.lower():
                task_type = TaskType.CREATE
            elif "debug" in keywords or "fix" in original_message.lower():
                task_type = TaskType.DEBUG
            elif "analyze" in keywords:
                task_type = TaskType.ANALYZE
            else:
                task_type = TaskType.SEARCH  # Default
        
        # Map task type to recommended tools
        tool_mapping = {
            TaskType.SEARCH: ["LS", "Glob", "Grep", "Read"],
            TaskType.ANALYZE: ["Read", "Grep", "Task"],
            TaskType.CREATE: ["Write", "Edit", "Bash"],
            TaskType.MODIFY: ["Read", "Edit", "Bash"],
            TaskType.DEBUG: ["Read", "Bash", "Task"],
            TaskType.EXECUTE: ["Bash", "Task"],
            TaskType.HELP: ["Read"],
            TaskType.CLEAR: [],
            TaskType.EXIT: [],
            TaskType.MEMORY_SAVE: ["Edit"]
        }
        
        # Generate execution strategy
        if complexity == TaskComplexity.HIGHEST:
            strategy = "use_subagent_with_parallel_processing"
        elif complexity == TaskComplexity.MIDDLE:
            strategy = "sequential_tool_execution_with_validation"
        else:
            strategy = "direct_tool_execution"
        
        return IntentRecognitionResult(
            task_type=task_type,
            intent=semantic_result["intent_summary"],
            complexity=complexity,
            parameters={
                "keywords": semantic_result["keywords"],
                "context_relevance": semantic_result["context_relevance"],
                "dependencies": semantic_result["task_dependencies"]
            },
            recommended_tools=tool_mapping[task_type],
            execution_strategy=strategy,
            confidence=0.8,  # TODO: Implement actual confidence scoring
            original_message=original_message
        )