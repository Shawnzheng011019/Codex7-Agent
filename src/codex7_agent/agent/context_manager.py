"""
Context management with 3-layer memory architecture.
Implements short-term, mid-term, and long-term memory systems.
"""

import os
import json
import time
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .loop import Message, MessageType


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""
    id: str
    content: str
    timestamp: float
    metadata: Dict[str, Any]
    relevance_score: float = 0.0
    memory_type: str = "short_term"


class ShortTermMemory:
    """Short-term memory for current session messages."""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.messages: List[Message] = []
        self.message_map: Dict[str, Message] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_message(self, message: Message) -> None:
        """Add a message to short-term memory."""
        self.messages.append(message)
        self.message_map[message.id] = message
        
        # Maintain size limit
        if len(self.messages) > self.max_size:
            removed = self.messages.pop(0)
            self.message_map.pop(removed.id, None)
    
    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get messages from short-term memory."""
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()
    
    def clear(self) -> None:
        """Clear short-term memory."""
        self.messages.clear()
        self.message_map.clear()
    
    def get_context_size(self) -> int:
        """Get current context size in number of messages."""
        return len(self.messages)


class MidTermMemory:
    """Mid-term memory with intelligent context compression."""
    
    def __init__(self, compression_threshold: float = 0.92):
        self.compression_threshold = compression_threshold
        self.compressed_context: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    def should_compress(self, context_size: int, max_context_size: int) -> bool:
        """Check if context compression should be triggered."""
        usage_ratio = context_size / max_context_size
        return usage_ratio >= self.compression_threshold
    
    def compress_context(self, messages: List[Message], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Eight-part intelligent context compression.
        
        Returns a structured compressed context with:
        1. Main user request and intent
        2. Key technical concepts
        3. Relevant file locations
        4. Problems and solutions
        5. Problem-solving approach, method, results
        6. Complete timeline of user messages
        7. Pending tasks and current work summary
        8. Next steps structure
        """
        if not messages:
            return {}
        
        compressed = {
            "main_intent": self._extract_main_intent(messages),
            "key_concepts": self._extract_key_concepts(messages),
            "file_locations": self._extract_file_locations(messages),
            "problems_solutions": self._extract_problems_solutions(messages),
            "approach_method_results": self._extract_approach_method_results(messages),
            "timeline": self._create_timeline(messages),
            "pending_tasks": self._extract_pending_tasks(messages),
            "next_steps": self._generate_next_steps(messages)
        }
        
        self.logger.debug("Context compressed successfully")
        return compressed
    
    def _extract_main_intent(self, messages: List[Message]) -> str:
        """Extract the main user request and intent."""
        user_messages = [msg for msg in messages if msg.type == MessageType.USER]
        if not user_messages:
            return "No user messages found"
        
        # Get the most recent user message
        latest_message = user_messages[-1]
        return latest_message.content[:200]  # Truncate if too long
    
    def _extract_key_concepts(self, messages: List[Message]) -> List[str]:
        """Extract key technical concepts using TF-IDF and keyword extraction."""
        all_content = " ".join([msg.content for msg in messages])
        
        # TF-IDF based concept extraction with technical term weighting
        import re
        from collections import Counter
        
        # Tokenize and clean
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', all_content.lower())
        
        # Technical term patterns with weights
        technical_patterns = {
            # Programming constructs
            'function|method|def|class|import|return|async|await': 3.0,
            'variable|const|let|var|static|public|private': 2.5,
            'file|directory|path|folder|module|package|library': 2.0,
            'test|debug|assert|mock|stub|fixture': 2.5,
            'refactor|optimize|improve|enhance|fix|resolve': 2.0,
            'configuration|config|setting|parameter|option': 1.8,
            'dependency|requirement|install|setup|deploy': 1.8,
            'framework|library|tool|utility|service|api': 2.0,
            'database|query|sql|nosql|table|schema': 2.2,
            'algorithm|data structure|pattern|design': 2.5
        }
        
        # Extract technical terms with TF-IDF weighting
        word_freq = Counter(words)
        concepts = []
        
        for pattern, weight in technical_patterns.items():
            matches = re.findall(pattern, all_content.lower())
            if matches:
                # Calculate weighted score
                score = len(matches) * weight
                concepts.append((matches[0], score))
        
        # Also extract camelCase and snake_case identifiers
        identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', all_content.lower())
        camel_case = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', all_content)
        
        # Add high-frequency identifiers
        identifier_freq = Counter(identifiers + camel_case)
        for identifier, freq in identifier_freq.most_common(10):
            if len(identifier) > 3 and identifier not in ['this', 'that', 'with']:
                concepts.append((identifier, freq * 1.5))
        
        # Sort by score and return unique concepts
        concept_scores = {}
        for concept, score in concepts:
            concept_scores[concept] = concept_scores.get(concept, 0) + score
        
        sorted_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, score in sorted_concepts[:15]]
    
    def _extract_file_locations(self, messages: List[Message]) -> List[str]:
        """Extract relevant file locations from messages."""
        file_pattern = r'\b([\w/\\.-]+\.(py|js|ts|java|cpp|c|go|rs|json|yaml|yml|md))\b'
        
        all_content = " ".join([msg.content for msg in messages])
        files = re.findall(file_pattern, all_content)
        
        return [file[0] for file in files]
    
    def _extract_problems_solutions(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Extract problems and their solutions using pattern matching and semantic analysis."""
        problems_solutions = []
        
        # Multi-pattern problem detection
        problem_patterns = [
            r'(?i)(error|exception|bug|issue|problem|fail|crash|broken|wrong|incorrect)',
            r'(?i)(cannot|can\'t|unable|fails?|failed|failing)',
            r'(?i)(traceback|stacktrace|exception.*trace)',
            r'(?i)(missing|not found|undefined|reference.*error)',
            r'(?i)(syntax.*error|type.*error|value.*error|attribute.*error)'
        ]
        
        solution_patterns = [
            r'(?i)(fix|solve|solution|resolved|correct|repair|patch|workaround)',
            r'(?i)(should|need to|must|try|change|update|modify|add|remove)',
            r'(?i)(instead|replace|alternative|option|workaround)'
        ]
        
        # Track problem-solution pairs across message flow
        current_problem = None
        for i, msg in enumerate(messages):
            content = msg.content
            
            # Detect problem
            for pattern in problem_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    # Extract problem context (context window around error)
                    lines = content.split('\n')
                    problem_lines = []
                    for j, line in enumerate(lines):
                        if any(re.search(p, line, re.IGNORECASE) for p in problem_patterns):
                            # Include context: 2 lines before and after
                            start = max(0, j-2)
                            end = min(len(lines), j+3)
                            problem_lines.extend(lines[start:end])
                    
                    problem_context = '\n'.join(problem_lines)[:200]
                    current_problem = {
                        "problem": problem_context,
                        "timestamp": msg.timestamp,
                        "message_index": i
                    }
                    break
            
            # Detect solution (look for solutions after problems)
            if current_problem and i > current_problem.get("message_index", -1):
                for pattern in solution_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Extract solution context
                        solution_context = content[:150]
                        problems_solutions.append({
                            "problem": current_problem["problem"],
                            "solution": solution_context,
                            "problem_timestamp": current_problem["timestamp"],
                            "solution_timestamp": msg.timestamp
                        })
                        current_problem = None
                        break
        
        # Handle orphaned problems (problems without detected solutions)
        if current_problem:
            problems_solutions.append({
                "problem": current_problem["problem"],
                "solution": "Under investigation",
                "problem_timestamp": current_problem["timestamp"],
                "solution_timestamp": None
            })
        
        return problems_solutions
    
    def _extract_approach_method_results(self, messages: List[Message]) -> Dict[str, str]:
        """Extract problem-solving approach, method, and results using semantic analysis."""
        
        # Define action verbs for approach detection
        approach_verbs = [
            'analyze', 'examine', 'investigate', 'explore', 'understand',
            'identify', 'determine', 'assess', 'evaluate', 'review'
        ]
        
        # Define method keywords
        method_keywords = [
            'implement', 'create', 'build', 'develop', 'write', 'code',
            'refactor', 'optimize', 'test', 'validate', 'debug',
            'use', 'apply', 'utilize', 'deploy', 'configure'
        ]
        
        # Define result indicators
        result_indicators = [
            'completed', 'finished', 'done', 'success', 'working',
            'fixed', 'resolved', 'solved', 'achieved', 'implemented',
            'tested', 'validated', 'verified', 'confirmed'
        ]
        
        all_content = " ".join([msg.content.lower() for msg in messages])
        
        # Extract approach from user intent and initial analysis
        approach = "Systematic analysis and implementation"
        user_messages = [msg for msg in messages if msg.type == MessageType.USER]
        if user_messages:
            first_user_msg = user_messages[0].content.lower()
            approach_words = [verb for verb in approach_verbs if verb in first_user_msg]
            if approach_words:
                approach = f"{', '.join(approach_words[:3])} of the problem"
        
        # Extract method from tool usage and implementation details
        method = "Iterative development with testing"
        method_words = [kw for kw in method_keywords if kw in all_content]
        if method_words:
            unique_methods = list(set(method_words))[:5]
            method = f"{', '.join(unique_methods)} approach"
        
        # Extract results from completion indicators
        results = "Work in progress"
        result_words = [ind for ind in result_indicators if ind in all_content]
        if result_words:
            results = f"Successfully {', '.join(result_words[:3])}"
        
        # Check for specific outcomes
        if "error" in all_content and "fixed" in all_content:
            results = "Issues identified and resolved"
        elif "test" in all_content and "pass" in all_content:
            results = "Implementation validated through testing"
        elif "optimize" in all_content and "improve" in all_content:
            results = "Performance improvements achieved"
        
        return {
            "approach": approach,
            "method": method,
            "results": results
        }
    
    def _create_timeline(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Create a timeline of user messages."""
        timeline = []
        
        for msg in messages:
            timeline.append({
                "type": msg.type.value,
                "content": msg.content[:100],  # Truncate for brevity
                "timestamp": msg.timestamp
            })
        
        return timeline
    
    def _extract_pending_tasks(self, messages: List[Message]) -> List[str]:
        """Extract pending tasks using TODO detection and semantic analysis."""
        pending_tasks = []
        
        # TODO pattern detection
        todo_patterns = [
            r'(?i)(todo|to-do|task|action.*item|next.*step|pending|remaining)',
            r'(?i)(need.*to|should|must|have.*to|require.*to)',
            r'(?i)(implement|create|add|fix|resolve|update|refactor|optimize)',
            r'(?i)(later|next|after|then|subsequent|follow.*up)'
        ]
        
        # Action item extraction
        action_patterns = [
            r'(?i)(implement.*function|create.*class|add.*feature|fix.*bug)',
            r'(?i)(write.*test|update.*documentation|refactor.*code)',
            r'(?i)(check.*dependency|review.*code|validate.*solution)',
            r'(?i)(deploy.*changes|merge.*branch|release.*version)'
        ]
        
        all_content = "\n".join([msg.content for msg in messages])
        lines = all_content.split('\n')
        
        # Extract TODO items and action items
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for TODO markers
            todo_match = re.search(r'(?i)(todo|to-do|task):?\s*(.+)', line)
            if todo_match:
                task = todo_match.group(2).strip()
                if len(task) > 5:
                    pending_tasks.append(task)
                continue
            
            # Check for action patterns
            for pattern in action_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Extract meaningful action
                    action = re.sub(r'^[-*\s]+', '', line)  # Remove list markers
                    action = re.sub(r'(?i)(todo|task|action|item):?', '', action).strip()
                    if len(action) > 10 and action not in pending_tasks:
                        pending_tasks.append(action)
                    break
        
        # Extract from conversation flow (implicit tasks)
        # Look for incomplete actions or pending work
        incomplete_indicators = [
            r'(?i)(working.*on|currently|in.*progress|not.*yet|still.*need)',
            r'(?i)(plan.*to|will.*need|going.*to|intend.*to)'
        ]
        
        for msg in messages[-5:]:  # Look at recent messages for active tasks
            content = msg.content.lower()
            for pattern in incomplete_indicators:
                if re.search(pattern, content):
                    # Extract the action part
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        if re.search(pattern, sentence):
                            task = sentence.strip()
                            if len(task) > 15 and task not in pending_tasks:
                                pending_tasks.append(task)
                    break
        
        # Remove duplicates and limit to most relevant
        unique_tasks = list(dict.fromkeys(pending_tasks))
        return unique_tasks[:10]
    
    def _generate_next_steps(self, messages: List[Message]) -> List[str]:
        """Generate intelligent next steps based on current context."""
        if not messages:
            return ["Start with user request analysis"]
        
        next_steps = []
        all_content = " ".join([msg.content.lower() for msg in messages])
        
        # Analyze current state and generate contextually relevant steps
        
        # 1. Check for incomplete implementations
        if any(word in all_content for word in ['implement', 'create', 'build', 'develop']):
            if 'test' not in all_content or 'validate' not in all_content:
                next_steps.append("Write comprehensive tests for the implementation")
        
        # 2. Check for errors that need resolution
        error_indicators = ['error', 'exception', 'bug', 'issue', 'failed']
        if any(indicator in all_content for indicator in error_indicators):
            next_steps.append("Debug and resolve identified issues")
            next_steps.append("Validate fix with targeted tests")
        
        # 3. Check for optimization opportunities
        optimization_keywords = ['optimize', 'improve', 'performance', 'efficiency', 'refactor']
        if any(keyword in all_content for keyword in optimization_keywords):
            next_steps.append("Profile current implementation for bottlenecks")
            next_steps.append("Apply targeted optimizations based on profiling results")
        
        # 4. Check for documentation needs
        doc_keywords = ['document', 'explain', 'readme', 'comment', 'guide']
        if any(keyword in all_content for keyword in doc_keywords):
            next_steps.append("Update documentation to reflect changes")
        
        # 5. Check for deployment considerations
        deploy_keywords = ['deploy', 'release', 'production', 'build', 'package']
        if any(keyword in all_content for keyword in deploy_keywords):
            next_steps.append("Prepare deployment configuration and testing")
        
        # 6. Check for code review needs
        review_keywords = ['review', 'check', 'validate', 'verify']
        if any(keyword in all_content for keyword in review_keywords):
            next_steps.append("Conduct thorough code review")
        
        # 7. Check for dependency management
        dep_keywords = ['dependency', 'import', 'require', 'install', 'setup']
        if any(keyword in all_content for keyword in dep_keywords):
            next_steps.append("Verify and update dependency requirements")
        
        # 8. Generate context-specific steps based on file types
        file_types = re.findall(r'\.(py|js|ts|java|cpp|c|go|rs|json|yaml|yml|md)', all_content)
        if file_types:
            unique_types = list(set(file_types))
            if 'py' in unique_types:
                next_steps.append("Ensure Python code follows PEP 8 conventions")
            if any(lang in unique_types for lang in ['js', 'ts']):
                next_steps.append("Verify JavaScript/TypeScript linting and formatting")
            if 'json' in unique_types or 'yaml' in unique_types:
                next_steps.append("Validate configuration file syntax")
        
        # 9. Check for testing gaps
        test_keywords = ['test', 'coverage', 'assert', 'validate']
        if not any(keyword in all_content for keyword in test_keywords):
            next_steps.append("Add comprehensive test coverage")
        
        # 10. Add general best practices
        general_steps = [
            "Review implementation against requirements",
            "Ensure proper error handling is in place",
            "Verify performance meets expectations",
            "Document any breaking changes"
        ]
        
        # Combine and deduplicate
        all_steps = next_steps + general_steps
        
        # Prioritize based on context relevance
        prioritized_steps = []
        for step in all_steps:
            if step not in prioritized_steps:
                prioritized_steps.append(step)
                if len(prioritized_steps) >= 5:  # Limit to top 5 most relevant
                    break
        
        return prioritized_steps


class LongTermMemory:
    """Long-term memory for persistent storage."""
    
    def __init__(self, storage_path: str = ".claude_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> None:
        """Save user preferences to long-term memory."""
        file_path = self.storage_path / f"{user_id}_preferences.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(preferences, f, indent=2)
            self.logger.debug(f"Saved preferences for user {user_id}")
        except Exception as e:
            self.logger.error(f"Failed to save preferences: {e}")
    
    def load_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Load user preferences from long-term memory."""
        file_path = self.storage_path / f"{user_id}_preferences.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load preferences: {e}")
        
        return {}
    
    def log_conversation(self, session_id: str, messages: List[Message]) -> None:
        """Log conversation to long-term memory."""
        file_path = self.storage_path / f"{session_id}_conversation.json"
        
        try:
            conversation_data = {
                "session_id": session_id,
                "timestamp": time.time(),
                "messages": [
                    {
                        "id": msg.id,
                        "type": msg.type.value,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "parent_uuid": msg.parent_uuid,
                        "metadata": msg.metadata
                    }
                    for msg in messages
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(conversation_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to log conversation: {e}")
    
    def get_conversation_history(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve conversation history from long-term memory."""
        file_path = self.storage_path / f"{session_id}_conversation.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return data.get("messages", [])
        except Exception as e:
            self.logger.error(f"Failed to load conversation: {e}")
        
        return None
    
    def save_task_state(self, task_id: str, state: Dict[str, Any]) -> None:
        """Save task state for resumption."""
        file_path = self.storage_path / f"{task_id}_state.json"
        
        try:
            state_data = {
                "task_id": task_id,
                "timestamp": time.time(),
                "state": state
            }
            
            with open(file_path, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save task state: {e}")
    
    def load_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load task state for resumption."""
        file_path = self.storage_path / f"{task_id}_state.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return data.get("state")
        except Exception as e:
            self.logger.error(f"Failed to load task state: {e}")
        
        return None


class ContextManager:
    """Manages the 3-layer memory architecture."""
    
    def __init__(
        self,
        short_term_max_size: int = 50,
        compression_threshold: float = 0.92,
        storage_path: str = ".claude_memory"
    ):
        self.short_term = ShortTermMemory(short_term_max_size)
        self.mid_term = MidTermMemory(compression_threshold)
        self.long_term = LongTermMemory(storage_path)
        self.logger = logging.getLogger(__name__)
    
    def add_message(self, message: Message) -> None:
        """Add a message to the memory system."""
        self.short_term.add_message(message)
    
    def get_context_for_llm(self, max_context_size: int = 100) -> List[Message]:
        """Get context for LLM consumption, with intelligent compression if needed."""
        messages = self.short_term.get_messages()
        
        # Check if compression is needed
        if self.mid_term.should_compress(len(messages), max_context_size):
            self.logger.info(f"Context compression triggered: {len(messages)} > {max_context_size}")
            compressed = self.mid_term.compress_context(messages, {})
            
            # Create summary message for compressed context
            summary_content = self._create_context_summary(compressed)
            summary_message = Message(
                id=f"summary_{int(time.time())}",
                type=MessageType.SYSTEM,
                content=summary_content,
                timestamp=time.time(),
                metadata={"type": "context_summary", "original_count": len(messages)}
            )
            
            # Return recent messages plus summary
            recent_messages = messages[-max_context_size//3:]
            return [summary_message] + recent_messages
        
        return messages[-max_context_size:]
    
    def _create_context_summary(self, compressed: Dict[str, Any]) -> str:
        """Create intelligent context summary from compressed data."""
        parts = []
        
        # Main intent
        if compressed.get("main_intent"):
            parts.append(f"User Intent: {compressed['main_intent']}")
        
        # Key concepts
        if compressed.get("key_concepts"):
            concepts = ", ".join(compressed["key_concepts"][:5])
            parts.append(f"Key Concepts: {concepts}")
        
        # Problems and solutions
        if compressed.get("problems_solutions"):
            recent_problems = compressed["problems_solutions"][-2:]
            for ps in recent_problems:
                parts.append(f"Issue: {ps['problem'][:50]}...")
                if ps["solution"] != "Under investigation":
                    parts.append(f"Solution: {ps['solution'][:50]}...")
        
        # Pending tasks
        if compressed.get("pending_tasks"):
            tasks = ", ".join(compressed["pending_tasks"][:3])
            parts.append(f"Next Tasks: {tasks}")
        
        # Next steps
        if compressed.get("next_steps"):
            steps = ", ".join(compressed["next_steps"][:3])
            parts.append(f"Suggested Steps: {steps}")
        
        return "\n".join(parts) if parts else "Context summary not available"
    
    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term.clear()
    
    def save_session(self, session_id: str) -> None:
        """Save session to long-term memory."""
        messages = self.short_term.get_messages()
        self.long_term.log_conversation(session_id, messages)
    
    def load_session(self, session_id: str) -> Optional[List[Message]]:
        """Load session from long-term memory."""
        history = self.long_term.get_conversation_history(session_id)
        if history:
            messages = []
            for msg_data in history:
                messages.append(Message(
                    id=msg_data["id"],
                    type=MessageType(msg_data["type"]),
                    content=msg_data["content"],
                    timestamp=msg_data["timestamp"],
                    parent_uuid=msg_data.get("parent_uuid"),
                    metadata=msg_data.get("metadata", {})
                ))
            
            # Restore to short-term memory
            self.short_term.clear()
            for msg in messages:
                self.short_term.add_message(msg)
            
            return messages
        
        return None