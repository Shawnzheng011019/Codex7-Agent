"""
Context management with 3-layer memory architecture.
Implements short-term, mid-term, and long-term memory systems.
"""

import os
import json
import time
import hashlib
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
        """Extract key technical concepts from messages."""
        all_content = " ".join([msg.content for msg in messages])
        
        # TODO: Use more sophisticated NLP for concept extraction
        technical_terms = [
            "function", "class", "method", "variable", "file", "directory",
            "import", "export", "return", "async", "await", "test", "debug",
            "refactor", "optimize", "search", "analyze", "create", "modify",
            "configuration", "dependency", "library", "framework"
        ]
        
        found_concepts = []
        for term in technical_terms:
            if term in all_content.lower():
                found_concepts.append(term)
        
        return found_concepts[:10]  # Limit to top 10 concepts
    
    def _extract_file_locations(self, messages: List[Message]) -> List[str]:
        """Extract relevant file locations from messages."""
        file_pattern = r'\b([\w/\\.-]+\.(py|js|ts|java|cpp|c|go|rs|json|yaml|yml|md))\b'
        
        all_content = " ".join([msg.content for msg in messages])
        files = re.findall(file_pattern, all_content)
        
        return [file[0] for file in files]
    
    def _extract_problems_solutions(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Extract problems and their solutions from messages."""
        problems_solutions = []
        
        # TODO: Implement more sophisticated problem/solution extraction
        for msg in messages:
            if "error" in msg.content.lower() or "issue" in msg.content.lower():
                problems_solutions.append({
                    "problem": msg.content[:100],
                    "solution": "Under investigation"
                })
        
        return problems_solutions
    
    def _extract_approach_method_results(self, messages: List[Message]) -> Dict[str, str]:
        """Extract problem-solving approach, method, and results."""
        return {
            "approach": "Systematic analysis and implementation",
            "method": "Iterative development with testing",
            "results": "Work in progress"
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
        """Extract pending tasks and current work summary."""
        pending_tasks = []
        
        # TODO: Implement task extraction from TODO items and messages
        for msg in messages:
            if "TODO" in msg.content.upper() or "task" in msg.content.lower():
                pending_tasks.append(msg.content[:100])
        
        return pending_tasks
    
    def _generate_next_steps(self, messages: List[Message]) -> List[str]:
        """Generate structured next steps."""
        # TODO: Implement intelligent next step generation
        return [
            "Continue with implementation",
            "Validate current progress",
            "Address any identified issues"
        ]


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
        """Get context for LLM consumption, with compression if needed."""
        messages = self.short_term.get_messages()
        
        # Check if compression is needed
        if self.mid_term.should_compress(len(messages), max_context_size):
            self.logger.info("Context compression triggered")
            compressed = self.mid_term.compress_context(messages, {})
            
            # TODO: Handle compressed context appropriately
            # For now, just return the most recent messages
            return messages[-max_context_size//2:]
        
        return messages[-max_context_size:]
    
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