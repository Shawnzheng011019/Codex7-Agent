"""
State update mechanism with multi-level state management.
Handles global state, React-like state updates, and caching optimization.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import logging
from dataclasses import dataclass


@dataclass
class StateChange:
    """Represents a single state change."""
    key: str
    old_value: Any
    new_value: Any
    timestamp: float
    source: str


class GlobalStateManager:
    """Global state management with smart caching and persistence."""
    
    def __init__(self, storage_path: str = ".claude_state"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # In-memory state cache
        self._state_cache: Dict[str, Any] = {}
        self._state_timestamps: Dict[str, float] = {}
        self._change_history: List[StateChange] = []
        
        # Protected configuration keys
        self._protected_keys = {
            "agent_id", "session_id", "user_id", "workspace_path"
        }
        
        # Cache expiration (5 minutes)
        self._cache_ttl = 300
    
    def update_state(self, key: str, value: Any, source: str = "unknown") -> None:
        """Update a single state value with change tracking."""
        old_value = self._state_cache.get(key)
        
        # Skip if no change
        if old_value == value:
            return
        
        # Protect critical configuration
        if key in self._protected_keys and old_value is not None:
            self.logger.warning(f"Attempted to modify protected key: {key}")
            return
        
        # Record change
        change = StateChange(
            key=key,
            old_value=old_value,
            new_value=value,
            timestamp=time.time(),
            source=source
        )
        self._change_history.append(change)
        
        # Update cache and timestamp
        self._state_cache[key] = value
        self._state_timestamps[key] = time.time()
        
        # Persist to disk
        self._persist_state(key, value)
        
        # Clear cache for this key to ensure fresh reads
        self._invalidate_cache(key)
        
        self.logger.debug(f"State updated: {key} = {value}")
    
    def batch_update(self, updates: Dict[str, Any], source: str = "unknown") -> None:
        """Update multiple state values atomically."""
        timestamp = time.time()
        
        for key, value in updates.items():
            if key in self._protected_keys and key in self._state_cache:
                continue
            
            old_value = self._state_cache.get(key)
            if old_value != value:
                change = StateChange(
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    timestamp=timestamp,
                    source=source
                )
                self._change_history.append(change)
                
                self._state_cache[key] = value
                self._state_timestamps[key] = timestamp
                self._persist_state(key, value)
                self._invalidate_cache(key)
        
        self.logger.info(f"Batch updated {len(updates)} state values")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value with caching and validation."""
        # Check cache first
        if key in self._state_cache:
            # Validate cache freshness
            if self._is_cache_valid(key):
                return self._state_cache[key]
        
        # Load from disk
        value = self._load_state(key)
        if value is not None:
            self._state_cache[key] = value
            self._state_timestamps[key] = time.time()
        
        return value or default
    
    def get_all_state(self) -> Dict[str, Any]:
        """Get all state values."""
        # Load from disk to ensure freshness
        all_state = {}
        state_file = self.storage_path / "global_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    all_state = json.load(f)
                    self._state_cache.update(all_state)
            except Exception as e:
                self.logger.error(f"Failed to load global state: {e}")
        
        return self._state_cache.copy()
    
    def has_state_changed(self, key: str, since: float) -> bool:
        """Check if state has changed since given timestamp."""
        timestamp = self._state_timestamps.get(key, 0)
        return timestamp > since
    
    def get_state_changes(self, since: float = 0) -> List[StateChange]:
        """Get all state changes since given timestamp."""
        return [change for change in self._change_history if change.timestamp > since]
    
    def _persist_state(self, key: str, value: Any) -> None:
        """Persist state value to disk."""
        try:
            # Save individual key file
            key_file = self.storage_path / f"{key}.json"
            with open(key_file, 'w') as f:
                json.dump({"key": key, "value": value, "timestamp": time.time()}, f, indent=2)
            
            # Update global state file
            self._update_global_state_file()
            
        except Exception as e:
            self.logger.error(f"Failed to persist state {key}: {e}")
    
    def _load_state(self, key: str) -> Optional[Any]:
        """Load state value from disk."""
        key_file = self.storage_path / f"{key}.json"
        
        if key_file.exists():
            try:
                with open(key_file, 'r') as f:
                    data = json.load(f)
                    return data.get("value")
            except Exception as e:
                self.logger.error(f"Failed to load state {key}: {e}")
        
        return None
    
    def _update_global_state_file(self) -> None:
        """Update global state file with all current values."""
        try:
            global_file = self.storage_path / "global_state.json"
            with open(global_file, 'w') as f:
                json.dump(self._state_cache, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to update global state file: {e}")
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached value is still valid."""
        timestamp = self._state_timestamps.get(key, 0)
        return time.time() - timestamp < self._cache_ttl
    
    def _invalidate_cache(self, key: str) -> None:
        """Invalidate cache for a specific key."""
        # Cache invalidation is handled by timestamp updates
        pass
    
    def clear_cache(self) -> None:
        """Clear all cached state values."""
        self._state_cache.clear()
        self._state_timestamps.clear()
        self.logger.info("State cache cleared")
    
    def reset_state(self, preserve_protected: bool = True) -> None:
        """Reset all state values."""
        if preserve_protected:
            protected_state = {
                k: v for k, v in self._state_cache.items() 
                if k in self._protected_keys
            }
            self._state_cache.clear()
            self._state_cache.update(protected_state)
        else:
            self._state_cache.clear()
            self._state_timestamps.clear()
        
        self.logger.info("State reset completed")


class ReactStateManager:
    """React-like state management for component-level state."""
    
    def __init__(self):
        self._component_states: Dict[str, Dict[str, Any]] = {}
        self._listeners: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_component_state(self, component_id: str, initial_state: Dict[str, Any]) -> None:
        """Create a new component state."""
        self._component_states[component_id] = initial_state.copy()
        self._listeners[component_id] = []
        self.logger.debug(f"Created component state: {component_id}")
    
    def update_component_state(
        self, 
        component_id: str, 
        updates: Dict[str, Any] or Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """Update component state with functional updates support."""
        if component_id not in self._component_states:
            self.logger.warning(f"Component {component_id} not found")
            return
        
        current_state = self._component_states[component_id]
        
        if callable(updates):
            new_state = updates(current_state)
        else:
            new_state = {**current_state, **updates}
        
        # Detect changes
        changes = {
            key: new_state[key] 
            for key in new_state 
            if key not in current_state or current_state[key] != new_state[key]
        }
        
        if changes:
            self._component_states[component_id] = new_state
            self._notify_listeners(component_id, changes)
            self.logger.debug(f"Updated {component_id}: {list(changes.keys())}")
    
    def get_component_state(self, component_id: str) -> Dict[str, Any]:
        """Get current component state."""
        return self._component_states.get(component_id, {}).copy()
    
    def subscribe(self, component_id: str, callback: Callable) -> None:
        """Subscribe to component state changes."""
        if component_id in self._listeners:
            self._listeners[component_id].append(callback)
    
    def unsubscribe(self, component_id: str, callback: Callable) -> None:
        """Unsubscribe from component state changes."""
        if component_id in self._listeners:
            try:
                self._listeners[component_id].remove(callback)
            except ValueError:
                pass
    
    def _notify_listeners(self, component_id: str, changes: Dict[str, Any]) -> None:
        """Notify all listeners of state changes."""
        if component_id in self._listeners:
            for callback in self._listeners[component_id]:
                try:
                    callback(changes)
                except Exception as e:
                    self.logger.error(f"Listener callback failed: {e}")


class CacheManager:
    """Smart caching with file modification time validation."""
    
    def __init__(self, cache_path: str = ".claude_cache"):
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(exist_ok=True)
        self.file_cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def get_cached_content(self, file_path: str) -> Optional[str]:
        """Get cached file content if still valid."""
        if file_path not in self.file_cache:
            return None
        
        cache_entry = self.file_cache[file_path]
        cached_mtime = cache_entry.get("mtime", 0)
        
        # Check if file exists and get current modification time
        try:
            current_mtime = os.path.getmtime(file_path)
            if cached_mtime >= current_mtime:
                self.logger.debug(f"Cache hit for {file_path}")
                return cache_entry.get("content")
        except OSError:
            # File doesn't exist, remove from cache
            del self.file_cache[file_path]
        
        return None
    
    def cache_content(self, file_path: str, content: str) -> None:
        """Cache file content with modification time."""
        try:
            mtime = os.path.getmtime(file_path)
            self.file_cache[file_path] = {
                "content": content,
                "mtime": mtime,
                "size": len(content)
            }
            
            # Persist to disk
            cache_file = self.cache_path / f"{hashlib.md5(file_path.encode()).hexdigest()}.json"
            with open(cache_file, 'w') as f:
                json.dump(self.file_cache[file_path], f)
                
        except Exception as e:
            self.logger.error(f"Failed to cache {file_path}: {e}")
    
    def invalidate_cache(self, file_path: str) -> None:
        """Invalidate cache for a specific file."""
        if file_path in self.file_cache:
            del self.file_cache[file_path]
            
            # Remove cache file
            cache_file = self.cache_path / f"{hashlib.md5(file_path.encode()).hexdigest()}.json"
            if cache_file.exists():
                cache_file.unlink()
    
    def clear_cache(self) -> None:
        """Clear all cached content."""
        self.file_cache.clear()
        
        # Remove all cache files
        for cache_file in self.cache_path.glob("*.json"):
            cache_file.unlink()
        
        self.logger.info("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(entry.get("size", 0) for entry in self.file_cache.values())
        
        return {
            "cached_files": len(self.file_cache),
            "total_size": total_size,
            "cache_path": str(self.cache_path)
        }


class EventLogger:
    """Event logging for state changes and system events."""
    
    def __init__(self, log_path: str = ".claude_logs"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("agent_events")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_path / "events.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        self.logger.addHandler(file_handler)
    
    def log_state_change(self, change: Any, source: str) -> None:
        """Log a state change event."""
        self.logger.info(f"State change: {change} from {source}")
    
    def log_tool_execution(self, tool_name: str, success: bool, duration: float) -> None:
        """Log tool execution event."""
        self.logger.info(f"Tool {tool_name}: {'success' if success else 'failure'} ({duration:.2f}s)")
    
    def log_error(self, error_type: str, message: str, context: Dict[str, Any]) -> None:
        """Log error event."""
        self.logger.error(f"Error [{error_type}]: {message} - Context: {context}")


class StateManager:
    """Main state management coordinator."""
    
    def __init__(self):
        self.global_state = GlobalStateManager()
        self.react_state = ReactStateManager()
        self.cache_manager = CacheManager()
        self.event_logger = EventLogger()
        self.logger = logging.getLogger(__name__)
    
    def update_global_state(self, key: str, value: Any, source: str = "unknown") -> None:
        """Update global state and log event."""
        self.global_state.update_state(key, value, source)
        self.event_logger.log_state_change({"key": key, "value": value}, source)
    
    def get_global_state(self, key: str, default: Any = None) -> Any:
        """Get global state value."""
        return self.global_state.get_state(key, default)
    
    def create_component(self, component_id: str, initial_state: Dict[str, Any]) -> None:
        """Create a new component with React-like state."""
        self.react_state.create_component_state(component_id, initial_state)
    
    def update_component_state(
        self, 
        component_id: str, 
        updates: Any, 
        source: str = "unknown"
    ) -> None:
        """Update component state and log event."""
        self.react_state.update_component_state(component_id, updates)
        self.event_logger.log_state_change({"component": component_id, "updates": updates}, source)
    
    def get_component_state(self, component_id: str) -> Dict[str, Any]:
        """Get component state."""
        return self.react_state.get_component_state(component_id)
    
    def cache_file_content(self, file_path: str, content: str) -> None:
        """Cache file content."""
        self.cache_manager.cache_content(file_path, content)
    
    def get_cached_content(self, file_path: str) -> Optional[str]:
        """Get cached file content."""
        return self.cache_manager.get_cached_content(file_path)
    
    def invalidate_file_cache(self, file_path: str) -> None:
        """Invalidate file cache."""
        self.cache_manager.invalidate_cache(file_path)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary."""
        return {
            "global_state_keys": len(self.global_state.get_all_state()),
            "component_states": len(self.react_state._component_states),
            "cache_stats": self.cache_manager.get_cache_stats(),
            "recent_changes": len(self.global_state.get_state_changes(time.time() - 300))
        }