"""
Configuration loading utilities for the agent.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and access for the agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config = {}
        self.load_config()
    
    def _find_config_file(self) -> str:
        """Find the main configuration file."""
        possible_paths = [
            "codex7_config.json",
            os.path.expanduser("~/.codex7/config.json"),
            "/etc/codex7/config.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default if no config found
        return "codex7_config.json"
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = self._get_default_config()
        
        return self.config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "llm": {
                "provider": "openai",
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "claude-3-5-sonnet-20241022",
                "temperature": 0.1,
                "max_tokens": 4000,
                "timeout": 60,
                "retry_attempts": 3,
                "proxy": {
                    "enabled": False,
                    "http_proxy": "",
                    "https_proxy": "",
                    "socks_proxy": ""
                },
                "headers": {
                    "User-Agent": "Codex7-Agent/1.0"
                }
            },
            "features": {
                "enable_streaming": True,
                "enable_tool_usage": True,
                "enable_context_awareness": True,
                "enable_memory": True,
                "enable_fallback": True
            }
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self.config.get("llm", {})
    
    def get_embedding_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Get embedding configuration."""
        embedding_path = config_path or "src/retrieval/embedding_config.json"
        
        try:
            if os.path.exists(embedding_path):
                with open(embedding_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Embedding config {embedding_path} not found")
                return {}
        except Exception as e:
            logger.error(f"Error loading embedding config: {e}")
            return {}
    
    def get_proxy_config(self) -> Dict[str, Any]:
        """Get proxy configuration."""
        llm_config = self.get_llm_config()
        return llm_config.get("proxy", {})
    
    def get_中转api_config(self) -> Dict[str, Any]:
        """Get 中转API configuration."""
        return self.config.get("中转api", {})
    
    def get_fallback_models(self) -> list:
        """Get fallback models configuration."""
        return self.config.get("fallback_models", [])
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._deep_update(self.config, updates)
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_env_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}
        
        # LLM environment variables
        if os.getenv("OPENAI_API_KEY"):
            overrides.setdefault("llm", {})["api_key"] = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            overrides.setdefault("llm", {})["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        
        if os.getenv("LLM_BASE_URL"):
            overrides.setdefault("llm", {})["base_url"] = os.getenv("LLM_BASE_URL")
        
        if os.getenv("HTTP_PROXY"):
            overrides.setdefault("llm", {}).setdefault("proxy", {})["http_proxy"] = os.getenv("HTTP_PROXY")
        
        if os.getenv("HTTPS_PROXY"):
            overrides.setdefault("llm", {}).setdefault("proxy", {})["https_proxy"] = os.getenv("HTTPS_PROXY")
        
        # Embedding environment variables
        if os.getenv("EMBEDDING_API_KEY"):
            overrides.setdefault("embedding", {})["api_key"] = os.getenv("EMBEDDING_API_KEY")
        
        if os.getenv("EMBEDDING_BASE_URL"):
            overrides.setdefault("embedding", {})["base_url"] = os.getenv("EMBEDDING_BASE_URL")
        
        return overrides


# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration."""
    return get_config_manager().get_llm_config()

def get_embedding_config() -> Dict[str, Any]:
    """Get embedding configuration."""
    return get_config_manager().get_embedding_config()

def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Load configuration from file."""
    return ConfigManager(config_path)