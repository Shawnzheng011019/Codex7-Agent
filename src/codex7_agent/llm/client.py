"""
LLM Client for handling API calls with configuration support.
"""

import aiohttp
import asyncio
import logging
from typing import Dict, Any, Optional, List
import json
from urllib.parse import urljoin
import os

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client with configuration support including proxy and 中转API."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self._setup_session()
    
    def _setup_session(self):
        """Setup aiohttp session with proxy and headers."""
        timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 60))
        
        # Setup proxy if enabled
        connector = None
        proxy_config = self.config.get('proxy', {})
        if proxy_config.get('enabled', False):
            proxy_url = proxy_config.get('https_proxy') or proxy_config.get('http_proxy')
            if proxy_url:
                connector = aiohttp.TCPConnector()
        
        headers = self.config.get('headers', {})
        headers['Authorization'] = f"Bearer {self.config.get('api_key', '')}"
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send chat completion request."""
        if not self.session:
            self._setup_session()
        
        # Build request payload
        payload = {
            'model': self.config.get('model', 'claude-3-5-sonnet-20241022'),
            'messages': messages,
            'temperature': kwargs.get('temperature', self.config.get('temperature', 0.1)),
            'max_tokens': kwargs.get('max_tokens', self.config.get('max_tokens', 4000)),
            'stream': kwargs.get('stream', False)
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        url = self._get_api_url()
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"LLM API error: {response.status} - {error_text}")
                    raise Exception(f"API error: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            raise
    
    def _get_api_url(self) -> str:
        """Get the correct API URL based on configuration."""
        base_url = self.config.get('base_url', 'https://api.openai.com/v1')
        
        # Handle 中转API configuration
        中转_config = self.config.get('中转api', {})
        if 中转_config.get('enabled', False):
            providers = 中转_config.get('providers', [])
            if providers:
                # Use first provider as default
                provider = providers[0]
                base_url = provider.get('base_url', base_url)
                
                # Check model mapping
                current_model = self.config.get('model', '')
                model_mapping = provider.get('model_mapping', {})
                if current_model in model_mapping:
                    # Update model in config for this provider
                    self.config['model'] = model_mapping[current_model]
        
        # Ensure URL ends with /chat/completions for OpenAI compatible APIs
        if 'openai.com' in base_url or 'proxy' in base_url:
            return urljoin(base_url.rstrip('/') + '/', 'chat/completions')
        elif 'anthropic.com' in base_url:
            return urljoin(base_url.rstrip('/') + '/', 'v1/messages')
        else:
            return urljoin(base_url.rstrip('/') + '/', 'chat/completions')
    
    async def test_connection(self) -> bool:
        """Test if the LLM connection is working."""
        try:
            test_messages = [{"role": "user", "content": "Hello"}]
            response = await self.chat_completion(test_messages, max_tokens=10)
            return response is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


class LLMManager:
    """Manages LLM clients with fallback support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.primary_client = LLMClient(config)
        self.fallback_clients = []
        self._setup_fallback_clients()
    
    def _setup_fallback_clients(self):
        """Setup fallback clients."""
        fallback_models = self.config.get('fallback_models', [])
        for fallback_config in fallback_models:
            self.fallback_clients.append(LLMClient(fallback_config))
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Send chat completion with fallback support."""
        
        # Try primary client
        try:
            return await self.primary_client.chat_completion(messages, **kwargs)
        except Exception as e:
            logger.warning(f"Primary LLM failed: {e}")
            
            # Try fallback clients
            for i, fallback_client in enumerate(self.fallback_clients):
                try:
                    logger.info(f"Trying fallback client {i+1}")
                    return await fallback_client.chat_completion(messages, **kwargs)
                except Exception as fe:
                    logger.warning(f"Fallback client {i+1} failed: {fe}")
            
            # All clients failed
            raise Exception("All LLM clients failed")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.primary_client.session:
            await self.primary_client.session.close()
        for client in self.fallback_clients:
            if client.session:
                await client.session.close()


# Factory function to create LLM manager
def create_llm_manager(config: Optional[Dict[str, Any]] = None) -> LLMManager:
    """Create LLM manager with configuration."""
    if config is None:
        from ..config import get_llm_config
        config = get_llm_config()
    
    return LLMManager(config)