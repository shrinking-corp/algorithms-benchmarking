"""
Backwards-compatibility shim for LLM client imports.
"""
from .base import LLMClient
from .factory import create_llm_client
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .proxy import CachingProxy

__all__ = ["LLMClient", "OpenAIClient", "OllamaClient", "CachingProxy", "create_llm_client"]

