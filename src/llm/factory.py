"""
Factory for LLM clients. Wraps in CachingProxy if cache_dir is set.
"""
from __future__ import annotations

from typing import Optional

from .base import LLMClient
from .litellm_client import LiteLLMClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient
from .proxy import CachingProxy


def _resolve_models(cfg: dict) -> list[str]:
    if "models" in cfg:
        return list(cfg["models"])
    return [cfg["model"]]


def create_llm_client(cfg: dict, cache_dir: Optional[str] = None, timeout: float = 90.0) -> LLMClient:
    t = cfg["type"]
    llm_params = cfg.get("llm_params", {})
    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
    models = _resolve_models(cfg)

    if t == "openai":
        real: LLMClient = OpenAIClient(
            name=cfg["name"],
            models=models,
            system_prompt=system_prompt,
            api_base=cfg["api_base"],
            api_key_env=cfg["api_key_env"],
            llm_params=llm_params,
            timeout=timeout,
        )
    elif t == "ollama":
        real = OllamaClient(
            name=cfg["name"],
            models=models,
            system_prompt=system_prompt,
            api_base=cfg["api_base"],
            llm_params=llm_params,
            timeout=timeout,
        )
    elif t == "litellm_provider":
        real = LiteLLMClient(
            name=cfg["name"],
            models=models,
            system_prompt=system_prompt,
            api_base=cfg["api_base"],
            api_key_env=cfg.get("api_key_env"),
            llm_params=llm_params,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown LLM type: '{t}'. Supported: openai, ollama, litellm_provider")

    return CachingProxy(real, cache_dir) if cache_dir else real
