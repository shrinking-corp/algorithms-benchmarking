"""
Abstract base and Template Method for all LLM clients.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """Returns True if the exception is worth retrying with exponential backoff."""

    if isinstance(exc, httpx.TimeoutException):
        return True
    
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg or "too many requests" in msg


class LLMClient(ABC):
    """Subject interface - the only contract the rest of the codebase depends on."""

    def __init__(
        self,
        name: str,
        models: list[str],
        system_prompt: str,
        llm_params: Optional[dict] = None,
    ):
        self.name = name
        self.models: list[str] = models
        self.system_prompt = system_prompt
        _p = llm_params or {}
        self._temperature: float = max(0.0, min(2.0, float(_p.get("temperature", 0.2))))
        self._top_p: float = max(0.0, min(1.0, float(_p.get("top_p", 0.9))))

    @property
    def model(self) -> str:
        return self.models[0]

    @abstractmethod
    async def query(self, prompt: str) -> str:
        pass

    async def query_model(self, model: str, prompt: str) -> str:
        return await self.query(prompt)

    async def query_all(self, prompt: str) -> dict[str, str]:
        """Sends a prompt to all models in parallel. Returns {model: response}."""
        results = await asyncio.gather(*[
            self.query_model(m, prompt) for m in self.models
        ])
        return dict(zip(self.models, results))


class BaseAPIClient(LLMClient):
    """
    Template Method - defines the skeleton of an LLM API call.

    Fixed sequence:
      1. _build_payload_for(model, prompt) → construct the request body
      2. _call_api(payload)                → perform the HTTP call, return raw response
      3. _parse_response(response)         → extract the text from the raw response

    Subclasses override only the steps that differ between providers.
    _build_payload(prompt) is a convenience shim → _build_payload_for(self.model, prompt).
    """

    _MAX_RETRIES: int = 5
    # Base wait in seconds for rate-limit backoff: attempt 1→10s, 2→20s, 3→40s, 4→60s, 5→60s
    _RATE_LIMIT_BASE_WAIT: float = 10.0
    _RATE_LIMIT_MAX_WAIT: float = 60.0

    async def query(self, prompt: str) -> str:
        payload = self._build_payload_for(self.model, prompt)
        return self._parse_response(await self._call_api_with_retry(payload))

    async def query_model(self, model: str, prompt: str) -> str:
        """Queries a specific model using the same Template Method skeleton."""
        payload = self._build_payload_for(model, prompt)
        return self._parse_response(await self._call_api_with_retry(payload))

    async def _call_api_with_retry(self, payload: dict) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return await self._call_api(payload)
            except Exception as exc:
                last_exc = exc
                if _is_retryable(exc):
                    wait = min(self._RATE_LIMIT_BASE_WAIT * (2 ** (attempt - 1)), self._RATE_LIMIT_MAX_WAIT)
                    logger.warning(
                        "[%s] %s (attempt %d/%d) - waiting %.0fs before retry.",
                        self.name, type(exc).__name__, attempt, self._MAX_RETRIES, wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.warning(
                        "[%s] API call failed - %s: %s",
                        self.name, type(exc).__name__, exc,
                    )
                    raise  # non-retryable errors propagate immediately
        raise last_exc  # type: ignore[misc]

    @abstractmethod
    def _build_payload_for(self, model: str, prompt: str) -> dict:
        """Constructs the provider-specific request payload for the given model."""

    @abstractmethod
    async def _call_api(self, payload: dict) -> Any:
        """Performs the HTTP call and returns the raw response object."""

    @abstractmethod
    def _parse_response(self, response: Any) -> str:
        """Extracts the generated text from the raw response."""
