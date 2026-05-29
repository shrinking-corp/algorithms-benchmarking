"""
Ollama local server Real Subject.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .base import BaseAPIClient

logger = logging.getLogger(__name__)


class OllamaClient(BaseAPIClient):

    def __init__(
        self,
        name: str,
        models: list[str],
        system_prompt: str,
        api_base: str,
        llm_params: Optional[dict] = None,
        timeout: float = 90.0,
    ):
        super().__init__(name, models, system_prompt, llm_params)
        self.api_base = api_base.rstrip("/")
        self._timeout = timeout

    def _build_payload_for(self, model: str, prompt: str) -> dict:
        return {
            "model": model,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature, "top_p": self._top_p},
        }

    async def _call_api(self, payload: dict) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{self.api_base}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()

    def _parse_response(self, response: Any) -> str:
        return response["response"]
