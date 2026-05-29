"""
OpenAI-compatible Real Subject.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from .base import BaseAPIClient

logger = logging.getLogger(__name__)


class OpenAIClient(BaseAPIClient):

    def __init__(
        self,
        name: str,
        models: list[str],
        system_prompt: str,
        api_base: str,
        api_key_env: str,
        llm_params: Optional[dict] = None,
        timeout: float = 90.0,
    ):
        super().__init__(name, models, system_prompt, llm_params)
        self.api_base = api_base.rstrip("/")
        self._timeout = timeout
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise EnvironmentError(
                f"Environment variable '{api_key_env}' is not set. "
                "Create a .env file based on .env.example"
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload_for(self, model: str, prompt: str) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "top_p": self._top_p,
        }

    async def _call_api(self, payload: dict) -> Any:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers=self._headers,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    def _parse_response(self, response: Any) -> str:
        return response["choices"][0]["message"]["content"]
