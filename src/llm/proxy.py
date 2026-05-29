"""
Caching Proxy for LLM clients.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from .base import LLMClient

logger = logging.getLogger(__name__)


class CachingProxy(LLMClient):
    """
    Transparent caching layer around any LLMClient.

    Cache key = SHA-256(model | temperature | top_p | system_prompt | prompt).
    Each unique request is stored as a single JSON file under cache_dir.
    """

    def __init__(self, real: LLMClient, cache_dir: str):
        super().__init__(
            name=real.name,
            models=real.models,
            system_prompt=real.system_prompt,
            llm_params={
                "temperature": real._temperature,
                "top_p": real._top_p,
            },
        )
        self._real = real
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._cache_dir / "_index.txt"
        self._index: set[str] = self._load_index()
        logger.debug("Cache index loaded: %d entries from %s", len(self._index), self._cache_dir)

    def _load_index(self) -> set[str]:
        """
        Load the in-memory key index.

        Fast path: read _index.txt (one SHA-256 hex per line) - single sequential read.
        Fallback:  scan *.json filenames (backward-compat with existing caches that
                   predate the index file) and write _index.txt for next startup.
        """
        if self._index_path.exists():
            keys = {
                line
                for line in self._index_path.read_text(encoding="utf-8").splitlines()
                if line
            }
            return keys

        # One-time migration: rebuild index from existing .json files.
        keys = {p.stem for p in self._cache_dir.glob("*.json")}
        if keys:
            logger.info(
                "Cache: no index file found - rebuilding from %d existing entries.", len(keys)
            )
            self._write_full_index(keys)
        return keys

    def _write_full_index(self, keys: set[str]) -> None:
        """Writes the complete key set to _index.txt (used for one-time migration)."""
        try:
            self._index_path.write_text("\n".join(keys) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.warning("Cache: failed to write index file: %s", exc)

    # ── Proxy interface ───────────────────────────────────────────────────

    async def query(self, prompt: str) -> str:
        key = self._build_key_for(self.model, prompt)

        cached = self._check_cache(key)
        if cached is not None:
            logger.debug("Cache hit  [%s]", key[:12])
            return cached

        response = await self._forward(prompt)
        self._store(key, response)
        return response

    async def query_model(self, model: str, prompt: str) -> str:
        key = self._build_key_for(model, prompt)

        cached = self._check_cache(key)
        if cached is not None:
            logger.debug("Cache hit  [%s]", key[:12])
            return cached

        response = await self._real.query_model(model, prompt)
        self._store(key, response)
        return response

    # ── Three proxy responsibilities ──────────────────────────────────────

    def _check_cache(self, key: str) -> Optional[str]:
        """Returns the cached response, or None on a miss."""
        if key not in self._index:
            return None
        path = self._cache_dir / f"{key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except Exception:
            pass
        return None

    async def _forward(self, prompt: str) -> str:
        """Delegates the query to the real LLM client."""
        return await self._real.query(prompt)

    def _store(self, key: str, response: str) -> None:
        """Persists the response to the cache directory."""
        path = self._cache_dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps({"response": response}, ensure_ascii=False),
                encoding="utf-8",
            )
            
            with self._index_path.open("a", encoding="utf-8") as f:
                f.write(key + "\n")
            self._index.add(key)
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    # ── Helper ────────────────────────────────────────────────────────────

    def _build_key_for(self, model: str, prompt: str) -> str:
        content = (
            f"{model}|{self._temperature:.4f}|{self._top_p:.4f}"
            f"|{self.system_prompt}|{prompt}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
