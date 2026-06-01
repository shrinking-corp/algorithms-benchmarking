"""
Context for the Strategy pattern over target (shrinking) algorithms.
"""
from __future__ import annotations

from typing import Optional

from .base import BaseTargetAlgorithm


class ShrinkingContext:
    """
    Context in the Strategy pattern.

    Decouples FitnessEvaluator (Client) from concrete shrinking algorithms.
    The caller (Benchmarker) decides which strategy to use by calling
    set_strategy() before each algorithm run.
    """

    def __init__(self) -> None:
        self._strategy: Optional[BaseTargetAlgorithm] = None

    def set_strategy(self, strategy: BaseTargetAlgorithm) -> None:
        """Switch the active shrinking strategy at runtime."""
        self._strategy = strategy

    @property
    def strategy_name(self) -> str:
        return self._strategy.name if self._strategy is not None else "<none>"

    def execute(self, puml: str, params: dict) -> str:
        """Delegate diagram processing to the current strategy."""
        if self._strategy is None:
            raise RuntimeError("No strategy set on ShrinkingContext - call set_strategy() first.")
        return self._strategy.run(puml, params)
