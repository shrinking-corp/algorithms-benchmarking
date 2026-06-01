"""
Context for the Strategy pattern over optimization algorithms.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import numpy as np

from .base import BaseAlgorithm, FitnessFn

if TYPE_CHECKING:
    from .memento import AlgorithmMemento


class OptimizerContext:
    """
    Context in the Strategy pattern for optimization algorithms.

    Decouples FitnessEvaluator / Benchmarker (Clients) from the concrete
    optimization algorithm.  Benchmarker decides which algorithm to use by
    calling set_strategy() before each optimization run.
    """

    def __init__(self) -> None:
        self._strategy: Optional[BaseAlgorithm] = None

    def set_strategy(self, strategy: BaseAlgorithm) -> None:
        """Switch the active optimization strategy at runtime."""
        self._strategy = strategy

    @property
    def strategy_name(self) -> str:
        return self._strategy.name if self._strategy is not None else "<none>"

    def interpret(self, solution: np.ndarray) -> dict:
        """Map a normalized solution vector to named parameter values."""
        return self._strategy.interpret(solution)

    async def optimize(
        self,
        fitness_fn: FitnessFn,
        on_iteration: Optional[Callable] = None,
        on_checkpoint: Optional[Callable] = None,
        on_particle: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, float]:
        """Delegate optimization to the active strategy."""
        if self._strategy is None:
            raise RuntimeError("No strategy set on OptimizerContext - call set_strategy() first.")
        return await self._strategy.optimize(
            fitness_fn,
            on_iteration=on_iteration,
            on_checkpoint=on_checkpoint,
            on_particle=on_particle,
        )

    def restore_memento(self, memento: "AlgorithmMemento", force: bool = False) -> None:
        """Restore optimizer state from a checkpoint memento."""
        if self._strategy is None:
            raise RuntimeError("No strategy set on OptimizerContext - call set_strategy() first.")
        self._strategy.restore_memento(memento, force=force)

    @property
    def gbest_fit_history(self) -> list:
        """gbest fitness value after each completed iteration."""
        if self._strategy is None:
            return []
        return list(getattr(self._strategy, "_state", {}).get("gbest_fit_history", []))

    @property
    def gbest_pos_history(self) -> list:
        """gbest position vector after each completed iteration."""
        if self._strategy is None:
            return []
        return list(getattr(self._strategy, "_state", {}).get("gbest_pos_history", []))
