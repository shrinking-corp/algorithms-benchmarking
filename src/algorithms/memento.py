"""
Abstract Memento + Factory Method for optimizer state snapshots.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class AlgorithmMemento(ABC):
    """
    Abstract base for all optimizer state snapshots.

    Subclasses declare their type key via the class keyword argument:

        class PSOMemento(AlgorithmMemento, memento_type="pso"): ...

    This auto-registers the subclass so that AlgorithmMemento.deserialize()
    can reconstruct any known type without importing it explicitly.
    """

    _registry: Dict[str, type["AlgorithmMemento"]] = {}

    def __init_subclass__(cls, memento_type: str = "", **kwargs: Any) -> None:
        """Auto-register every concrete subclass by its memento_type key."""
        super().__init_subclass__(**kwargs)
        if memento_type:
            AlgorithmMemento._registry[memento_type] = cls
            cls._memento_type = memento_type

    # ── Abstract Factory Method steps ─────────────────────────────────────

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to a JSON-serialisable dict."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlgorithmMemento":
        """Factory Method step - each subclass constructs itself from data."""

    # ── Templates that hide concrete types from callers ───────────────────

    def to_storable(self) -> Dict[str, Any]:
        """Wraps to_dict() adding the __type__ tag required by deserialize()."""
        return {"__type__": self._memento_type, **self.to_dict()}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "AlgorithmMemento":
        """
        Factory Method dispatcher - reads __type__ and delegates to the
        registered subclass from_dict().  Callers never reference concrete types.
        """
        memento_type = data.pop("__type__", None)
        cls = AlgorithmMemento._registry.get(memento_type)
        if cls is None:
            raise ValueError(f"Unknown memento type '{memento_type}'.")
        return cls.from_dict(data)
