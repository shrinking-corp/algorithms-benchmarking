"""
Abstract base class for optimization algorithms.
Defines interface for parameter space, normalization, and memento support.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Tuple

import numpy as np

from .memento import AlgorithmMemento


class ConfigMismatchError(Exception):
    def __init__(self, mismatches: list):
        self.mismatches = mismatches
        super().__init__(
            "Checkpoint config differs from current config:\n" + "\n".join(mismatches)
        )


def _snapshot_hash(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _diff_snapshots(saved: dict, current: dict, _path: str = "") -> list:
    mismatches = []
    for key in sorted(set(saved) | set(current)):
        full_key = f"{_path}.{key}" if _path else key
        if key not in saved:
            mismatches.append(f"  {full_key}: not in checkpoint, current={current[key]!r}")
        elif key not in current:
            mismatches.append(f"  {full_key}: checkpoint={saved[key]!r}, not in current config")
        elif isinstance(saved[key], dict) and isinstance(current[key], dict):
            mismatches.extend(_diff_snapshots(saved[key], current[key], full_key))
        elif isinstance(saved[key], list) and isinstance(current[key], list):
            if len(saved[key]) != len(current[key]):
                mismatches.append(
                    f"  {full_key}: checkpoint length={len(saved[key])}, current={len(current[key])}"
                )
            else:
                for i, (s, c) in enumerate(zip(saved[key], current[key])):
                    if isinstance(s, dict) and isinstance(c, dict):
                        mismatches.extend(_diff_snapshots(s, c, f"{full_key}[{i}]"))
                    elif s != c:
                        mismatches.append(f"  {full_key}[{i}]: checkpoint={s!r}, current={c!r}")
        elif saved[key] != current[key]:
            mismatches.append(f"  {full_key}: checkpoint={saved[key]!r}, current={current[key]!r}")
    return mismatches


@dataclass
class ParamDef:
    name: str
    min_val: float
    max_val: float
    is_integer: bool = False


FitnessFn = Callable[[np.ndarray], Awaitable[float]]


class BaseAlgorithm(ABC):
    def __init__(
        self,
        name: str,
        param_space: List[Dict],
        n_iterations: int,
        population_size: int,
    ):
        self.name = name
        self.param_defs: List[ParamDef] = [
            ParamDef(
                p["name"],
                float(p["min"]),
                float(p["max"]),
                bool(p.get("is_integer", False)),
            )
            for p in param_space
        ]
        self.n_dim = len(self.param_defs)
        self.n_iterations = n_iterations
        self.population_size = population_size

    def interpret(self, solution: np.ndarray) -> Dict[str, float]:
        # Maps a normalized vector [0,1]^n to a dict of named parameter values.
        result: Dict[str, float] = {}
        for i, p in enumerate(self.param_defs):
            value = p.min_val + float(solution[i]) * (p.max_val - p.min_val)
            result[p.name] = int(round(value)) if p.is_integer else value
        return result

    @abstractmethod
    async def optimize(self, fitness_fn: FitnessFn) -> Tuple[np.ndarray, float]:
        pass

    @abstractmethod
    def create_memento(self) -> AlgorithmMemento:
        pass

    @abstractmethod
    def restore_memento(self, memento: AlgorithmMemento) -> None:
        pass

    def _build_config_snapshot(self) -> dict:
        # Returns a serializable snapshot of this algorithm's configuration.
        return {
            "n_iterations": self.n_iterations,
            "population_size": self.population_size,
            "param_space": [
                {
                    "name": p.name,
                    "min": p.min_val,
                    "max": p.max_val,
                    "is_integer": p.is_integer,
                }
                for p in self.param_defs
            ],
        }

    def _random_pop(self, n: int) -> np.ndarray:
        return np.random.rand(n, self.n_dim)

    @staticmethod
    def _clip(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0.0, 1.0)
