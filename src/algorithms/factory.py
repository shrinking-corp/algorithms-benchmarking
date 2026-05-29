"""
Factory for optimisation algorithms. Dispatches on optimizer.type from config.yaml.
"""
from __future__ import annotations

from .base import BaseAlgorithm
from .pso import PSOAlgorithm


def create_optimizer(
    optimizer_cfg: dict,
    param_space: list,
    n_iterations: int,
    population_size: int,
) -> BaseAlgorithm:
    """Creates an optimiser from config.yaml optimizer block."""
    opt_type: str = optimizer_cfg.get("type", "pso")

    if opt_type == "pso":
        cfg = optimizer_cfg.get("pso", {})
        return PSOAlgorithm(
            name="PSO",
            param_space=param_space,
            n_iterations=n_iterations,
            population_size=population_size,
            inertia=cfg.get("inertia", 0.729),
            cognitive=cfg.get("cognitive", 1.494),
            social=cfg.get("social", 1.494),
        )

    raise ValueError(
        f"Unknown optimizer type: '{opt_type}'. Supported: pso"
    )
