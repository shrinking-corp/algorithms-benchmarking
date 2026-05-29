"""
Factory for target algorithms.
"""
from __future__ import annotations

from .base import BaseTargetAlgorithm
from .genetic import GeneticTargetAlgorithm
from .kruskal import KruskalAlgorithm


def create_target_algorithm(cfg: dict) -> BaseTargetAlgorithm:
    """Creates a target algorithm from a config.yaml target_algorithms entry."""
    alg_type: str = cfg["type"]

    if alg_type == "kruskal":
        alg: BaseTargetAlgorithm = KruskalAlgorithm()
    elif alg_type == "genetic_target":
        alg = GeneticTargetAlgorithm()
    else:
        raise ValueError(
            f"Unknown target_algorithm type: '{alg_type}'. "
            "Supported: kruskal, genetic_target"
        )

    # Override name from config if provided
    if "name" in cfg:
        alg.name = cfg["name"]

    # param_space is required in config.yaml - there is no in-code fallback
    if "param_space" not in cfg:
        raise ValueError(
            f"target_algorithm '{alg.name}' has no 'param_space' defined in config.yaml."
        )
    alg.param_space = cfg["param_space"]

    return alg
