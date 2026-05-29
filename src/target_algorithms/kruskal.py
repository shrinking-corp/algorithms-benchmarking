"""
Kruskal adapter for shrinking-algorithms library.
"""
from __future__ import annotations

from shrinking_algorithms.shrinkers import KruskalDiagramShrinker

from .base import BaseTargetAlgorithm


class KruskalAlgorithm(BaseTargetAlgorithm):
    name = "Kruskal"

    def run(self, puml_content: str, params: dict) -> str:
        config = {key: float(val) for key, val in params.items()}
        try:
            result = KruskalDiagramShrinker(
                puml_content=puml_content,
                config=config,
            ).shrink().get_result_puml()
            return result if result else puml_content
        except Exception:
            return puml_content

