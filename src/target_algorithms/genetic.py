"""
Genetic/Evolutionary adapter for shrinking-algorithms library.
"""
from __future__ import annotations

from shrinking_algorithms.shrinkers import EvolDiagramShrinker

from .base import BaseTargetAlgorithm


class GeneticTargetAlgorithm(BaseTargetAlgorithm):
    name = "Genetic"

    def run(self, puml_content: str, params: dict) -> str:
        int_params = {p["name"] for p in self.param_space if p.get("is_integer", False)}

        config: dict = {}
        for key, val in params.items():
            converted = int(val) if key in int_params else float(val)
            parts = key.split("__")
            d = config
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = converted
        try:
            result = EvolDiagramShrinker(
                puml_content=puml_content,
                config=config,
            ).shrink().get_result_puml()
            return result if result else puml_content
        except Exception:
            return puml_content

