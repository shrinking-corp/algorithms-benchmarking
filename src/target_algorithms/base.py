"""
Abstract base class for target algorithms.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseTargetAlgorithm(ABC):
    """
    Interface for algorithms whose parameters are being tuned.

    Implementation example:
        class MyAlg(BaseTargetAlgorithm):
            name = "MyAlg"

            def run(self, puml_content: str, params: dict) -> str:
                # 1. Read params
                # 2. Run algorithm on PUML
                # 3. Return representation for LLM
                ...
    """

    name: str
    param_space: List[Dict] = []  # Set by factory from config.yaml - do not define in subclasses

    @abstractmethod
    def run(self, puml_content: str, params: dict) -> str:
        """
        Runs the algorithm on a PUML diagram with the given parameters.

        Args:
            puml_content:  Contents of the .puml file
            params:        Parameter dict (e.g. {"weight_dependency": 2.3, ...})

        Returns:
            Processed representation - a string included in the LLM prompt.
            Can be: modified PUML, ordered class list, or any other structure.
        """
