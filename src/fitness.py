"""
Fitness evaluation for PSO and validation.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .metrics import calculate_codebleu

if TYPE_CHECKING:
    from .algorithms.context import OptimizerContext
    from .llm.client import LLMClient
    from .target_algorithms.context import ShrinkingContext

logger = logging.getLogger(__name__)

# Each item: (reference_code, puml)
DatasetItem = Tuple[str, str]


class FitnessEvaluator:
    """
    Client in the Strategy pattern.

    Uses ShrinkingContext to process PUML diagrams before LLM evaluation.
    The concrete strategy (which algorithm to apply) is decided externally
    by Benchmarker via context.set_strategy() before this evaluator is used.
    """

    def __init__(
        self,
        llm: "LLMClient",
        dataset: List[DatasetItem],
        prompt_template: str,
        context: "Optional[ShrinkingContext]" = None,
        optimizer_context: "Optional[OptimizerContext]" = None,
        language: str = "python",
        max_concurrent: int = 5,
        model: Optional[str] = None,
    ):
        self.llm = llm
        self.model = model
        self.dataset = dataset
        self.context = context
        self.optimizer_context = optimizer_context
        self.language = language
        self.prompt_template = prompt_template
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.eval_count = 0

    def build_prompt(self, puml: str) -> str:
        """Formats the user-turn message using the configured prompt template.
        Supported placeholders: {puml}, {language}."""
        return self.prompt_template.format(puml=puml, language=self.language)

    async def evaluate(self, solution: np.ndarray) -> float:
        """
        PSO fitness function - called on training set.

        fitness = AVG CodeBLEU(reference_code, LLM(context.execute(puml, params)))
        """
        if self.optimizer_context is None or self.context is None:
            raise RuntimeError("evaluate() requires both optimizer_context and context to be set.")
        self.eval_count += 1
        params = self.optimizer_context.interpret(solution)

        tasks = [self._score_shrunken(ref, puml, params) for ref, puml in self.dataset]
        scores = await asyncio.gather(*tasks)

        avg = float(np.mean(scores)) if scores else 0.0
        logger.debug("Eval #%d  avg=%.4f  params=%s", self.eval_count, avg, params)
        return avg

    async def compute_final_scores(self, best_params: dict) -> Dict[int, float]:
        """
        Per-item CodeBLEU scores on the validation set using best_params.

        Delegates processing to the active strategy via context.execute().
        Returns {index: score}.
        """
        if self.context is None:
            raise RuntimeError("compute_final_scores() requires context to be set.")
        tasks = [self._score_shrunken(ref, puml, best_params) for ref, puml in self.dataset]
        scores = await asyncio.gather(*tasks)
        return {i: float(s) for i, s in enumerate(scores)}

    async def compute_baseline_scores(self) -> Dict[int, float]:
        """
        Baseline per-item CodeBLEU scores - no strategy applied.

        baseline = CodeBLEU(reference_code, LLM(original_puml))
        The original unmodified diagram is sent directly to the LLM.

        Returns {index: score}.
        """
        tasks = [self._score_original(ref, puml) for ref, puml in self.dataset]
        scores = await asyncio.gather(*tasks)
        return {i: float(s) for i, s in enumerate(scores)}

    # ── private helpers ────────────────────────────────────────────────────

    async def _query(self, prompt: str) -> str:
        if self.model is not None:
            return await self.llm.query_model(self.model, prompt)
        return await self.llm.query(prompt)

    async def _score_shrunken(self, reference_code: str, puml: str, params: dict) -> float:
        """CodeBLEU(reference_code, LLM(context.execute(puml, params)))."""
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            shrunken = await loop.run_in_executor(
                None, functools.partial(self.context.execute, puml, params)
            )
            prompt = self.build_prompt(shrunken)
            generated = await self._query(prompt)
            return calculate_codebleu(reference_code, generated, self.language)

    async def _score_original(self, reference_code: str, puml: str) -> float:
        """CodeBLEU(reference_code, LLM(original_puml))."""
        async with self._semaphore:
            prompt = self.build_prompt(puml)
            generated = await self._query(prompt)
            return calculate_codebleu(reference_code, generated, self.language)
