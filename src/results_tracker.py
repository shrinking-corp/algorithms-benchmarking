"""
Storing and managing benchmark results.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .algorithms.memento import AlgorithmMemento

logger = logging.getLogger(__name__)


class ResultsTracker:
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = self.results_path / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_iteration_checkpoint(
        self,
        llm_name: str,
        alg_name: str,
        iteration: int,
        n_iterations: int,
        gbest_fitness: float,
        gbest_params: dict,
        gbest_solution: list,
    ) -> None:
        """Appends one line per PSO iteration to a JSONL checkpoint file."""
        safe_llm = llm_name.replace("/", "_")
        safe_alg = alg_name.replace("/", "_")
        path = self.results_path / f"{self.run_id}_{safe_llm}_{safe_alg}_iterations.jsonl"
        record = {
            "iteration": iteration,
            "n_iterations": n_iterations,
            "best_fitness": gbest_fitness,
            "best_params": gbest_params,
            "best_solution": gbest_solution,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    # ── Memento ──────────────────────────────────────────

    def _memento_path(self, llm_name: str, alg_name: str) -> Path:
        safe_llm = llm_name.replace("/", "_")
        safe_alg = alg_name.replace("/", "_")
        return self._checkpoint_dir / f"{safe_llm}_{safe_alg}.json"

    def save_memento(self, llm_name: str, alg_name: str, memento: AlgorithmMemento) -> None:
        """Overwrites the checkpoint file with the latest optimizer state."""
        path = self._memento_path(llm_name, alg_name)
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(memento.to_storable(), ensure_ascii=False))
            f.flush()

    def load_memento(self, llm_name: str, alg_name: str) -> Optional[AlgorithmMemento]:
        """Returns an AlgorithmMemento if a checkpoint exists, otherwise None."""
        path = self._memento_path(llm_name, alg_name)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info("Checkpoint found for %s / %s at iteration %d", llm_name, alg_name, data.get("iteration", "?"))
        return AlgorithmMemento.deserialize(data)

    def clear_memento(self, llm_name: str, alg_name: str) -> None:
        """Deletes the checkpoint after a successful run."""
        path = self._memento_path(llm_name, alg_name)
        if path.exists():
            path.unlink()
            logger.info("Checkpoint cleared for %s / %s", llm_name, alg_name)

    # ── Done markers ──────────────────────────────────────────────────────

    def _done_path(self, llm_name: str, alg_name: str) -> Path:
        safe_llm = llm_name.replace("/", "_")
        safe_alg = alg_name.replace("/", "_")
        return self._checkpoint_dir / f"{safe_llm}_{safe_alg}.done"

    def mark_done(self, llm_name: str, alg_name: str) -> None:
        """Creates a marker indicating this (llm, alg) combination completed successfully."""
        self._done_path(llm_name, alg_name).touch()
        logger.info("Marked done: %s / %s", llm_name, alg_name)

    def is_done(self, llm_name: str, alg_name: str) -> bool:
        """Returns True if a .done marker exists for this (llm, alg) combination."""
        return self._done_path(llm_name, alg_name).exists()

    def clear_all_done_markers(self) -> None:
        """Deletes all .done markers - called after a fully successful run."""
        for marker in self._checkpoint_dir.glob("*.done"):
            marker.unlink()
            logger.info("Done marker removed: %s", marker.name)

    def save_algorithm_run(
        self,
        llm_name: str,
        alg_name: str,
        best_solution: np.ndarray,
        best_fitness: float,
        best_params: dict,
        baseline_avg_codebleu: float = 0.0,
        final_scores: Optional[Dict[str, float]] = None,
        retention: float = 0.0,
        gbest_fit_history: Optional[list] = None,
        gbest_pos_history: Optional[list] = None,
    ) -> None:
        """Saves the result of a single run (LLM × algorithm)."""
        safe_llm = llm_name.replace("/", "_")
        safe_alg = alg_name.replace("/", "_")
        path = self.results_path / f"{self.run_id}_{safe_llm}_{safe_alg}.json"
        final_scores = final_scores or {}
        data = {
            "run_id": self.run_id,
            "llm": llm_name,
            "algorithm": alg_name,
            "best_fitness": best_fitness,
            "best_params": best_params,
            "best_solution": best_solution.tolist() if hasattr(best_solution, "tolist") else best_solution,
            "baseline_avg_codebleu": baseline_avg_codebleu,
            "final_avg_codebleu": float(sum(final_scores.values()) / len(final_scores)) if final_scores else 0.0,
            "retention": retention,
            "final_scores": final_scores,
            "gbest_fit_history": gbest_fit_history or [],
            "gbest_pos_history": gbest_pos_history or [],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Result saved: %s", path)


