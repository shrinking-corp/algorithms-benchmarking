"""
Particle Swarm Optimization (PSO) implementation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

ParticleFn = Callable[[int, int, float, bool, "np.ndarray"], None]

import numpy as np
from tqdm import tqdm

from .base import BaseAlgorithm, ConfigMismatchError, FitnessFn, _diff_snapshots, _snapshot_hash
from .memento import AlgorithmMemento

logger = logging.getLogger(__name__)


@dataclass
class PSOMemento(AlgorithmMemento, memento_type="pso"):
    """
    Concrete Memento for PSO - frozen snapshot of PSO internal state.
    """
    iteration: int           # last completed iteration (1-based)
    config_snapshot: dict    # full PSO + param_space config captured at run start
    positions: list          # shape (n, dim)
    velocities: list         # shape (n, dim)
    pbest_pos: list          # shape (n, dim)
    pbest_fit: list          # shape (n,)
    gbest_fit_history: list  # gbest_fit after each iteration [iter1, iter2, ...]
    gbest_pos_history: list  # gbest_pos after each iteration [iter1, iter2, ...]

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "config_snapshot": self.config_snapshot,
            "positions": self.positions,
            "velocities": self.velocities,
            "pbest_pos": self.pbest_pos,
            "pbest_fit": self.pbest_fit,
            "gbest_fit_history": self.gbest_fit_history,
            "gbest_pos_history": self.gbest_pos_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PSOMemento":
        return cls(
            iteration=d["iteration"],
            config_snapshot=d["config_snapshot"],
            positions=d["positions"],
            velocities=d["velocities"],
            pbest_pos=d["pbest_pos"],
            pbest_fit=d["pbest_fit"],
            gbest_fit_history=d.get("gbest_fit_history", []),
            gbest_pos_history=d.get("gbest_pos_history", []),
        )


class PSOAlgorithm(BaseAlgorithm):
    """Standard PSO with inertia weight."""

    def __init__(
        self,
        name: str,
        param_space,
        n_iterations: int,
        population_size: int,
        inertia: float = 0.729,
        cognitive: float = 1.494,
        social: float = 1.494,
    ):
        super().__init__(name, param_space, n_iterations, population_size)
        self.w = inertia
        self.c1 = cognitive
        self.c2 = social
        self._config_snapshot: dict = self._build_config_snapshot()
        self._config_snapshot_hash: str = _snapshot_hash(self._config_snapshot)

    def _build_config_snapshot(self) -> dict:
        snap = super()._build_config_snapshot()
        snap.update({"inertia": self.w, "cognitive": self.c1, "social": self.c2})
        return snap

    def restore_memento(self, memento: AlgorithmMemento, force: bool = False) -> None:
        """
        Restore PSO state from a checkpoint.

        If the config saved in the memento differs from the current config,
        raises ConfigMismatchError (unless force=True, in which case the
        checkpoint is restored as-is and the current config is ignored for
        the duration of this run).

        Args:
            memento: snapshot to restore from
            force:   if True, skip the config equality check and restore anyway
                     (caller has already confirmed the user wants to use the
                     original parameters stored in the checkpoint)
        """
        assert isinstance(memento, PSOMemento)
        if not force and _snapshot_hash(memento.config_snapshot) != self._config_snapshot_hash:
            mismatches = _diff_snapshots(memento.config_snapshot, self._config_snapshot)
            raise ConfigMismatchError(mismatches)
        if memento.iteration >= self.n_iterations:
            logger.warning(
                "Checkpoint is at iteration %d but n_iterations=%d - "
                "PSO loop will not run. Increase n_iterations or delete the checkpoint.",
                memento.iteration, self.n_iterations,
            )
        self._checkpoint: Optional[PSOMemento] = memento

    async def optimize(
        self,
        fitness_fn: FitnessFn,
        on_iteration: Optional[Callable[[int, float, np.ndarray], None]] = None,
        on_checkpoint: Optional[Callable[["PSOMemento"], None]] = None,
        on_particle: Optional[ParticleFn] = None,
    ) -> Tuple[np.ndarray, float]:
        n, dim = self.population_size, self.n_dim
        checkpoint = getattr(self, "_checkpoint", None)

        # ── Restore from memento or start fresh ──────────────────────────
        if checkpoint is not None:
            start_iteration = checkpoint.iteration
            positions  = np.array(checkpoint.positions)
            velocities = np.array(checkpoint.velocities)
            pbest_pos  = np.array(checkpoint.pbest_pos)
            pbest_fit  = np.array(checkpoint.pbest_fit)
            gbest_fit_history: list = list(checkpoint.gbest_fit_history)
            gbest_pos_history: list = list(checkpoint.gbest_pos_history)
            gbest_fit  = gbest_fit_history[-1]
            gbest_pos  = np.array(gbest_pos_history[-1])
        else:
            start_iteration = 0
            positions  = self._random_pop(n)
            velocities = np.random.uniform(-0.1, 0.1, (n, dim))
            pbest_pos  = positions.copy()
            pbest_fit  = np.full(n, -np.inf)
            gbest_pos  = positions[0].copy()
            gbest_fit  = -np.inf
            gbest_fit_history = []
            gbest_pos_history = []

        with tqdm(total=self.n_iterations, desc=f"PSO [{self.name}]", unit="iter", initial=start_iteration) as pbar:
            for iteration in range(start_iteration, self.n_iterations):

                # ── Evaluate all particles ────────────────────────────────
                for i in range(n):
                    fit = await fitness_fn(positions[i])

                    if fit > pbest_fit[i]:
                        pbest_fit[i] = fit
                        pbest_pos[i] = positions[i].copy()

                    is_new_best = fit > gbest_fit
                    if is_new_best:
                        gbest_fit = fit
                        gbest_pos = positions[i].copy()

                    if on_particle is not None:
                        on_particle(iteration + 1, i + 1, fit, is_new_best, positions[i].copy())

                # ── Update velocities and positions ───────────────────────
                r1 = np.random.rand(n, dim)
                r2 = np.random.rand(n, dim)

                velocities = (
                    self.w * velocities
                    + self.c1 * r1 * (pbest_pos - positions)
                    + self.c2 * r2 * (gbest_pos - positions)
                )
                positions = self._clip(positions + velocities)

                pbar.update(1)
                pbar.set_postfix(best=f"{gbest_fit:.4f}")

                gbest_fit_history.append(gbest_fit)
                gbest_pos_history.append(gbest_pos.tolist())

                if on_iteration is not None:
                    on_iteration(iteration + 1, gbest_fit, gbest_pos.copy())

                # store current state for create_memento()
                self._state = dict(
                    iteration=iteration + 1,
                    positions=positions,
                    velocities=velocities,
                    pbest_pos=pbest_pos,
                    pbest_fit=pbest_fit,
                    gbest_fit_history=gbest_fit_history,
                    gbest_pos_history=gbest_pos_history,
                )
                if on_checkpoint is not None:
                    on_checkpoint(self.create_memento())

        return gbest_pos, gbest_fit

    def create_memento(self) -> PSOMemento:
        """Capture current PSO state as a PSOMemento."""
        s = self._state
        return PSOMemento(
            iteration=s["iteration"],
            config_snapshot=self._config_snapshot,
            positions=s["positions"].tolist(),
            velocities=s["velocities"].tolist(),
            pbest_pos=s["pbest_pos"].tolist(),
            pbest_fit=s["pbest_fit"].tolist(),
            gbest_fit_history=list(s["gbest_fit_history"]),
            gbest_pos_history=list(s["gbest_pos_history"]),
        )
