"""
Main benchmark orchestration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data.iterator import TrainIterator, ValidationIterator

from .algorithms import create_optimizer
from .algorithms.base import BaseAlgorithm, ConfigMismatchError
from .fitness import FitnessEvaluator
from . import log_messages as MSG
from .llm.factory import create_llm_client
from .results_tracker import ResultsTracker
from .target_algorithms import create_target_algorithm

logger = logging.getLogger(__name__)


def _ask_resume() -> bool:
    print(
        "\nDo you want to resume with the ORIGINAL parameters from the checkpoint?\n"
        "  yes - continue from checkpoint (original parameters)\n"
        "  no  - start fresh with the current (new) parameters\n"
    )
    while True:
        try:
            answer = input("Your choice [yes/no]: ").strip().lower()
        except EOFError:
            logger.warning("Non-interactive terminal detected, defaulting to 'no' (start fresh).")
            return False
        if answer in ("yes", "y", "Yes", "Y"):
            return True
        if answer in ("no", "n", "No", "N"):
            return False
        print("Please type 'yes' or 'no'.")


class Benchmarker:
    def __init__(self, config: dict):
        bench = config["benchmark"]
        self.n_iterations: int = bench["n_iterations"]
        self.population_size: int = bench["population_size"]
        self.language: str = bench.get("language", "python")
        self.max_concurrent: int = bench.get("max_concurrent_llm", 5)
        self.request_timeout: float = float(bench.get("request_timeout", 90.0))
        cache_dir: str = bench.get("cache_dir", ".cache/llm")
        self.prompt_template: str = bench["user_prompt_template"]

        self.llm_clients = [
            create_llm_client(cfg, cache_dir=cache_dir, timeout=self.request_timeout) for cfg in config["llms"]
        ]
        self.target_algorithm_configs = config["target_algorithms"]
        self.optimizer_config = config.get("optimizer", {})
        self.results_tracker = ResultsTracker(bench["results_path"])

    def _collect_train_items(self) -> List[Tuple[str, str]]:
        return list(TrainIterator())

    def _collect_val_items(self) -> List[Tuple[str, str]]:
        return list(ValidationIterator())

    async def run(self) -> Dict[str, Any]:
        """
        Runs the full benchmark.

        Returns:
            { llm_name: { algorithm_name: { best_fitness, best_params, ... } } }
        """
        all_results: Dict[str, Any] = {}

        val_items = self._collect_val_items()
        train_items = self._collect_train_items()

        if not val_items:
            logger.error("Validation set is empty - check dataset/divider/validation.txt")
            return all_results
        if not train_items:
            logger.error("Training set is empty - check dataset/divider/train.txt")
            return all_results

        logger.info(
            MSG.DATASET_SIZES.format(train=len(train_items), val=len(val_items))
        )

        for llm in self.llm_clients:
            all_results[llm.name] = {}
            logger.info(MSG.CLIENT_START.format(llm=llm.name))

            for model in llm.models:
                run_key = f"{llm.name}/{model}"
                all_results[llm.name][model] = {}
                logger.info(MSG.MODEL_START.format(llm=llm.name, model=model))

                # Baseline: CodeBLEU(ref_code, LLM(original_puml)) on validation set.
                baseline_evaluator = FitnessEvaluator(
                    llm=llm,
                    dataset=val_items,
                    language=self.language,
                    max_concurrent=self.max_concurrent,
                    prompt_template=self.prompt_template,
                    model=model,
                )
                baseline_scores = await baseline_evaluator.compute_baseline_scores()
                baseline_avg = (
                    float(np.mean(list(baseline_scores.values()))) if baseline_scores else 0.0
                )
                logger.info(
                    MSG.BASELINE_DONE.format(llm=llm.name, model=model, score=baseline_avg)
                )

                for alg_cfg in self.target_algorithm_configs:
                    target_alg = create_target_algorithm(alg_cfg)

                    if self.results_tracker.is_done(run_key, target_alg.name):
                        logger.info(
                            MSG.ALG_SKIPPED.format(llm=llm.name, model=model, alg=target_alg.name)
                        )
                        continue

                    logger.info(MSG.ALG_START.format(llm=llm.name, model=model, alg=target_alg.name))

                    optimizer = create_optimizer(
                        optimizer_cfg=self.optimizer_config,
                        param_space=target_alg.param_space,
                        n_iterations=self.n_iterations,
                        population_size=self.population_size,
                    )

                    # PSO trains on the training split
                    train_evaluator = FitnessEvaluator(
                        llm=llm,
                        dataset=train_items,
                        optimizer=optimizer,
                        target_algorithm=target_alg,
                        language=self.language,
                        max_concurrent=self.max_concurrent,
                        prompt_template=self.prompt_template,
                        model=model,
                    )

                    def make_checkpoint_callback(
                        llm_name: str, model_name: str, alg_name: str, opt: BaseAlgorithm
                    ):
                        def _callback(iteration: int, gbest_fitness: float, gbest_pos) -> None:
                            self.results_tracker.save_iteration_checkpoint(
                                llm_name=llm_name,
                                alg_name=alg_name,
                                iteration=iteration,
                                n_iterations=self.n_iterations,
                                gbest_fitness=gbest_fitness,
                                gbest_params=opt.interpret(gbest_pos),
                                gbest_solution=gbest_pos.tolist(),
                            )
                            logger.info(
                                MSG.PSO_ITER_DONE.format(
                                    llm=llm_name, model=model_name, alg=alg_name,
                                    iter=iteration, n_iter=self.n_iterations,
                                    gbest=gbest_fitness,
                                )
                            )
                        return _callback

                    def make_particle_callback(
                        llm_name: str, model_name: str, alg_name: str, opt: BaseAlgorithm
                    ):
                        n_iter = self.n_iterations
                        n_agents = self.population_size

                        def _callback(
                            iteration: int,
                            particle: int,
                            fitness: float,
                            is_new_best: bool,
                            position: "np.ndarray",
                        ) -> None:
                            if is_new_best:
                                logger.info(
                                    MSG.PSO_NEW_BEST.format(
                                        llm=llm_name, model=model_name, alg=alg_name,
                                        iter=iteration, n_iter=n_iter,
                                        agent=particle, n_agents=n_agents,
                                        fitness=fitness,
                                        params=opt.interpret(position),
                                    )
                                )
                            else:
                                logger.info(
                                    MSG.PSO_PARTICLE.format(
                                        llm=llm_name, model=model_name, alg=alg_name,
                                        iter=iteration, n_iter=n_iter,
                                        agent=particle, n_agents=n_agents,
                                        fitness=fitness,
                                    )
                                )

                        return _callback

                    def make_memento_callback(llm_name: str, alg_name: str):
                        def _callback(memento) -> None:
                            self.results_tracker.save_memento(llm_name, alg_name, memento)
                        return _callback

                    checkpoint = self.results_tracker.load_memento(run_key, target_alg.name)
                    if checkpoint is not None:
                        try:
                            optimizer.restore_memento(checkpoint)
                            logger.info(
                                MSG.PSO_RESUMED.format(
                                    llm=llm.name, model=model, alg=target_alg.name,
                                    from_iter=checkpoint.iteration,
                                    n_iter=self.n_iterations,
                                    gbest=checkpoint.gbest_fit_history[-1]
                                    if checkpoint.gbest_fit_history else 0.0,
                                )
                            )
                        except ConfigMismatchError as exc:
                            print(
                                f"\n[WARNING] Checkpoint config differs from current config "
                                f"({run_key} / {target_alg.name}):\n"
                                + "\n".join(exc.mismatches)
                            )
                            answer = _ask_resume()
                            if answer:
                                logger.info("User chose to resume with original checkpoint parameters.")
                                optimizer.restore_memento(checkpoint, force=True)
                            else:
                                logger.info("User chose to start fresh with new parameters.")
                                self.results_tracker.clear_memento(run_key, target_alg.name)

                    best_solution, best_fitness = await optimizer.optimize(
                        train_evaluator.evaluate,
                        on_iteration=make_checkpoint_callback(run_key, model, target_alg.name, optimizer),
                        on_checkpoint=make_memento_callback(run_key, target_alg.name),
                        on_particle=make_particle_callback(llm.name, model, target_alg.name, optimizer),
                    )
                    best_params = optimizer.interpret(best_solution)
                    logger.info(
                        MSG.PSO_RESULT.format(
                            llm=llm.name, model=model, alg=target_alg.name,
                            fitness=best_fitness, params=best_params,
                        )
                    )

                    # Final evaluation on the validation split
                    val_evaluator = FitnessEvaluator(
                        llm=llm,
                        dataset=val_items,
                        optimizer=optimizer,
                        target_algorithm=target_alg,
                        language=self.language,
                        max_concurrent=self.max_concurrent,
                        prompt_template=self.prompt_template,
                        model=model,
                    )
                    final_scores = await val_evaluator.compute_final_scores(best_params)
                    final_avg = (
                        float(np.mean(list(final_scores.values()))) if final_scores else 0.0
                    )
                    retention = final_avg / baseline_avg if baseline_avg > 0.0 else 0.0

                    gbest_fit_history = list(optimizer._state.get("gbest_fit_history", []))
                    gbest_pos_history = list(optimizer._state.get("gbest_pos_history", []))

                    result = {
                        "best_fitness": best_fitness,
                        "best_params": best_params,
                        "best_solution": best_solution.tolist(),
                        "n_evaluations": train_evaluator.eval_count,
                        "baseline_avg_codebleu": baseline_avg,
                        "final_scores": final_scores,
                        "final_avg_codebleu": final_avg,
                        "retention": retention,
                        "gbest_fit_history": gbest_fit_history,
                        "gbest_pos_history": gbest_pos_history,
                    }
                    all_results[llm.name][model][target_alg.name] = result

                    self.results_tracker.save_algorithm_run(
                        llm_name=run_key,
                        alg_name=target_alg.name,
                        best_solution=best_solution,
                        best_fitness=best_fitness,
                        best_params=best_params,
                        baseline_avg_codebleu=baseline_avg,
                        final_scores=final_scores,
                        retention=retention,
                        gbest_fit_history=gbest_fit_history,
                        gbest_pos_history=gbest_pos_history,
                    )
                    self.results_tracker.clear_memento(run_key, target_alg.name)
                    self.results_tracker.mark_done(run_key, target_alg.name)
                    logger.info(
                        MSG.FINAL_RESULT.format(
                            llm=llm.name, model=model, alg=target_alg.name,
                            baseline=baseline_avg, final=final_avg, retention=retention,
                        )
                    )
                    logger.info(MSG.ALG_END.format(llm=llm.name, model=model, alg=target_alg.name))

                logger.info(MSG.MODEL_END.format(llm=llm.name, model=model))

            logger.info(MSG.CLIENT_END.format(llm=llm.name))

        self.results_tracker.clear_all_done_markers()
        return all_results
