# src/

This folder contains the main source code for the benchmarking tool. It is organized into modules for orchestration, fitness calculation, logging, metrics, results tracking, and subpackages for algorithms, LLM clients, and target algorithm adapters.

## Main Modules
- `benchmarker.py`: Orchestrates the benchmarking workflow (dataset iteration, LLM calls, optimization, evaluation).
- `fitness.py`: Computes fitness scores using CodeBLEU and other metrics.
- `log_messages.py`: Centralized log message templates for consistent logging.
- `metrics.py`: Wrapper for CodeBLEU and metric calculations.
- `results_tracker.py`: Handles saving results and PSO checkpoints.

## Subfolders
- `algorithms/`: PSO implementation, Memento pattern, base algorithm classes.
- `llm/`: LLM client abstractions, caching, and client factories.
- `target_algorithms/`: Adapters for diagram shrinking algorithms (Kruskal, Genetic).

See each subfolder's README for more details on their specific roles.