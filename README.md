# ASWS Benchmarking Tool

## Overview

This project provides a benchmarking tool for evaluating PlantUML diagram shrinking algorithms using LLM-generated code quality as the metric. It automates the process of optimizing algorithm parameters (via PSO), running LLMs on original and shrunk diagrams, and comparing the generated code to reference implementations using CodeBLEU.

- **Optimization algorithms:** Kruskal, Genetic (Evolutionary)
- **LLM backends:** OpenAI, Ollama, LiteLLM (all with disk cache)
- **Metric:** CodeBLEU
- **Dataset:** PlantUML diagrams + reference Python code

## Features
- Automated parameter search for diagram shrinking algorithms (PSO)
- Modular LLM client with disk/in-memory caching
- Robust logging and checkpointing for resumable runs
- Extensible architecture with design patterns (Template Method, Proxy, Adapter, Memento, Factory, Iterator)

## Installation

1. **Clone the repository**

2. **Install dependencies**

```bash
py -3.12 -m pip install -e .
```

3. **Set up API key**

Create a copy of `.env` file and insert your OpenAI API key:

```bash
echo OPENAI_API_KEY=sk-... > .env
```

4. **Download the dataset**

Clone the dataset repository into the correct location:

```bash
git clone <dataset-url> data/dataset
```

## Usage

Run the benchmarking tool:

```bash
py -3.12 main.py
```

Results are saved in the `results/` directory as JSON and JSONL files.

## Project Structure

```
config.yaml           # All configuration (single source of truth)
main.py               # Entry point
src/
  benchmarker.py      # Main orchestration
  fitness.py          # CodeBLEU fitness calculation
  log_messages.py     # Centralized log message templates
  metrics.py          # CodeBLEU wrapper
  results_tracker.py  # Results and checkpoint management
  algorithms/         # PSO, Memento, BaseAlgorithm
  llm/                # LLM clients, caching, factory
  target_algorithms/  # Adapters for shrinking algorithms
  ...
data/
  iterator/           # TrainIterator, ValidationIterator
  dataset/            # PlantUML diagrams + reference code (gitignored)
results/              # Output files (gitignored)
logs/                 # Log files (gitignored)
.cache/llm/           # LLM disk cache (gitignored)
```

## How It Works

1. **Baseline:** For each LLM/model, run on original diagrams and compute baseline CodeBLEU.
2. **PSO Optimization:** For each algorithm, optimize parameters on the training set using PSO (fitness = CodeBLEU of LLM output on shrunk diagrams).
3. **Final Evaluation:** Evaluate best parameters on the validation set.
4. **Retention:** Compute retention = final_avg / baseline_avg (measures improvement from shrinking).

Checkpoints and logs are created for every run, allowing for crash recovery and detailed analysis.

## Configuration

All settings (LLM models, algorithms, parameter spaces, dataset splits, etc.) are defined in `config.yaml`.

## Logging & Checkpoints
- Logs: `logs/<run_id>.log` (streamed to stdout)
- Checkpoints: `results/checkpoints/` (PSO state per iteration)
- Results: `results/<run_id>_<llm>/<model>_<alg>.json` (summary), `..._iterations.jsonl` (per-iteration)


## Extending
- **Add new LLMs:** Implement a client in `src/llm/` and register in the factory.
- **Add new target algorithms:** Implement an adapter in `src/target_algorithms/` and register in the factory.
- **Add new optimization algorithms:** Implement a new optimizer class in `src/algorithms/` and register it in the `create_optimizer` factory function (`src/algorithms/factory.py`). The factory pattern allows easy extensibility for new optimization strategies.
- **Add metrics:** Extend `src/metrics.py` and `src/fitness.py`.
