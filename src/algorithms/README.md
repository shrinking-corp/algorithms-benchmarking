# algorithms/

This module contains the core optimization and algorithmic logic for the benchmarking tool.

## Contents
- `pso.py`: Particle Swarm Optimization (PSO) implementation for parameter search.
- `memento.py`: Memento pattern for saving and restoring PSO state (checkpoints).
- `base.py`: Base classes for algorithms, providing common interfaces and utilities.


## Design Patterns
- **Memento:** Used for checkpointing and restoring optimization state.
- **Factory:** The `create_optimizer` function in `factory.py` enables adding new optimization algorithms via the factory pattern. To add a new optimizer, implement a new algorithm class and register it in the factory.
- **Factory Method:** For deserializing algorithm mementos.

## Usage
Algorithms in this folder are used by the main benchmarking orchestrator to optimize parameters for diagram shrinking algorithms. Checkpoints are saved after each iteration for crash recovery and analysis.

See the main README for workflow details.