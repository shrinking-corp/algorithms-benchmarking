# target_algorithms/

This module contains adapters for diagram shrinking algorithms used in the benchmarking process.

## Contents
- `kruskal.py`: Adapter for Kruskal-based diagram shrinking (KruskalDiagramShrinker from PyPI).
- `genetic.py`: Adapter for Genetic/Evolutionary shrinking (EvolDiagramShrinker from PyPI).
- `base.py`: Abstract base class for target algorithms.
- `factory.py`: Factory for creating target algorithm adapters based on configuration.

## Features
- Maps PSO-optimized parameters to the APIs of external shrinking libraries
- Ensures consistent interface for all shrinking algorithms
- Parameter spaces are defined exclusively in `config.yaml` (not in code)

## Design Patterns
- **Adapter:** For integrating external shrinking algorithms.
- **Factory:** For adapter instantiation.

Target algorithm adapters are used by the benchmarking orchestrator to shrink diagrams before LLM evaluation.