
# data/

This directory contains the dataset, split files, and iterators used for benchmarking the diagram shrinking algorithms.

## Structure
```
data/
  dataset/
    <model_name>/
      *.puml
      python_code.py
    divider/
      train.txt
      validation.txt
  iterator/
    __init__.py
```
- `<model_name>/` - Subdirectories for each model, containing PlantUML diagrams and reference Python code
- `divider/` - Text files listing training and validation splits
- `iterator/` - Python code for dataset iteration (e.g., TrainIterator, ValidationIterator)

## Setup
Clone the dataset repository into the following directory:

```bash
git clone <dataset-repo-url> data/dataset
```

## Usage
- The dataset provides input diagrams and reference code for evaluation.
- Training and validation splits are defined in `divider/`.
- Iterators in `iterator/` are used by the benchmarking tool to access dataset items during optimization and evaluation.

**Note:** The actual dataset content is not included in this repository. Please clone the dataset as described above.

## More Information
- The dataset is used to benchmark the effect of diagram shrinking on LLM-generated code quality.
- For details on the dataset format and usage, see the main project README and code in `data/iterator/`.
