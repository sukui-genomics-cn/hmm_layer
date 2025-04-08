# Hidden Markov Model (HMM) Layer for Genomic Structure Prediction

![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)

## Overview

This repository implements a Torch-based Hidden Markov Model (HMM) Layer designed for genomic structure prediction. The implementation draws inspiration from two notable projects:
- [Tiberius](https://github.com/Gaius-Augustus/Tiberius)
- [learnMSA](https://github.com/Gaius-Augustus/learnMSA)

The HMM Layer provides a modular framework that can be integrated into deep learning pipelines for sequence analysis tasks.

## Project Structure

The repository is organized as follows:

- `hmm_layer/` - Core implementation of the HMM Layer.
  - `BaseRNN.py` - Base RNN implementation.
  - `Bidirectional.py` - Bidirectional RNN support.
  - `Emitter.py` - Emission probability manager.
  - `HMMCell.py` - Core HMM cell implementing the forward algorithm.
  - `MsaHmmCell.py` - HMM cell for multiple sequence alignment.
  - `MsaHMMLayer.py` - HMM layer for multiple sequence alignment.
  - `TotalProbabilityCell.py` - Total probability computation cell.
  - `Transitioner.py` - State transition probability manager.
- `requirements.txt` - List of dependencies.
- `README.md` - Project documentation.
- `setup.py` - Installation script for the package.

## Core Modules

### 1. Transitioner
Implements state transition probabilities with efficient matrix operations. Handles both learned and constrained transition parameters.

### 2. Emitter
Manages emission probabilities with customizable distributions. Supports various observation models suitable for genomic data.

### 3. HMM Cell
The fundamental computational unit implementing the forward algorithm. Optimized for batched operations on sequence data.

### 4. HMM Layer
The main interface layer that can be integrated into neural networks. Handles sequence input/output and connects with other network components.

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- Python 3.9
- torch
- NumPy
- SciPy


## Usage Example
```python
import torch

from hmm_layer import GenePredHMMLayer

dim = 15
stacked_inputs = torch.randn(size=(1,2,9999,20))
layer = GenePredHMMLayer(
    parallel_factor=99
)
outputs = layer(
    inputs=stacked_inputs,
    training=True,
)
print(f'outputs shape: {outputs.shape}')
```

## References

1. Tiberius: [GitHub Repository](https://github.com/Gaius-Augustus/Tiberius)
2. learnMSA: [GitHub Repository](https://github.com/Gaius-Augustus/learnMSA)
3. Original HMM papers: Rabiner, L. R. (1989)

## Contributing

Contributions are welcome. Please submit pull requests or open issues for discussion.