# Hidden Markov Model (HMM) Layer for Genomic Structure Prediction

![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.10.0-orange.svg)

## Overview

This repository implements a Torch-based Hidden Markov Model (HMM) Layer designed for genomic structure prediction. The implementation draws inspiration from two notable projects:
- [Tiberius](https://github.com/Gaius-Augustus/Tiberius)
- [learnMSA](https://github.com/Gaius-Augustus/learnMSA)

The HMM Layer provides a modular framework that can be integrated into deep learning pipelines for sequence analysis tasks.

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
- TensorFlow 2.10.0
- NumPy
- SciPy


## References

1. Tiberius: [GitHub Repository](https://github.com/Gaius-Augustus/Tiberius)
2. learnMSA: [GitHub Repository](https://github.com/Gaius-Augustus/learnMSA)
3. Original HMM papers: Rabiner, L. R. (1989)

## Contributing

Contributions are welcome. Please submit pull requests or open issues for discussion.