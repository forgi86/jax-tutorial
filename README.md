# JAX Tutorial

This repository contains a Jupyter notebook that provides a step-by-step introduction to JAX - a high-performance numerical computing library from Google Research.
It is meant for researchers/practitioners that are already familiar with deep learning in another environment, most likely PyTorch

## Contents

- `00_from_zero_to_noob.ipynb`: A beginner-level tutorial notebook that introduces JAX fundamentals, covering:
  - Numpy-style array operations
  - Automatic differentiation and optimization
  - Parallel execution with vmap
  - Introduction to PyTrees
  - Just-In-Time compilation with jit
  - Random numbers
  - Introduction to flax
  - Recurrent nets with scan

- `01_meta_learning_maml.ipynb` Meta learning with MAML, implementation sketch

- `02_meta_learning_hypernet.ipynb` Meta learning with Hypernets, implementation sketch

## Getting Started

1. Make sure you have Python installed
2. Install dependencies:
```bash
pip install jax jaxlib jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `00_from_zero_to_noob.ipynb` to begin learning JAX