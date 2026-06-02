# Predictively oriented Kalman filtering

This repository contains the code and instructions needed to reproduce the experiments for:

Predictively oriented Kalman filtering, Zheyang Shen, Gerardo Duran-Martin, and Chris J Oates. 

## Repository structure
```text
.
├── README.md
├── requirements-full.txt     # Exact package snapshot from the original environment
└── src/                      # Reusable source code
```

## Installation
The experiments were developed using Python 3.11.5.

This implementation depends on an older version of [rebayes-mini](https://github.com/gerdm/rebayes-mini/tree/main), which can be installed under `src/` in editable mode:

```bash
python -m pip install -e .
```

## Experiments
The code to reproduce predictively oriented KF for linear and nonlinear filters can be found in the attached jupyter notebooks.

## Contact

Zheyang Shen: [email](mailto:zheyangshen@gmail.com)

::: 
