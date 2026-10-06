# An Unscented PrO Kalman Filter

This repository contains the code and instructions needed to reproduce the experiments for:

An Unscented PrO Kalman Filter, Dan Kean, Zheyang Shen, and Chris J Oates.

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
The code to reproduce the unscented and cubature PrO Kalman filters (UKF-PrO, CKF-PrO, UKF-PrO+ and CKF-PrO+) on the Lorenz96 experiments can be found in the attached jupyter notebooks.

## Contact

Dan Kean: [email](mailto:c3039565@newcastle.ac.uk)