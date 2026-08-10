# Custom APRL Algorithm: Baseline Benchmark

Results use 5-fold cross-validation with seed `42`. Classification reports accuracy; regression reports R². Values are mean ± population standard deviation across folds.

> These small built-in datasets are smoke benchmarks, not evidence of state-of-the-art performance or scientific novelty. Runtime is machine-dependent.

## Classification

| Dataset | Model | Score | Runtime (s) |
|---|---|---:|---:|
| Iris | APRL | 0.953 ± 0.045 | 0.030 |
| Iris | Logistic regression | 0.953 ± 0.045 | 0.041 |
| Iris | K-nearest neighbors | 0.973 ± 0.025 | 0.022 |
| Iris | Random forest | 0.960 ± 0.039 | 1.477 |
| Wine | APRL | 0.983 ± 0.014 | 0.044 |
| Wine | Logistic regression | 0.983 ± 0.014 | 0.046 |
| Wine | K-nearest neighbors | 0.972 ± 0.018 | 0.023 |
| Wine | Random forest | 0.977 ± 0.021 | 1.291 |
| Breast cancer | APRL | 0.968 ± 0.013 | 0.131 |
| Breast cancer | Logistic regression | 0.974 ± 0.017 | 0.042 |
| Breast cancer | K-nearest neighbors | 0.963 ± 0.018 | 2.193 |
| Breast cancer | Random forest | 0.954 ± 0.010 | 1.910 |

## Regression

| Dataset | Model | Score | Runtime (s) |
|---|---|---:|---:|
| Diabetes | APRL | 0.492 ± 0.078 | 0.053 |
| Diabetes | Ridge | 0.479 ± 0.083 | 0.014 |
| Diabetes | K-nearest neighbors | 0.391 ± 0.040 | 0.017 |
| Diabetes | Random forest | 0.427 ± 0.093 | 1.924 |
| Friedman #1 | APRL | 0.899 ± 0.021 | 0.063 |
| Friedman #1 | Ridge | 0.745 ± 0.030 | 0.014 |
| Friedman #1 | K-nearest neighbors | 0.648 ± 0.047 | 0.019 |
| Friedman #1 | Random forest | 0.827 ± 0.022 | 2.733 |

## Environment

- Python: `3.12.10`
- NumPy: `2.5.2`
- scikit-learn: `1.9.0`

Regenerate from the repository root:

```bash
python benchmarks/benchmark_aprl.py --output algorithms/aprl/BENCHMARKS.md
```
