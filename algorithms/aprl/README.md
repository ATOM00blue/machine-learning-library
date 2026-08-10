# Custom Machine-Learning Algorithm: APRL

APRL is an experimental supervised-learning algorithm for small and medium-sized
numeric tabular datasets. It combines a conservative global prediction with
relevance-weighted prototypes and local polynomial experts that learn only the
global model's residual errors.

The implementation supports:

- classification and single- or multi-output regression;
- median imputation and automatic feature scaling;
- feature-relevance-weighted distances;
- nonlinear local residual experts;
- distance-aware uncertainty estimates;
- exact incremental updates through `partial_fit`;
- a scikit-learn-like `fit` / `predict` / `score` interface;
- NumPy as its only runtime dependency.

> **Research status:** APRL is an original implementation and custom composition
> created for this repository. Its ingredients have substantial prior art. We do
> not claim that the composition is scientifically novel or state of the art.
> See [RELATED_WORK.md](RELATED_WORK.md) and [BENCHMARKS.md](BENCHMARKS.md).

## Install

From the repository root:

```bash
python -m pip install -e .
```

For tests and benchmarks:

```bash
python -m pip install -e ".[dev,benchmark]"
python -m pytest
python benchmarks/benchmark_aprl.py
```

## Quick start

```python
import numpy as np

from aprl import APRLClassifier, APRLRegressor

rng = np.random.default_rng(7)
X = rng.normal(size=(500, 5))
y = np.sin(X[:, 0]) + X[:, 1] ** 2 - 0.4 * X[:, 2]

model = APRLRegressor(random_state=7).fit(X[:400], y[:400])
prediction, uncertainty = model.predict_with_uncertainty(X[400:])
print(model.score(X[400:], y[400:]))
```

Classification uses the same interface:

```python
labels = np.where(X[:, 0] * X[:, 1] + X[:, 2] > 0, "yes", "no")
classifier = APRLClassifier(random_state=7).fit(X[:400], labels[:400])
probability, uncertainty = classifier.predict_proba_with_uncertainty(X[400:])
```

Run the complete example with:

```bash
python examples/aprl_quickstart.py
```

## Why combine global and local learning?

A global linear model is stable and extrapolates predictably, but it misses
nonlinear local structure. A purely local learner can model that structure, but
its behavior far from observed data is often unreliable. APRL gives the two
components separate jobs:

1. the global model learns the broad trend;
2. local prototype experts learn only the remaining error;
3. a confidence gate gradually removes the local correction away from known data.

The intended benefit is nonlinear accuracy in familiar regions with a controlled
fallback outside them.

## Algorithm

Let `X` contain `n` samples and `d` numeric features. Let `Y` be either regression
targets or one-hot class targets.

### 1. Preprocess and measure feature relevance

Missing values are median-imputed and every feature is standardized, producing
`z`. APRL estimates a non-negative relevance value for each feature:

- regression uses the mean absolute feature-target correlation;
- classification uses a between-class to within-class variance ratio.

The relevance is normalized and given a configurable floor `f`:

```text
w_j = f + (1 - f) relevance_j
u_j = sqrt(w_j) z_j
```

The floor prevents a feature with weak marginal relevance but useful interactions
from disappearing entirely. Prototype distances are calculated in `u` space;
the prediction models use standardized `z` space.

### 2. Fit a global model

APRL fits a multi-output ridge model with an unregularized intercept:

```text
B = argmin_B ||Y - ZB||² + lambda_global ||B_without_intercept||²
```

For regression, its output is the global prediction. For classification, the
outputs are converted to probabilities with softmax.

### 3. Discover prototypes

APRL runs k-means++ followed by Lloyd iterations in relevance-weighted feature
space. By default, the prototype count is:

```text
K = clip(round(sqrt(n)), 2, 32)
```

The bandwidth `h` is the median distance from each training point to its nearest
prototype, multiplied by `bandwidth_scale`.

For input `x`, soft membership in prototype `k` is:

```text
a_k(x) = exp(-||u(x) - c_k||² / (2h²)) / sum_j exp(-||u(x) - c_j||² / (2h²))
```

### 4. Learn local residual experts

Each prototype fits a distance-weighted ridge model to the error left by the
global model. With the default `local_degree=2`, the local feature map is:

```text
phi_k(z) = [z - c_k, (z - c_k)²]
```

For regression, the residual target is `Y - global_prediction`. For
classification, it is `one_hot(Y) - global_probability`.

### 5. Gate the correction

APRL combines nearest-prototype distance with the effective mass of nearby
prototypes. The distance gate is deliberately flat near known data and falls
quickly outside it:

```text
distance_gate(x) = 1 / (1 + (nearest_distance(x) / (2.5h))^4)
confidence(x) = distance_gate(x) * density_gate(x)
```

The final regression prediction is:

```text
prediction(x) = global(x)
              + residual_strength * confidence(x)
                * sum_k a_k(x) local_residual_k(x)
```

Classification applies the same correction to the global probability vector,
clips negative values, and normalizes the result to sum to one.

## Uncertainty

`APRLRegressor.predict_with_uncertainty` combines locally weighted residual
variance, global residual variance, global/local disagreement, and a penalty for
distance from prototypes.

`APRLClassifier.predict_proba_with_uncertainty` combines normalized predictive
entropy with the same prototype-confidence gate. Classification uncertainty is in
`[0, 1]`.

These values are useful ranking and out-of-distribution heuristics. They are **not
calibrated confidence intervals and carry no coverage guarantee**. Use a held-out
calibration method such as conformal prediction when guarantees are required.

## Incremental updates

Both estimators implement `partial_fit`. Version 0.1 stores the batches observed
so far and exactly refits all APRL components after each batch. This is slower than
a true online update but avoids inconsistent prototypes, feature relevance, and
uncertainty statistics.

```python
model = APRLRegressor(random_state=0)
model.partial_fit(X_batch_1, y_batch_1)
model.partial_fit(X_batch_2, y_batch_2)
```

Set `store_training_data=False` to reduce retained memory when `partial_fit` is
not needed.

## Main parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `n_prototypes` | `None` | Automatic `sqrt(n)` rule, capped at 32 |
| `regularization` | `1.0` | Global ridge penalty |
| `local_regularization` | `0.5` | Local expert ridge penalty |
| `local_degree` | `2` | Local linear (`1`) or diagonal-quadratic (`2`) features |
| `residual_strength` | `1.0` | Strength of the gated local correction |
| `bandwidth_scale` | `1.0` | Multiplier for the data-derived RBF bandwidth |
| `feature_weight_floor` | `0.15` | Minimum distance relevance for any feature |
| `max_iter` | `100` | Maximum k-means iterations |
| `random_state` | `None` | Reproducible prototype initialization |

Learned attributes end in `_`, including `prototype_centers_`,
`feature_weights_`, `bandwidth_`, and `classes_`.

## Complexity

With `p = d` for degree 1 or `p = 2d` for degree 2:

- prototype discovery: approximately `O(iterations * n * K * d)`;
- local weighted fits: `O(K * (n * p² + p³))`;
- prediction: `O(m * K * p)` for `m` query samples;
- retained training memory: `O(nd)` when incremental updates are enabled.

APRL is therefore aimed at numeric tabular data with tens or low hundreds of
features, not raw images, text tokens, or very high-dimensional sparse matrices.

## Limitations

- Numeric dense arrays only; categoricals must be encoded by the caller.
- Median imputation is intentionally simple and may be inappropriate for some data.
- Marginal feature relevance can undervalue features useful only in interactions;
  the relevance floor mitigates but does not eliminate this issue.
- K-means favors roughly spherical regions and can degrade in high dimensions.
- Quadratic local experts use squared terms but not all pairwise interactions.
- Exact `partial_fit` retains and retrains on all data; it is not streaming-scale.
- Uncertainty is heuristic rather than statistically calibrated.
- Current benchmarks are small; use nested validation on the actual target problem.

## Project files

```text
src/aprl/_core.py          shared numerical algorithm
src/aprl/classifier.py     classification API
src/aprl/regressor.py      regression API
tests/                     behavioral and correctness tests
examples/aprl_quickstart.py
benchmarks/benchmark_aprl.py
algorithms/aprl/BENCHMARKS.md
algorithms/aprl/RELATED_WORK.md
```
