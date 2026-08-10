"""Shared numerical core for APRL estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_2d_float(X: ArrayLike, *, allow_nan: bool = True) -> FloatArray:
    """Return a validated, two-dimensional float array."""
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {array.shape!r}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("X must contain at least one sample and one feature")
    if np.isinf(array).any():
        raise ValueError("X cannot contain positive or negative infinity")
    if not allow_nan and np.isnan(array).any():
        raise ValueError("X cannot contain NaN values")
    return array


def solve_ridge(
    X: FloatArray,
    y: FloatArray,
    regularization: float,
    sample_weight: FloatArray | None = None,
) -> FloatArray:
    """Fit a multi-output ridge model with an unregularized intercept."""
    design = np.column_stack((np.ones(X.shape[0]), X))
    if sample_weight is None:
        lhs = design.T @ design
        rhs = design.T @ y
    else:
        root_weight = np.sqrt(np.maximum(sample_weight, 0.0))[:, None]
        weighted_design = design * root_weight
        weighted_y = y * root_weight
        lhs = weighted_design.T @ weighted_design
        rhs = weighted_design.T @ weighted_y

    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    lhs = lhs + penalty
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def squared_distances(X: FloatArray, centers: FloatArray) -> FloatArray:
    """Compute stable pairwise squared Euclidean distances."""
    x_norm = np.sum(X * X, axis=1)[:, None]
    center_norm = np.sum(centers * centers, axis=1)[None, :]
    distances = x_norm + center_norm - 2.0 * (X @ centers.T)
    return np.maximum(distances, 0.0)


def softmax(values: FloatArray) -> FloatArray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -700.0, 0.0))
    return exponential / np.sum(exponential, axis=1, keepdims=True)


class APRLCore:
    """Internal implementation shared by the classifier and regressor."""

    _parameter_names = (
        "n_prototypes",
        "regularization",
        "local_regularization",
        "local_degree",
        "residual_strength",
        "bandwidth_scale",
        "feature_weight_floor",
        "max_iter",
        "tol",
        "random_state",
        "store_training_data",
    )

    def __init__(
        self,
        *,
        n_prototypes: int | None = None,
        regularization: float = 1.0,
        local_regularization: float = 0.5,
        local_degree: int = 2,
        residual_strength: float = 1.0,
        bandwidth_scale: float = 1.0,
        feature_weight_floor: float = 0.15,
        max_iter: int = 100,
        tol: float = 1e-5,
        random_state: int | None = None,
        store_training_data: bool = True,
    ) -> None:
        self.n_prototypes = n_prototypes
        self.regularization = regularization
        self.local_regularization = local_regularization
        self.local_degree = local_degree
        self.residual_strength = residual_strength
        self.bandwidth_scale = bandwidth_scale
        self.feature_weight_floor = feature_weight_floor
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.store_training_data = store_training_data

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {name: getattr(self, name) for name in self._parameter_names}

    def set_params(self, **params: Any) -> APRLCore:
        unknown = set(params) - set(self._parameter_names)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown parameter(s): {names}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def _validate_hyperparameters(self, n_samples: int) -> int:
        if self.n_prototypes is not None:
            if not isinstance(self.n_prototypes, (int, np.integer)):
                raise TypeError("n_prototypes must be an integer or None")
            if self.n_prototypes < 1:
                raise ValueError("n_prototypes must be at least 1")
            n_prototypes = int(self.n_prototypes)
        else:
            n_prototypes = int(np.clip(round(np.sqrt(n_samples)), 2, 32))
        if self.regularization < 0 or self.local_regularization < 0:
            raise ValueError("regularization values must be non-negative")
        if self.local_degree not in (1, 2):
            raise ValueError("local_degree must be either 1 or 2")
        if not 0.0 <= self.residual_strength <= 2.0:
            raise ValueError("residual_strength must be between 0 and 2")
        if self.bandwidth_scale <= 0:
            raise ValueError("bandwidth_scale must be positive")
        if not 0.0 < self.feature_weight_floor <= 1.0:
            raise ValueError("feature_weight_floor must be in (0, 1]")
        if self.max_iter < 1 or self.tol <= 0:
            raise ValueError("max_iter and tol must be positive")
        return min(n_prototypes, n_samples)

    def _fit_core(
        self,
        X: FloatArray,
        targets: FloatArray,
        raw_relevance: FloatArray,
    ) -> None:
        n_samples, self.n_features_in_ = X.shape
        n_prototypes = self._validate_hyperparameters(n_samples)

        all_nan = np.all(np.isnan(X), axis=0)
        safe_X = X.copy()
        safe_X[:, all_nan] = 0.0
        self.impute_values_ = np.nanmedian(safe_X, axis=0)
        imputed = np.where(np.isnan(X), self.impute_values_, X)
        self.feature_means_ = np.mean(imputed, axis=0)
        self.feature_scales_ = np.std(imputed, axis=0)
        self.feature_scales_[self.feature_scales_ < 1e-12] = 1.0
        standardized = (imputed - self.feature_means_) / self.feature_scales_

        relevance = np.nan_to_num(np.asarray(raw_relevance, dtype=np.float64), nan=0.0)
        relevance = np.maximum(relevance, 0.0)
        peak = float(np.max(relevance)) if relevance.size else 0.0
        if peak > 0.0:
            relevance = relevance / peak
        else:
            relevance = np.ones(self.n_features_in_, dtype=np.float64)
        floor = self.feature_weight_floor
        self.feature_weights_ = floor + (1.0 - floor) * relevance
        distance_X = standardized * np.sqrt(self.feature_weights_)

        self.global_coef_ = solve_ridge(standardized, targets, self.regularization)
        global_prediction = self._linear_prediction(standardized, self.global_coef_)
        residuals = targets - global_prediction
        self.global_residual_variance_ = np.mean(residuals * residuals, axis=0) + 1e-12

        rng = np.random.default_rng(self.random_state)
        self.prototype_centers_ = self._kmeans(distance_X, n_prototypes, rng)
        distances = squared_distances(distance_X, self.prototype_centers_)
        nearest = np.sqrt(np.min(distances, axis=1))
        positive_nearest = nearest[nearest > 1e-12]
        typical_distance = (
            float(np.median(positive_nearest))
            if positive_nearest.size
            else float(np.sqrt(max(self.n_features_in_, 1)))
        )
        self.bandwidth_ = max(typical_distance * self.bandwidth_scale, 1e-6)

        weights, _ = self._prototype_weights_from_distances(distances)
        self.prototype_masses_ = np.sum(weights, axis=0)
        self.expected_prototype_mass_ = n_samples / n_prototypes
        center_standardized = self.prototype_centers_ / np.sqrt(self.feature_weights_)
        n_local_features = self.n_features_in_ * self.local_degree
        self.local_coef_ = np.empty(
            (n_prototypes, n_local_features + 1, targets.shape[1]), dtype=np.float64
        )
        self.local_residual_variance_ = np.empty((n_prototypes, targets.shape[1]), dtype=np.float64)

        for index in range(n_prototypes):
            local_X = self._local_features(standardized, center_standardized[index])
            sample_weight = weights[:, index]
            coefficients = solve_ridge(local_X, residuals, self.local_regularization, sample_weight)
            self.local_coef_[index] = coefficients
            fitted = self._linear_prediction(local_X, coefficients)
            error = residuals - fitted
            mass = max(float(np.sum(sample_weight)), 1e-12)
            variance = np.sum(sample_weight[:, None] * error * error, axis=0) / mass
            self.local_residual_variance_[index] = variance + 1e-12

    def _kmeans(self, X: FloatArray, n_clusters: int, rng: np.random.Generator) -> FloatArray:
        """Run deterministic-seed k-means++ with empty-cluster recovery."""
        n_samples = X.shape[0]
        centers = np.empty((n_clusters, X.shape[1]), dtype=np.float64)
        first = int(rng.integers(n_samples))
        centers[0] = X[first]
        closest = squared_distances(X, centers[:1])[:, 0]

        for index in range(1, n_clusters):
            total = float(np.sum(closest))
            if total <= 1e-15:
                candidate = int(rng.integers(n_samples))
            else:
                candidate = int(rng.choice(n_samples, p=closest / total))
            centers[index] = X[candidate]
            new_distance = squared_distances(X, centers[index : index + 1])[:, 0]
            closest = np.minimum(closest, new_distance)

        previous_inertia = np.inf
        for _ in range(self.max_iter):
            distances = squared_distances(X, centers)
            labels = np.argmin(distances, axis=1)
            inertia = float(np.sum(distances[np.arange(n_samples), labels]))
            updated = centers.copy()
            for index in range(n_clusters):
                members = X[labels == index]
                if members.size:
                    updated[index] = np.mean(members, axis=0)
                else:
                    farthest = int(np.argmax(np.min(distances, axis=1)))
                    updated[index] = X[farthest]
            centers = updated
            if np.isfinite(previous_inertia) and abs(previous_inertia - inertia) <= (
                self.tol * max(previous_inertia, 1.0)
            ):
                break
            previous_inertia = inertia
        return centers

    @staticmethod
    def _linear_prediction(X: FloatArray, coefficients: FloatArray) -> FloatArray:
        return coefficients[0] + X @ coefficients[1:]

    def _local_features(self, X: FloatArray, center: FloatArray) -> FloatArray:
        difference = X - center
        if self.local_degree == 1:
            return difference
        return np.concatenate((difference, difference * difference), axis=1)

    def _transform_X(self, X: ArrayLike) -> tuple[FloatArray, FloatArray]:
        self._check_is_fitted()
        array = as_2d_float(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {array.shape[1]} features, expected {self.n_features_in_}")
        imputed = np.where(np.isnan(array), self.impute_values_, array)
        standardized = (imputed - self.feature_means_) / self.feature_scales_
        distance_X = standardized * np.sqrt(self.feature_weights_)
        return standardized, distance_X

    def _prototype_weights_from_distances(
        self, distances: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        logits = -distances / (2.0 * self.bandwidth_ * self.bandwidth_)
        weights = np.exp(np.clip(logits, -700.0, 0.0))
        totals = np.sum(weights, axis=1, keepdims=True)
        normalized = weights / np.maximum(totals, 1e-300)
        return weights, normalized

    def _components(self, X: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        standardized, distance_X = self._transform_X(X)
        distances = squared_distances(distance_X, self.prototype_centers_)
        _, normalized = self._prototype_weights_from_distances(distances)
        global_prediction = self._linear_prediction(standardized, self.global_coef_)
        center_standardized = self.prototype_centers_ / np.sqrt(self.feature_weights_)

        local_by_prototype = np.empty(
            (standardized.shape[0], self.prototype_centers_.shape[0], self.global_coef_.shape[1]),
            dtype=np.float64,
        )
        for index, coefficients in enumerate(self.local_coef_):
            local_X = self._local_features(standardized, center_standardized[index])
            local_by_prototype[:, index, :] = self._linear_prediction(local_X, coefficients)
        local_residual = np.einsum("nk,nko->no", normalized, local_by_prototype)

        # Use a flat in-distribution gate and a steep tail. An exponential gate
        # suppressed useful local corrections even for ordinary training-region
        # samples; the quartic gate stays near one locally but still retreats to
        # the global model for genuinely distant inputs.
        nearest_ratio = np.sqrt(np.min(distances, axis=1)) / self.bandwidth_
        coverage = 1.0 / (1.0 + (nearest_ratio / 2.5) ** 4)
        mass_ratio = np.minimum(
            self.prototype_masses_ / max(self.expected_prototype_mass_, 1e-12), 1.0
        )
        density = normalized @ mass_ratio
        density_gate = 0.5 + 0.5 * np.sqrt(np.clip(density, 0.0, 1.0))
        confidence = np.clip(coverage * density_gate, 0.0, 1.0)
        return global_prediction, local_residual, confidence, normalized

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "global_coef_"):
            raise RuntimeError("This APRL estimator has not been fitted")

    def _clear_stored_data(self) -> None:
        for name in ("_stored_X", "_stored_y"):
            if hasattr(self, name):
                delattr(self, name)
