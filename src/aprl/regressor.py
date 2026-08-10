"""Regression estimator for Adaptive Prototype Residual Learning."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._core import APRLCore, FloatArray, as_2d_float


class APRLRegressor(APRLCore):
    """Adaptive Prototype Residual Learner for numeric regression.

    The global ridge prediction handles broad trends. Relevance-weighted
    prototypes define local regions, and a ridge expert at each prototype learns
    the residual left by the global model. Distance and local sample mass decide
    how strongly the residual correction should be trusted.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> APRLRegressor:
        self._clear_stored_data()
        X_array = as_2d_float(X)
        y_array = np.asarray(y, dtype=np.float64)
        self._single_output = y_array.ndim == 1
        if self._single_output:
            y_array = y_array[:, None]
        if y_array.ndim != 2 or y_array.shape[0] != X_array.shape[0]:
            raise ValueError("y must have one row per sample in X")
        if y_array.shape[1] == 0 or not np.isfinite(y_array).all():
            raise ValueError("y must contain finite numeric targets")

        imputed_for_relevance = self._temporary_impute(X_array)
        relevance = self._regression_relevance(imputed_for_relevance, y_array)
        self._fit_core(X_array, y_array, relevance)
        self.n_outputs_ = y_array.shape[1]
        if self.store_training_data:
            self._stored_X = X_array.copy()
            self._stored_y = np.asarray(y).copy()
        return self

    @staticmethod
    def _temporary_impute(X: FloatArray) -> FloatArray:
        safe = X.copy()
        all_nan = np.all(np.isnan(safe), axis=0)
        safe[:, all_nan] = 0.0
        medians = np.nanmedian(safe, axis=0)
        return np.where(np.isnan(safe), medians, safe)

    @staticmethod
    def _regression_relevance(X: FloatArray, y: FloatArray) -> FloatArray:
        centered_X = X - np.mean(X, axis=0)
        centered_y = y - np.mean(y, axis=0)
        numerator = np.abs(centered_X.T @ centered_y)
        x_norm = np.sqrt(np.sum(centered_X * centered_X, axis=0))[:, None]
        y_norm = np.sqrt(np.sum(centered_y * centered_y, axis=0))[None, :]
        correlations = numerator / np.maximum(x_norm * y_norm, 1e-12)
        return np.mean(correlations, axis=1)

    def predict(self, X: ArrayLike) -> np.ndarray:
        prediction, _ = self.predict_with_uncertainty(X)
        return prediction

    def predict_with_uncertainty(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return predictions and distance-aware one-standard-deviation estimates."""
        global_prediction, local_residual, confidence, normalized = self._components(X)
        correction = self.residual_strength * confidence[:, None] * local_residual
        prediction = global_prediction + correction

        local_variance = normalized @ self.local_residual_variance_
        confidence_2d = confidence[:, None]
        variance = (
            confidence_2d * local_variance
            + (1.0 - confidence_2d) * self.global_residual_variance_[None, :]
            + 2.0 * (1.0 - confidence_2d) * self.global_residual_variance_[None, :]
            + 0.05 * correction * correction
        )
        uncertainty = np.sqrt(np.maximum(variance, 1e-12))
        if self._single_output:
            return prediction[:, 0], uncertainty[:, 0]
        return prediction, uncertainty

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination, averaged across outputs."""
        truth = np.asarray(y, dtype=np.float64)
        predicted = np.asarray(self.predict(X), dtype=np.float64)
        if truth.shape != predicted.shape:
            raise ValueError("y has a different shape from the model predictions")
        if truth.ndim == 1:
            truth = truth[:, None]
            predicted = predicted[:, None]
        residual = np.sum((truth - predicted) ** 2, axis=0)
        total = np.sum((truth - np.mean(truth, axis=0)) ** 2, axis=0)
        values = 1.0 - residual / np.maximum(total, 1e-12)
        return float(np.mean(values))

    def partial_fit(self, X: ArrayLike, y: ArrayLike) -> APRLRegressor:
        """Add a batch and exactly refit APRL on all batches observed so far."""
        if not self.store_training_data:
            raise RuntimeError("partial_fit requires store_training_data=True")
        X_array = as_2d_float(X)
        y_array = np.asarray(y)
        if hasattr(self, "_stored_X"):
            X_array = np.concatenate((self._stored_X, X_array), axis=0)
            y_array = np.concatenate((self._stored_y, y_array), axis=0)
        return self.fit(X_array, y_array)
