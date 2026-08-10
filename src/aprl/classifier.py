"""Classification estimator for Adaptive Prototype Residual Learning."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._core import APRLCore, FloatArray, as_2d_float, softmax


class APRLClassifier(APRLCore):
    """Adaptive Prototype Residual Learner for classification."""

    def fit(self, X: ArrayLike, y: ArrayLike) -> APRLClassifier:
        self._clear_stored_data()
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError("y must be a one-dimensional array of class labels")
        try:
            classes = np.unique(y_array)
        except TypeError as error:
            raise ValueError("class labels must be mutually comparable") from error
        return self._fit_with_classes(X, y_array, classes)

    def _fit_with_classes(self, X: ArrayLike, y: np.ndarray, classes: np.ndarray) -> APRLClassifier:
        self._clear_stored_data()
        X_array = as_2d_float(X)
        if y.ndim != 1 or y.shape[0] != X_array.shape[0]:
            raise ValueError("y must have one class label per sample in X")
        if classes.size < 2:
            raise ValueError("classification requires at least two classes")
        self.classes_ = np.asarray(classes)
        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        try:
            encoded = np.array([class_to_index[value] for value in y.tolist()], dtype=int)
        except (KeyError, TypeError) as error:
            raise ValueError("y contains a label that is absent from classes") from error

        targets = np.eye(self.classes_.size, dtype=np.float64)[encoded]
        imputed_for_relevance = self._temporary_impute(X_array)
        relevance = self._classification_relevance(
            imputed_for_relevance, encoded, self.classes_.size
        )

        # The core learns local errors relative to ridge outputs. Refit those
        # residuals against probabilities so the correction has a probabilistic
        # interpretation rather than a raw-logit interpretation.
        self._fit_core(X_array, targets, relevance)
        standardized, _ = self._transform_X(X_array)
        global_probability = softmax(self._linear_prediction(standardized, self.global_coef_))
        residuals = targets - global_probability
        self._refit_local_residuals(standardized, residuals)

        if self.store_training_data:
            self._stored_X = X_array.copy()
            self._stored_y = y.copy()
        return self

    @staticmethod
    def _temporary_impute(X: FloatArray) -> FloatArray:
        safe = X.copy()
        all_nan = np.all(np.isnan(safe), axis=0)
        safe[:, all_nan] = 0.0
        medians = np.nanmedian(safe, axis=0)
        return np.where(np.isnan(safe), medians, safe)

    @staticmethod
    def _classification_relevance(X: FloatArray, encoded: np.ndarray, n_classes: int) -> FloatArray:
        overall = np.mean(X, axis=0)
        between = np.zeros(X.shape[1], dtype=np.float64)
        within = np.zeros(X.shape[1], dtype=np.float64)
        for index in range(n_classes):
            group = X[encoded == index]
            if not group.size:
                continue
            group_mean = np.mean(group, axis=0)
            between += group.shape[0] * (group_mean - overall) ** 2
            within += np.sum((group - group_mean) ** 2, axis=0)
        return between / np.maximum(within, 1e-12)

    def _refit_local_residuals(self, standardized: FloatArray, residuals: FloatArray) -> None:
        from ._core import solve_ridge, squared_distances

        distance_X = standardized * np.sqrt(self.feature_weights_)
        distances = squared_distances(distance_X, self.prototype_centers_)
        weights, _ = self._prototype_weights_from_distances(distances)
        center_standardized = self.prototype_centers_ / np.sqrt(self.feature_weights_)
        for index in range(self.prototype_centers_.shape[0]):
            local_X = self._local_features(standardized, center_standardized[index])
            sample_weight = weights[:, index]
            coefficients = solve_ridge(local_X, residuals, self.local_regularization, sample_weight)
            self.local_coef_[index] = coefficients
            fitted = self._linear_prediction(local_X, coefficients)
            error = residuals - fitted
            mass = max(float(np.sum(sample_weight)), 1e-12)
            self.local_residual_variance_[index] = (
                np.sum(sample_weight[:, None] * error * error, axis=0) / mass + 1e-12
            )

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        probabilities, _ = self.predict_proba_with_uncertainty(X)
        return probabilities

    def predict_proba_with_uncertainty(self, X: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return class probabilities and uncertainty values in the interval [0, 1]."""
        global_logits, local_residual, confidence, _ = self._components(X)
        global_probability = softmax(global_logits)
        probability = global_probability + (
            self.residual_strength * confidence[:, None] * local_residual
        )
        probability = np.maximum(probability, 1e-12)
        probability /= np.sum(probability, axis=1, keepdims=True)

        entropy = -np.sum(probability * np.log(probability), axis=1)
        entropy /= np.log(self.classes_.size)
        uncertainty = 1.0 - confidence * (1.0 - entropy)
        return probability, np.clip(uncertainty, 0.0, 1.0)

    def predict(self, X: ArrayLike) -> np.ndarray:
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        truth = np.asarray(y)
        prediction = self.predict(X)
        if truth.shape != prediction.shape:
            raise ValueError("y has a different shape from the model predictions")
        return float(np.mean(truth == prediction))

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        classes: ArrayLike | None = None,
    ) -> APRLClassifier:
        """Add a batch and exactly refit APRL on all batches observed so far."""
        if not self.store_training_data:
            raise RuntimeError("partial_fit requires store_training_data=True")
        X_array = as_2d_float(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if hasattr(self, "_stored_X"):
            known_classes = self.classes_
            if classes is not None and not np.array_equal(np.asarray(classes), known_classes):
                raise ValueError("classes must match the classes supplied on the first call")
            X_array = np.concatenate((self._stored_X, X_array), axis=0)
            y_array = np.concatenate((self._stored_y, y_array), axis=0)
        else:
            known_classes = np.unique(y_array) if classes is None else np.asarray(classes)
        return self._fit_with_classes(X_array, y_array, known_classes)
