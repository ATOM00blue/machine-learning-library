#!/usr/bin/env python3
"""Train APRL on a nonlinear dataset and inspect its uncertainty."""

from __future__ import annotations

import numpy as np

from aprl import APRLClassifier, APRLRegressor


def regression_example() -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(500, 5))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 - 0.4 * X[:, 2]

    model = APRLRegressor(random_state=12).fit(X[:400], y[:400])
    predictions, uncertainty = model.predict_with_uncertainty(X[400:])

    print(f"Regression R^2: {model.score(X[400:], y[400:]):.3f}")
    for prediction, standard_deviation in zip(predictions[:3], uncertainty[:3]):
        print(f"  prediction={prediction: .3f}  uncertainty={standard_deviation:.3f}")


def classification_example() -> None:
    rng = np.random.default_rng(21)
    X = rng.normal(size=(600, 4))
    y = np.where(X[:, 0] * X[:, 1] + X[:, 2] > 0, "accept", "reject")

    model = APRLClassifier(random_state=21).fit(X[:480], y[:480])
    probabilities, uncertainty = model.predict_proba_with_uncertainty(X[480:])

    print(f"Classification accuracy: {model.score(X[480:], y[480:]):.3f}")
    for probability, confidence_risk in zip(probabilities[:3], uncertainty[:3]):
        print(f"  probabilities={probability.round(3)}  uncertainty={confidence_risk:.3f}")


if __name__ == "__main__":
    regression_example()
    classification_example()
