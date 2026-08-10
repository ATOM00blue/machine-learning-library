import numpy as np
import pytest

from aprl import APRLClassifier


def nonlinear_classification(seed: int = 17, n_samples: int = 500):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 5))
    margin = X[:, 0] * X[:, 1] + 0.7 * X[:, 2] - 0.2 * X[:, 3] ** 2
    y = np.where(margin > 0.0, "positive", "negative")
    return X, y


def test_classifier_learns_nonlinear_boundary_and_probabilities_are_valid():
    X, y = nonlinear_classification()
    model = APRLClassifier(n_prototypes=22, random_state=2).fit(X[:380], y[:380])

    probability, uncertainty = model.predict_proba_with_uncertainty(X[380:])

    assert model.score(X[380:], y[380:]) > 0.78
    assert probability.shape == (120, 2)
    np.testing.assert_allclose(np.sum(probability, axis=1), 1.0)
    assert np.all(probability >= 0.0)
    assert np.all((0.0 <= uncertainty) & (uncertainty <= 1.0))


def test_multiclass_and_missing_values():
    rng = np.random.default_rng(8)
    centers = np.array([[-2.0, -1.0], [2.0, -1.0], [0.0, 2.0]])
    X = np.vstack([center + rng.normal(scale=0.45, size=(70, 2)) for center in centers])
    y = np.repeat(["red", "green", "blue"], 70)
    X[::17, 0] = np.nan

    model = APRLClassifier(random_state=4).fit(X, y)

    assert model.score(X, y) > 0.95
    assert model.predict_proba(X[:10]).shape == (10, 3)


def test_partial_fit_can_receive_classes_before_every_class_is_seen():
    X, y = nonlinear_classification(n_samples=260)
    negative = np.flatnonzero(y == "negative")[:70]
    remaining = np.setdiff1d(np.arange(y.size), negative)[:140]
    model = APRLClassifier(random_state=9)

    model.partial_fit(X[negative], y[negative], classes=np.array(["negative", "positive"]))
    model.partial_fit(X[remaining], y[remaining])

    assert set(model.classes_) == {"negative", "positive"}
    assert model._stored_X.shape[0] == 210
    assert model.score(X, y) > 0.70


def test_classifier_uncertainty_rises_far_from_training_data():
    X, y = nonlinear_classification(n_samples=300)
    model = APRLClassifier(random_state=1).fit(X, y)

    _, in_distribution = model.predict_proba_with_uncertainty(X[:60])
    _, out_of_distribution = model.predict_proba_with_uncertainty(X[:60] + 20.0)

    assert np.mean(out_of_distribution) > np.mean(in_distribution)


def test_classifier_validation():
    with pytest.raises(ValueError, match="at least two classes"):
        APRLClassifier().fit([[0.0], [1.0]], ["one", "one"])
    with pytest.raises(ValueError, match="one-dimensional"):
        APRLClassifier().fit([[0.0], [1.0]], [[0], [1]])
    with pytest.raises(ValueError, match="absent"):
        APRLClassifier().partial_fit([[0.0], [1.0]], [0, 2], classes=np.array([0, 1]))
