import pickle

import numpy as np
import pytest

from aprl import APRLRegressor


def nonlinear_regression(seed: int = 7, n_samples: int = 420):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 5))
    y = (
        np.sin(1.5 * X[:, 0])
        + 0.7 * X[:, 1] ** 2
        - 0.5 * X[:, 2]
        + 0.25 * X[:, 0] * X[:, 3]
        + rng.normal(scale=0.04, size=n_samples)
    )
    return X, y


def test_regressor_learns_nonlinear_signal_and_reports_uncertainty():
    X, y = nonlinear_regression()
    model = APRLRegressor(n_prototypes=20, random_state=11).fit(X[:320], y[:320])

    prediction, uncertainty = model.predict_with_uncertainty(X[320:])

    assert prediction.shape == (100,)
    assert uncertainty.shape == (100,)
    assert np.all(uncertainty > 0)
    assert model.score(X[320:], y[320:]) > 0.80


def test_out_of_distribution_points_are_more_uncertain():
    X, y = nonlinear_regression()
    model = APRLRegressor(random_state=3).fit(X, y)

    _, in_distribution = model.predict_with_uncertainty(X[:80])
    _, out_of_distribution = model.predict_with_uncertainty(X[:80] + 20.0)

    assert np.mean(out_of_distribution) > 1.5 * np.mean(in_distribution)


def test_multioutput_missing_values_and_constant_feature():
    X, y = nonlinear_regression(n_samples=180)
    X[:, 4] = 2.0
    X[::9, 1] = np.nan
    targets = np.column_stack((y, 2.0 * y + 1.0))

    model = APRLRegressor(random_state=5).fit(X, targets)
    prediction, uncertainty = model.predict_with_uncertainty(X[:12])

    assert prediction.shape == (12, 2)
    assert uncertainty.shape == (12, 2)
    assert np.isfinite(prediction).all()
    assert np.isfinite(uncertainty).all()


def test_partial_fit_adds_new_batches():
    X, y = nonlinear_regression(n_samples=240)
    model = APRLRegressor(random_state=13)
    model.partial_fit(X[:120], y[:120])
    first_score = model.score(X[120:], y[120:])
    model.partial_fit(X[120:200], y[120:200])

    assert model._stored_X.shape[0] == 200
    assert model.score(X[200:], y[200:]) >= first_score - 0.15


def test_reproducible_pickleable_and_parameter_compatible():
    X, y = nonlinear_regression(n_samples=160)
    first = APRLRegressor(random_state=19).fit(X, y)
    second = APRLRegressor(random_state=19).fit(X, y)
    restored = pickle.loads(pickle.dumps(first))

    np.testing.assert_allclose(first.predict(X[:20]), second.predict(X[:20]))
    np.testing.assert_allclose(first.predict(X[:20]), restored.predict(X[:20]))
    assert first.get_params()["random_state"] == 19
    assert first.set_params(residual_strength=0.8) is first


def test_invalid_inputs_fail_clearly():
    model = APRLRegressor()
    with pytest.raises(ValueError, match="2D"):
        model.fit([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        model.fit([[1.0], [2.0]], [1.0, np.nan])
    with pytest.raises(RuntimeError, match="not been fitted"):
        model.predict([[1.0]])
    with pytest.raises(ValueError, match="local_degree"):
        APRLRegressor(local_degree=3).fit([[1.0], [2.0]], [1.0, 2.0])
