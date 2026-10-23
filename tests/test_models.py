"""Baseline definitions, finite predictions and an exact linear relationship."""

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.data import FEATURE_COLUMNS
from airfoil_noise.models import BASELINE_MODELS, build_model


@pytest.mark.parametrize("name", BASELINE_MODELS)
def test_model_shape_finiteness_settings_and_repeatability(name: str) -> None:
    x = pd.DataFrame(np.random.default_rng(73).uniform(1, 10, (50, 5)), columns=FEATURE_COLUMNS)
    y = 4 + x.to_numpy() @ np.array([2, -3, 1, 0.2, 6])
    first, second = build_model(name), build_model(name)
    assert first is not second
    assert first.get_params(deep=True) == second.get_params(deep=True)
    first.fit(x.iloc[:35], y[:35])
    second.fit(x.iloc[:35], y[:35])
    prediction = first.predict(x.iloc[35:])
    assert prediction.shape == (15,)
    assert np.isfinite(prediction).all()
    np.testing.assert_allclose(prediction, second.predict(x.iloc[35:]), rtol=1e-12, atol=1e-12)
    assert first.feature_names_in_.tolist() == list(FEATURE_COLUMNS)
    if name == "linear":
        np.testing.assert_allclose(prediction, y[35:], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(first.coef_, [2, -3, 1, 0.2, 6], atol=1e-11)
    else:
        np.testing.assert_allclose(prediction, np.mean(y[:35]))


@pytest.mark.parametrize("name", ["tree", "linear_log_frequency", "", None, 2])
def test_unknown_model_rejected(name) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(name)


def test_fixed_settings_are_declared() -> None:
    assert build_model("training_mean").get_params()["strategy"] == "mean"
    parameters = build_model("linear").get_params()
    assert parameters["fit_intercept"] is True
    assert parameters["positive"] is False
    assert parameters["n_jobs"] == 1
