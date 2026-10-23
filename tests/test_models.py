"""Five model constructors, study controls and complete JSON reconstruction."""

from copy import deepcopy
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline

from airfoil_noise.data import FEATURE_COLUMNS
from airfoil_noise.models import (
    BASELINE_MODELS, DEFAULT_MODEL_SEED, MODEL_NAMES, build_model,
    model_configuration, model_from_configuration,
)


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


@pytest.mark.parametrize("name", ["forest", "linear_log_frequency", "", None, 2])
def test_unknown_model_rejected(name) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(name)


def test_fixed_settings_are_declared() -> None:
    assert build_model("training_mean").get_params()["strategy"] == "mean"
    parameters = build_model("linear").get_params()
    assert parameters["fit_intercept"] is True
    assert parameters["positive"] is False
    assert parameters["n_jobs"] == 1


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_all_models_round_trip_complete_configuration_and_predictions(name) -> None:
    x = pd.DataFrame(np.random.default_rng(73).uniform(1, 10, (40, 5)), columns=FEATURE_COLUMNS)
    y = 3 + .2 * x.iloc[:, 0].to_numpy() - .1 * x.iloc[:, 1].to_numpy()
    parameters = {"hidden_layer_sizes": (6, 3), "max_iter": 35} if name == "mlp" else None
    original = build_model(name, parameters=parameters)
    record = model_configuration(original)
    decoded = json.loads(json.dumps(record, allow_nan=False))
    rebuilt = model_from_configuration(decoded)
    assert rebuilt is not original
    assert model_configuration(rebuilt) == record
    if isinstance(original, Pipeline):
        assert original.named_steps["scaler"] is not rebuilt.named_steps["scaler"]
        assert original.named_steps["regressor"] is not rebuilt.named_steps["regressor"]
        steps = decoded["parameters"]["steps"]
        assert steps[0]["items"][0] == "scaler"
        assert steps[0]["items"][1]["estimator_class"].endswith(".StandardScaler")
        assert isinstance(steps[1]["items"][1]["parameters"], dict)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        original.fit(x.iloc[:30], y[:30])
        rebuilt.fit(x.iloc[:30], y[:30])
    assert all(issubclass(warning.category, ConvergenceWarning) for warning in caught)
    before = original.predict(x.iloc[30:])
    after = rebuilt.predict(x.iloc[30:])
    assert before.shape == (10,) and np.isfinite(before).all()
    np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)
    assert original.feature_names_in_.tolist() == list(FEATURE_COLUMNS)
    assert model_configuration(original) == record  # No fitted state in constructor records.


def test_fixed_pipeline_policies_and_independent_steps() -> None:
    tree = build_model("tree")
    assert tree.max_depth == 5 and tree.min_samples_leaf == 5
    assert tree.random_state == DEFAULT_MODEL_SEED
    for name in ("svr", "mlp"):
        first, second = build_model(name), build_model(name)
        assert first.named_steps["scaler"] is not second.named_steps["scaler"]
        assert first.named_steps["regressor"] is not second.named_steps["regressor"]
        assert first.named_steps["scaler"].with_mean is True
        assert first.named_steps["scaler"].with_std is True
    svr = build_model("svr").named_steps["regressor"]
    assert (svr.kernel, svr.C, svr.epsilon, svr.gamma) == ("rbf", 10, .1, "scale")
    mlp = build_model("mlp").named_steps["regressor"]
    assert mlp.hidden_layer_sizes == (20,)
    assert mlp.solver == "adam" and mlp.max_iter == 2000
    assert mlp.early_stopping is False and mlp.warm_start is False
    assert mlp.random_state == DEFAULT_MODEL_SEED


@pytest.mark.parametrize("name", ["tree", "mlp"])
def test_explicit_seeds_and_parameter_overrides(name) -> None:
    parameters = {"max_depth": 3} if name == "tree" else {"hidden_layer_sizes": [7, 4]}
    model = build_model(name, parameters=parameters, seed=np.int64(29))
    regressor = model.named_steps["regressor"] if name == "mlp" else model
    assert regressor.random_state == 29
    for key, value in parameters.items():
        assert regressor.get_params()[key] == value
    if name == "mlp":
        parameters["hidden_layer_sizes"][0] = 50
        assert regressor.hidden_layer_sizes == [7, 4]  # No shared caller state.


@pytest.mark.parametrize("layers", [(6, 3), [6, 3], 6])
def test_hidden_layer_parameter_types_survive_json(layers) -> None:
    model = build_model("mlp", parameters={"hidden_layer_sizes": layers})
    record = json.loads(json.dumps(model_configuration(model), allow_nan=False))
    restored = model_from_configuration(record).named_steps["regressor"].hidden_layer_sizes
    assert type(restored) is type(layers)
    assert restored == layers


@pytest.mark.parametrize("seed", [-1, 2 ** 32, True, 1.5, "17"])
@pytest.mark.parametrize("name", ["tree", "mlp"])
def test_invalid_seeds_rejected(name, seed) -> None:
    with pytest.raises(ValueError):
        build_model(name, seed=seed)


@pytest.mark.parametrize("name", BASELINE_MODELS + ("svr",))
def test_deterministic_models_reject_unused_seed(name) -> None:
    with pytest.raises(ValueError, match="deterministic"):
        build_model(name, seed=17)


@pytest.mark.parametrize("name,parameters", [
    ("tree", {"max_depth": 0}), ("tree", {"min_samples_leaf": 0}),
    ("tree", {"min_samples_split": 1}), ("tree", {"random_state": 17}),
    ("svr", {"C": 0}), ("svr", {"C": "10"}), ("svr", {"epsilon": -1}),
    ("svr", {"gamma": 0}), ("svr", {"kernel": "linear"}),
    ("svr", {"scaler__with_mean": False}), ("linear", {"unknown": 3}),
    ("training_mean", {"strategy": "median"}),
    ("mlp", {"early_stopping": True}), ("mlp", {"warm_start": True}),
    ("mlp", {"solver": "lbfgs"}), ("mlp", {"alpha": -1}),
    ("mlp", {"max_iter": 0}), ("mlp", {"learning_rate_init": 0}),
    ("mlp", {"hidden_layer_sizes": ()}), ("mlp", {"hidden_layer_sizes": (65,)}),
    ("mlp", {"hidden_layer_sizes": (4, 4, 4)}), ("mlp", {"hidden_layer_sizes": (True,)}),
    ("mlp", {"activation": "unknown"}),
])
def test_invalid_settings_and_study_policy_rejected(name, parameters) -> None:
    with pytest.raises(ValueError):
        build_model(name, parameters=parameters)


@pytest.mark.parametrize("parameters", [[("C", 1)], {1: 2}, {"C": np.inf}])
def test_invalid_parameter_containers_and_values(parameters) -> None:
    with pytest.raises(ValueError):
        build_model("svr", parameters=parameters)


def test_opaque_configuration_values_are_rejected() -> None:
    model = build_model("svr")
    model.memory = object()
    with pytest.raises(TypeError, match="Unsupported configuration"):
        model_configuration(model)


@pytest.mark.parametrize("fault", ["unknown_class", "version", "missing_parameter", "extra_parameter", "typed_value", "nonfinite", "early_stopping", "seed", "extra_field"])
def test_corrupt_configuration_records_are_rejected(fault) -> None:
    record = deepcopy(model_configuration(build_model("mlp")))
    nested = record["parameters"]["steps"][1]["items"][1]
    if fault == "unknown_class":
        nested["estimator_class"] = "unknown.Estimator"
    elif fault == "version":
        record["configuration_schema_version"] = 9
    elif fault == "missing_parameter":
        del nested["parameters"]["alpha"]
    elif fault == "extra_parameter":
        nested["parameters"]["unknown"] = 1
    elif fault == "typed_value":
        nested["parameters"]["hidden_layer_sizes"]["__type__"] = "opaque"
    elif fault == "nonfinite":
        nested["parameters"]["alpha"] = float("nan")
    elif fault == "early_stopping":
        nested["parameters"]["early_stopping"] = True
    elif fault == "seed":
        nested["parameters"]["random_state"] = None
    else:
        record["unexpected"] = True
    with pytest.raises(ValueError):
        model_from_configuration(record)


def test_original_baseline_configuration_records_remain_readable() -> None:
    path = Path(__file__).resolve().parents[1] / "reports/development/milestone-2-baselines/run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    assert run["schema_version"] == 1
    for name, record in run["models"].items():
        restored = model_from_configuration(record)
        assert model_configuration(restored) == model_configuration(build_model(name))
