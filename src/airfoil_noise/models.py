"""Build five regressors and preserve their complete construction settings."""

from copy import deepcopy
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


BASELINE_MODELS = ("training_mean", "linear")
MODEL_NAMES = BASELINE_MODELS + ("tree", "svr", "mlp")
DEFAULT_MODEL_SEED = 17
"""Construction seed chosen before new model comparisons, separate from split seeds."""

_CLASSES = (
    DummyRegressor, LinearRegression, DecisionTreeRegressor, SVR, MLPRegressor,
    StandardScaler, Pipeline,
)
_CLASS_REGISTRY = {f"{cls.__module__}.{cls.__qualname__}": cls for cls in _CLASSES}


def _positive_integer(value, name: str, minimum: int = 1) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _real(value, name: str, *, allow_zero: bool = False) -> None:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")


def _check_policy(estimator: BaseEstimator) -> None:
    """Check study controls and common tuning ranges before any fit."""
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _check_policy(step)
    if isinstance(estimator, (DecisionTreeRegressor, MLPRegressor)):
        _positive_integer(estimator.random_state, "random_state", minimum=0)
        if estimator.random_state >= 2 ** 32:
            raise ValueError("random_state must be smaller than 2**32.")
    if isinstance(estimator, DummyRegressor) and estimator.strategy != "mean":
        raise ValueError("The training-mean method requires strategy='mean'.")
    if isinstance(estimator, DecisionTreeRegressor):
        if estimator.max_depth is not None:
            _positive_integer(estimator.max_depth, "max_depth")
        _positive_integer(estimator.min_samples_leaf, "min_samples_leaf")
        _positive_integer(estimator.min_samples_split, "min_samples_split", minimum=2)
    if isinstance(estimator, SVR):
        if estimator.kernel != "rbf":
            raise ValueError("This study supports only the RBF SVR kernel.")
        _real(estimator.C, "C")
        _real(estimator.epsilon, "epsilon", allow_zero=True)
        _real(estimator.tol, "tol")
        if not (isinstance(estimator.gamma, str) and estimator.gamma in ("scale", "auto")):
            _real(estimator.gamma, "gamma")
    if isinstance(estimator, MLPRegressor):
        if estimator.early_stopping is not False:
            raise ValueError("Automatic MLP early stopping must remain disabled for grouped validation.")
        if estimator.solver != "adam" or estimator.warm_start is not False:
            raise ValueError("The small MLP uses solver='adam' and warm_start=False.")
        layers = estimator.hidden_layer_sizes
        if isinstance(layers, Integral):
            layers = (layers,)
        if not isinstance(layers, (tuple, list)) or not 1 <= len(layers) <= 2:
            raise ValueError("Use one or two small hidden layers.")
        for width in layers:
            _positive_integer(width, "hidden layer width")
            if width > 64:
                raise ValueError("Small MLP hidden layers are limited to 64 neurons each.")
        for name in ("max_iter", "n_iter_no_change"):
            _positive_integer(getattr(estimator, name), name)
        _real(estimator.alpha, "alpha", allow_zero=True)
        _real(estimator.tol, "tol", allow_zero=True)
        _real(estimator.learning_rate_init, "learning_rate_init")
        if estimator.activation not in ("identity", "logistic", "tanh", "relu"):
            raise ValueError("Unsupported MLP activation.")
        if estimator.batch_size != "auto":
            _positive_integer(estimator.batch_size, "batch_size")


def build_model(
    name: str, *, parameters: dict | None = None, seed: int | None = None,
) -> BaseEstimator:
    """Return a new estimator with explicit overrides and reproducible seeds.

    Parameters
    ----------
    name : {'training_mean', 'linear', 'tree', 'svr', 'mlp'}
        Mean and linear defaults are unchanged from Milestone 2. Tree uses raw
        features. SVR and MLP use StandardScaler followed by the regressor.
    parameters : dict, optional
        Overrides for the regressor's sklearn parameter names, without pipeline
        prefixes. Preprocessing stays fixed. Set random_state through seed.
        Common tuning ranges and study controls are checked here; sklearn
        validates remaining solver-specific constraints during fit.
    seed : int, optional
        Tree/MLP random_state, default 17. Deterministic methods require None.
        This is independent of the frozen holdout seed. No seed is selected
        using model performance.

    Returns
    -------
    sklearn.base.BaseEstimator
        Independent, unfitted estimator or pipeline. Defaults are construction
        settings, not tuned choices. Targets stay in dB without scaling.

    Raises
    ------
    ValueError
        Unknown method/parameters, invalid seed, common invalid tuning values,
        non-RBF SVR, oversized MLP or prohibited early stopping/warm starts.
    TypeError
        A parameter cannot be represented as supported structured JSON.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")
    if parameters is not None and (not isinstance(parameters, dict)
                                   or not all(isinstance(key, str) for key in parameters)):
        raise ValueError("parameters must be a dictionary with string keys.")
    overrides = {} if parameters is None else parameters
    if "random_state" in overrides:
        raise ValueError("Supply random_state through the explicit seed argument.")
    _encode_value(overrides)  # Reject nonfinite values and opaque objects early.
    if name in ("tree", "mlp"):
        chosen_seed = DEFAULT_MODEL_SEED if seed is None else seed
        _positive_integer(chosen_seed, "seed", minimum=0)
        if chosen_seed >= 2 ** 32:
            raise ValueError("seed must be smaller than 2**32.")
        chosen_seed = int(chosen_seed)
    elif seed is not None:
        raise ValueError(f"{name} is deterministic and does not use a model seed.")
    if name == "training_mean":
        regressor = DummyRegressor(strategy="mean", constant=None, quantile=None)
    elif name == "linear":
        regressor = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=1, positive=False)
    elif name == "tree":
        regressor = DecisionTreeRegressor(
            max_depth=5, min_samples_leaf=5, random_state=chosen_seed,
        )
    elif name == "svr":
        regressor = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    else:
        regressor = MLPRegressor(
            hidden_layer_sizes=(20,), activation="relu", solver="adam",
            alpha=0.0001, learning_rate_init=0.001, max_iter=2000,
            tol=0.0001, n_iter_no_change=20, random_state=chosen_seed,
            early_stopping=False, warm_start=False,
        )
    regressor.set_params(**deepcopy(overrides))
    _check_policy(regressor)
    if name in ("svr", "mlp"):
        return Pipeline([("scaler", StandardScaler()), ("regressor", regressor)])
    return regressor


def _encode_value(value):
    if isinstance(value, BaseEstimator):
        return model_configuration(value)
    if isinstance(value, np.generic):
        return _encode_value(value.item())
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value) or "__type__" in value:
            raise ValueError("Configuration dictionaries need string keys; __type__ is reserved.")
        return {key: _encode_value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Configuration values must be finite.")
        return value
    raise TypeError(f"Unsupported configuration value type: {type(value).__name__}.")


def model_configuration(estimator: BaseEstimator) -> dict:
    """Encode complete constructor settings as strict, structured JSON values.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Unfitted or fitted estimator. Only constructor parameters are recorded;
        learned weights, scaler statistics and other fitted state are excluded.

    Returns
    -------
    dict
        Configuration schema version, qualified estimator_class and every
        shallow parameter, recursively encoding nested estimators and pipeline
        steps. Tuples use an explicit type tag so they survive JSON round trips.
        No estimator is reduced to a repr string.

    Raises
    ------
    TypeError
        An unsupported object occurs; it is never silently stringified.
    ValueError
        Nonfinite values or unsupported dictionary keys occur.
    """
    if not isinstance(estimator, BaseEstimator):
        raise TypeError("Expected a sklearn estimator.")
    return {
        "configuration_schema_version": 1,
        "estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "parameters": {
            key: _encode_value(value) for key, value in estimator.get_params(deep=False).items()
        },
    }


def _decode_value(value):
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if isinstance(value, dict):
        if "__type__" in value:
            if (set(value) != {"__type__", "items"} or value["__type__"] != "tuple"
                    or not isinstance(value["items"], list)):
                raise ValueError("Invalid typed configuration value.")
            return tuple(_decode_value(item) for item in value["items"])
        if "estimator_class" in value:
            return model_from_configuration(value)
        return {key: _decode_value(item) for key, item in value.items()}
    # Reuse strict scalar validation; JSON NaN/Infinity are rejected.
    return _encode_value(value)


def model_from_configuration(record: dict) -> BaseEstimator:
    """Reconstruct a fresh estimator from complete JSON configuration records.

    Parameters
    ----------
    record : dict
        Output of model_configuration, optionally after json.loads. Original
        Milestone 2 baseline records without a schema-version field also work.
        Reconstruction is limited to this module's seven known sklearn classes.

    Returns
    -------
    sklearn.base.BaseEstimator
        Fresh, unfitted estimator with the recorded parameters and no learned
        state. Re-fitting needs the recorded data, folds, seeds and environment.

    Raises
    ------
    ValueError
        Schema/class/parameters are unsupported, or study controls are violated.
        Incompatible installed sklearn versions fail explicitly.
    TypeError
        Values are not supported JSON configuration types.
    """
    if not isinstance(record, dict) or not {"estimator_class", "parameters"}.issubset(record):
        raise ValueError("An estimator class and parameter dictionary are required.")
    if (set(record) - {"configuration_schema_version", "estimator_class", "parameters"}
            or type(record.get("configuration_schema_version", 1)) is not int
            or record.get("configuration_schema_version", 1) != 1
            or not isinstance(record["parameters"], dict)
            or not isinstance(record["estimator_class"], str)
            or record["estimator_class"] not in _CLASS_REGISTRY):
        raise ValueError("Unsupported configuration schema or estimator class.")
    parameters = {key: _decode_value(value) for key, value in record["parameters"].items()}
    try:
        estimator = _CLASS_REGISTRY[record["estimator_class"]](**parameters)
    except TypeError as exc:
        raise ValueError("Configuration parameters are incompatible with this sklearn class/version.") from exc
    if set(parameters) != set(estimator.get_params(deep=False)):
        raise ValueError("Configuration must include every constructor parameter for this sklearn version.")
    _check_policy(estimator)
    return estimator
