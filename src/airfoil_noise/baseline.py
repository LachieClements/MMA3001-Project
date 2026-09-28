"""Compare introductory regression baselines using development groups only."""

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from airfoil_noise.data import FEATURE_COLUMNS, TARGET_COLUMN
from airfoil_noise.split import CONDITION_COLUMNS


def regression_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare numeric inputs for an introductory linear baseline.

    Parameters
    ----------
    data : pandas.DataFrame
        Valid observations with frequency in Hz, angle in degrees, chord and
        thickness in m and velocity in m/s.

    Returns
    -------
    pandas.DataFrame
        Five numerical features. Frequency is expressed as log10(Hz) to
        reflect its wide 200--20,000 Hz range; other inputs are unchanged.
        Scaling is fitted inside each training fold by the model pipeline.

    Raises
    ------
    ValueError
        Frequency is nonpositive or required features are missing.
    """
    if not set(FEATURE_COLUMNS).issubset(data.columns):
        raise ValueError("Required airfoil features are missing.")
    features = data.loc[:, list(FEATURE_COLUMNS)].copy()
    if (features["frequency_hz"] <= 0).any():
        raise ValueError("Frequency must be positive for log10 transformation.")
    features["frequency_hz"] = np.log10(features["frequency_hz"])
    return features.rename(columns={"frequency_hz": "log10_frequency_hz"})


def development_baselines(
    development: pd.DataFrame, *, n_splits: int = 5
) -> pd.DataFrame:
    """Evaluate two baseline predictors on grouped development folds.

    Parameters
    ----------
    development : pandas.DataFrame
        Development rows only, obtained via the checked split manifest.
        Do not pass final-test observations to this function.
    n_splits : int, default 5
        Number of non-overlapping operating-condition folds.

    Returns
    -------
    pandas.DataFrame
        Per-fold model, sample counts, mean absolute error (dB), root mean
        squared error (dB), and elapsed fit-plus-prediction seconds. The
        models are fitted afresh for each fold; no final-test row is used.

    Raises
    ------
    ValueError
        Too few distinct operating conditions for the requested folds.

    Notes
    -----
    The constant baseline predicts the training mean. The second baseline is
    standardised ordinary least squares on the five features. The runtime is
    for a small dataset and is intended only as a local comparison.
    """
    groups = development.groupby(list(CONDITION_COLUMNS), sort=True).ngroup()
    if n_splits < 2 or groups.nunique() < n_splits:
        raise ValueError("Need at least n_splits distinct development conditions.")
    x = regression_features(development)
    y = development[TARGET_COLUMN]
    models = {
        "training_mean": DummyRegressor(strategy="mean"),
        "linear_log_frequency": make_pipeline(StandardScaler(), LinearRegression()),
    }
    records: list[dict[str, float | int | str]] = []
    folds = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
    for model_name, estimator in models.items():
        for fold, (fit_idx, valid_idx) in enumerate(folds, start=1):
            model = clone(estimator)
            start = perf_counter()
            model.fit(x.iloc[fit_idx], y.iloc[fit_idx])
            predictions = model.predict(x.iloc[valid_idx])
            elapsed = perf_counter() - start
            truth = y.iloc[valid_idx]
            records.append({
                "model": model_name,
                "fold": fold,
                "n_train": len(fit_idx),
                "n_validation": len(valid_idx),
                "mae_db": float(mean_absolute_error(truth, predictions)),
                "rmse_db": float(np.sqrt(mean_squared_error(truth, predictions))),
                "fit_predict_seconds": elapsed,
            })
    return pd.DataFrame.from_records(records)

