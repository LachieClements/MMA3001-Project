"""Observation and equal-condition regression errors, expressed in dB."""

import numpy as np
import pandas as pd


def _paired_arrays(truth, prediction) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for values in (truth, prediction):
        array = np.asarray(values)
        if (array.ndim != 1 or array.size == 0
                or not np.issubdtype(array.dtype, np.number)
                or np.issubdtype(array.dtype, np.complexfloating)
                or not np.isfinite(array).all()):
            raise ValueError("Truth and prediction must be nonempty finite numeric 1D arrays.")
        arrays.append(array.astype(float))
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("Truth and prediction must have identical shapes.")
    return arrays[0], arrays[1]


def pooled_metrics(truth, prediction) -> dict:
    """Calculate observation-weighted MAE, RMSE and supplementary R-squared.

    Parameters
    ----------
    truth, prediction : array-like of shape (n_observations,)
        Finite measured and predicted scaled sound pressure levels in dB.

    Returns
    -------
    dict
        ``n_observations``, ``mae_db``, ``rmse_db``, ``r2`` and
        ``r2_undefined_reason``. R-squared is NaN for fewer than two rows or
        exactly constant truth, including perfect predictions of a constant.
        Its reason is otherwise None. Negative defined R-squared is retained.

    Raises
    ------
    ValueError
        Arrays are empty, nonnumeric, nonfinite, non-1D or unequal in shape.
    """
    measured, predicted = _paired_arrays(truth, prediction)
    residual = measured - predicted
    reason = None
    r2 = float("nan")
    if len(measured) < 2:
        reason = "fewer_than_two_observations"
    elif np.all(measured == measured[0]):
        reason = "constant_truth"
    else:
        r2 = float(1 - np.sum(residual ** 2) / np.sum((measured - measured.mean()) ** 2))
    return {
        "n_observations": len(measured),
        "mae_db": float(np.mean(np.abs(residual))),
        "rmse_db": float(np.sqrt(np.mean(residual ** 2))),
        "r2": r2,
        "r2_undefined_reason": reason,
    }


def condition_metrics(truth, prediction, condition_ids) -> pd.DataFrame:
    """Calculate a separate MAE and RMSE for each operating condition.

    Parameters
    ----------
    truth, prediction : array-like of shape (n_observations,)
        Measured and predicted levels in dB.
    condition_ids : array-like of shape (n_observations,)
        Nonempty strings identifying conditions; repetitions are expected.

    Returns
    -------
    pandas.DataFrame
        Sorted condition_id, n_observations, mae_db and rmse_db columns.

    Raises
    ------
    ValueError
        Measurement validation fails or identifiers are missing/misaligned.
    """
    measured, predicted = _paired_arrays(truth, prediction)
    groups = np.asarray(condition_ids, dtype=object)
    if (groups.ndim != 1 or len(groups) != len(measured)
            or not all(isinstance(value, str) and value.strip() for value in groups)):
        raise ValueError("Condition IDs must be aligned, nonempty strings.")
    errors = pd.DataFrame({
        "condition_id": groups,
        "absolute_error": np.abs(measured - predicted),
        "squared_error": (measured - predicted) ** 2,
    })
    result = errors.groupby("condition_id", sort=True).agg(
        n_observations=("absolute_error", "size"),
        mae_db=("absolute_error", "mean"),
        mean_squared_error=("squared_error", "mean"),
    ).reset_index()
    result["rmse_db"] = np.sqrt(result.pop("mean_squared_error"))
    return result


def regression_metrics(truth, prediction, condition_ids) -> dict:
    """Add equal-condition MAE to the pooled metrics.

    Parameters
    ----------
    truth, prediction, condition_ids : array-like
        Aligned vectors as described in ``condition_metrics``.

    Returns
    -------
    dict
        Pooled metrics, n_conditions and equal_condition_mae_db. For G
        conditions, this primary metric is (1/G) sum_g [(1/n_g) sum_i
        abs(y_gi - prediction_gi)]. Each condition receives equal weight.
        Pass all out-of-fold rows together for the overall CV score; an
        unweighted average of fold scores gives different condition weights.

    Raises
    ------
    ValueError
        Measurements or condition identifiers fail validation.
    """
    per_condition = condition_metrics(truth, prediction, condition_ids)
    return {
        **pooled_metrics(truth, prediction),
        "n_conditions": len(per_condition),
        "equal_condition_mae_db": float(per_condition["mae_db"].mean()),
    }
