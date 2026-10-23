"""Thin entry point for mean and raw-input linear development comparisons."""

import pandas as pd

from airfoil_noise.models import BASELINE_MODELS
from airfoil_noise.validation import evaluate_development


def development_baselines(development: pd.DataFrame, row_folds: pd.DataFrame) -> dict:
    """Evaluate the two fixed baselines through the shared implementation.

    Parameters
    ----------
    development : pandas.DataFrame
        Development observations only, returned by split.load_development.
    row_folds : pandas.DataFrame
        Saved row-level fold assignments returned by the same loader.

    Returns
    -------
    dict
        Shared evaluation results, predictions, metrics and diagnostics.
        See validation.evaluate_development for validation and failure policy.

    Notes
    -----
    There is no feature transformation or fold-generation path here.
    """
    return evaluate_development(development, row_folds, model_names=BASELINE_MODELS)

