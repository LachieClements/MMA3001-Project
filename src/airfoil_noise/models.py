"""Build the two raw-input baselines with explicit, reproducible settings."""

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression


BASELINE_MODELS = ("training_mean", "linear")


def build_model(name: str) -> RegressorMixin:
    """Return a new, unfitted baseline estimator.

    Parameters
    ----------
    name : {'training_mean', 'linear'}
        The mean is calculated from training observations, giving each row
        equal weight. Linear regression uses an intercept and the original
        five inputs, without scaling, transformations or additional features.

    Returns
    -------
    sklearn.base.RegressorMixin
        Independent estimator. Call ``get_params(deep=True)`` for all settings.

    Raises
    ------
    ValueError
        The model name is unsupported. Configuration search is outside this
        milestone; there is one declared configuration per method.
    """
    if name == "training_mean":
        return DummyRegressor(strategy="mean", constant=None, quantile=None)
    if name == "linear":
        return LinearRegression(
            fit_intercept=True, copy_X=True, n_jobs=1, positive=False,
        )
    raise ValueError(f"Unknown model {name!r}; choose from {BASELINE_MODELS}.")
