"""Shared development-only evaluation on supplied, frozen row-fold assignments."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_info

from airfoil_noise.data import FEATURE_COLUMNS, TARGET_COLUMN, feature_matrix, validate_identifiers
from airfoil_noise.metrics import condition_metrics, regression_metrics
from airfoil_noise.models import BASELINE_MODELS, MODEL_NAMES, build_model, model_configuration


PREDICTION_COLUMNS = (
    "model", "row_id", "condition_id", "fold", "truth_db", "prediction_db",
)


def _validated_inputs(development: pd.DataFrame, row_folds: pd.DataFrame):
    validate_identifiers(development)
    if (TARGET_COLUMN not in development
            or not pd.api.types.is_numeric_dtype(development[TARGET_COLUMN])
            or pd.api.types.is_bool_dtype(development[TARGET_COLUMN])
            or pd.api.types.is_complex_dtype(development[TARGET_COLUMN])
            or development[TARGET_COLUMN].isna().any()
            or not np.isfinite(development[TARGET_COLUMN].to_numpy(dtype=float)).all()):
        raise ValueError("Development targets must be finite numeric levels in dB.")
    required = {"row_id", "condition_id", "fold"}
    if (not row_folds.columns.is_unique or set(row_folds.columns) != required
            or row_folds.empty or row_folds.isna().any().any()
            or row_folds["row_id"].duplicated().any()
            or set(row_folds["row_id"]) != set(development["row_id"])):
        raise ValueError("Fold assignments must uniquely cover exactly the development row IDs.")
    aligned = row_folds.set_index("row_id").loc[development["row_id"]].reset_index()
    if aligned["condition_id"].tolist() != development["condition_id"].tolist():
        raise ValueError("Fold condition IDs do not match development observations.")
    folds = aligned["fold"]
    if (not pd.api.types.is_integer_dtype(folds) or pd.api.types.is_bool_dtype(folds)
            or sorted(folds.unique()) != list(range(1, folds.nunique() + 1))
            or folds.nunique() < 2):
        raise ValueError("Fold labels must be consecutive integers starting at 1, with at least two folds.")
    if aligned.groupby("condition_id")["fold"].nunique().ne(1).any():
        raise ValueError("Every condition must belong to exactly one validation fold.")
    return feature_matrix(development), development[TARGET_COLUMN].to_numpy(dtype=float), aligned


def _fitted_diagnostics(estimator) -> dict:
    # These describe fitted complexity/coefficients; they are not cost units.
    details = {"estimator_class": f"{type(estimator).__module__}.{type(estimator).__qualname__}"}
    for attribute in ("constant_", "coef_", "intercept_", "rank_", "singular_",
                      "n_features_in_", "feature_names_in_"):
        if hasattr(estimator, attribute):
            details[attribute] = _json_ready(getattr(estimator, attribute))
    if isinstance(estimator, Pipeline):
        details["steps"] = {name: _fitted_diagnostics(step) for name, step in estimator.steps}
    if isinstance(estimator, StandardScaler):
        for attribute in ("mean_", "var_", "scale_", "n_samples_seen_"):
            details[attribute] = _json_ready(getattr(estimator, attribute))
    if isinstance(estimator, DecisionTreeRegressor):
        details.update(node_count=int(estimator.tree_.node_count),
                       depth=estimator.get_depth(), leaf_count=estimator.get_n_leaves())
    if isinstance(estimator, SVR):
        details.update(support_vector_count=len(estimator.support_),
                       n_support_=_json_ready(estimator.n_support_),
                       n_iter_=_json_ready(estimator.n_iter_), fit_status_=int(estimator.fit_status_))
    if isinstance(estimator, MLPRegressor):
        for attribute in ("n_iter_", "loss_", "loss_curve_", "best_loss_", "t_", "n_layers_", "n_outputs_"):
            if hasattr(estimator, attribute):
                details[attribute] = _json_ready(getattr(estimator, attribute))
        details["parameter_count"] = sum(array.size for array in estimator.coefs_ + estimator.intercepts_)
    return details


def _prepare_models(model_names, configurations) -> tuple[dict, dict]:
    if configurations is not None and model_names is not None:
        raise ValueError("Specify model_names or configurations, not both.")
    if configurations is None:
        names = BASELINE_MODELS if model_names is None else model_names
        if (not isinstance(names, (tuple, list)) or not names
                or not all(isinstance(name, str) and name in MODEL_NAMES for name in names)
                or len(set(names)) != len(names)):
            raise ValueError(f"Specify distinct model names from {MODEL_NAMES}.")
        configurations = {name: {"model": name} for name in names}
    if not isinstance(configurations, dict) or not configurations:
        raise ValueError("configurations must be a nonempty dictionary keyed by configuration ID.")
    prototypes, methods = {}, {}
    for label, specification in configurations.items():
        if (not isinstance(label, str) or not label.strip()
                or not isinstance(specification, dict) or "model" not in specification
                or set(specification) - {"model", "parameters", "seed"}
                or not isinstance(specification["model"], str)
                or specification["model"] not in MODEL_NAMES):
            raise ValueError("Each configuration needs a nonempty ID and a supported model specification.")
        methods[label] = specification["model"]
        options = {key: specification[key] for key in ("parameters", "seed") if key in specification}
        prototypes[label] = build_model(methods[label], **options)
    return prototypes, methods


def evaluate_development(
    development: pd.DataFrame,
    row_folds: pd.DataFrame,
    *,
    model_names: tuple[str, ...] | None = None,
    configurations: dict[str, dict] | None = None,
    record_training: bool = False,
) -> dict:
    """Fit fresh estimators and collect one prediction per development row/model.

    Parameters
    ----------
    development : pandas.DataFrame
        Development observations only, including five inputs, target and stable
        IDs. Obtain this through ``split.load_development``. This function has
        no access to final-test observations, targets or source-file paths.
    row_folds : pandas.DataFrame
        Saved row_id, condition_id and fold mapping, e.g. from load_development.
        Order may differ from development; alignment uses row_id, never index.
        No folds are generated here. The caller supplies the approved mapping.
    model_names : tuple of str, optional
        Distinct supported methods using construction defaults. Omitting both
        options preserves the original mean/linear evaluation.
    configurations : dict of str to dict, optional
        Configuration ID mapped to model, optional parameters and optional seed.
        Mutually exclusive with model_names. For example,
        {'mlp_seed17': {'model': 'mlp', 'seed': 17, 'parameters': {'max_iter': 50}}}.
        Several explicitly supplied settings/seeds for a method are allowed;
        no configurations are generated or selected by this function. The
        existing 'model' column holds these IDs, linked to methods in run.json.
    record_training : bool, default False
        Also save predictions on each fold's training rows, labelled with
        fit_fold separately from their saved validation fold. These repeated
        training observations describe complexity; they never select models.

    Returns
    -------
    dict
        status, predictions, summary, condition_metrics, fold_metrics,
        diagnostics and run records. Tables are DataFrames. Failed fits or
        invalid predictions are recorded; status becomes 'failed' and all
        metric tables are empty, preventing a partial successful comparison.
        Successful predictions cover every row once per model. Overall metrics
        are calculated from all out-of-fold predictions, not fold averages.

    Raises
    ------
    ValueError
        Invalid measurements, incomplete/conflicting assignments or model names.

    Notes
    -----
    All configurations are built before fitting, then cloned independently for
    every fold, including their nested scalers and regressors. The original five features are
    passed in their declared order; IDs and truth never enter the feature
    matrix. Timings isolate fit and predict calls but use only one measurement
    per fit: these are preliminary diagnostics, not final speed benchmarks.
    """
    x, y, aligned = _validated_inputs(development, row_folds)
    started = datetime.now(timezone.utc).isoformat()
    start = perf_counter()
    prototypes, methods = _prepare_models(model_names, configurations)
    settings = {name: model_configuration(estimator) for name, estimator in prototypes.items()}
    preprocessing, seeds = {}, {}
    for name, estimator in prototypes.items():
        steps = estimator.steps[:-1] if isinstance(estimator, Pipeline) else []
        preprocessing[name] = [
            {"step_name": step_name, **model_configuration(step)} for step_name, step in steps
        ]
        regressor = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
        seeds[name] = regressor.get_params(deep=False).get("random_state")
    predictions, diagnostics, training = [], [], []
    for name, prototype in prototypes.items():
        for fold in sorted(aligned["fold"].unique()):
            estimator = clone(prototype)
            valid = aligned["fold"].to_numpy() == fold
            record = {
                "model": name, "fold": int(fold), "status": "failed",
                "n_train": int((~valid).sum()), "n_validation": int(valid.sum()),
                "fit_seconds": None, "prediction_seconds": None,
                "training_prediction_seconds": None,
                "warnings": [], "convergence_warning": False, "error": None, "fitted": {},
            }
            # Warnings are retained even if the same fit subsequently fails.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                phase = "fit"
                phase_start = perf_counter()
                try:
                    estimator.fit(x.iloc[np.flatnonzero(~valid)], y[~valid])
                    record["fit_seconds"] = perf_counter() - phase_start
                    phase = "prediction"
                    phase_start = perf_counter()
                    predicted = np.asarray(estimator.predict(x.iloc[np.flatnonzero(valid)]))
                    record["prediction_seconds"] = perf_counter() - phase_start
                    # Metric validation rejects wrong shape and nonfinite predictions.
                    regression_metrics(y[valid], predicted, aligned.loc[valid, "condition_id"])
                    rows = aligned.loc[valid].copy()
                    rows.insert(0, "model", name)
                    rows["truth_db"] = y[valid]
                    rows["prediction_db"] = predicted
                    record["fitted"] = _fitted_diagnostics(estimator)
                    if record_training:
                        phase = "training_prediction"
                        phase_start = perf_counter()
                        train_predicted = np.asarray(estimator.predict(x.iloc[np.flatnonzero(~valid)]))
                        record["training_prediction_seconds"] = perf_counter() - phase_start
                        regression_metrics(y[~valid], train_predicted, aligned.loc[~valid, "condition_id"])
                        train_rows = aligned.loc[~valid].copy()
                        train_rows.insert(0, "model", name)
                        train_rows["fit_fold"] = int(fold)
                        train_rows["truth_db"] = y[~valid]
                        train_rows["prediction_db"] = train_predicted
                        training.append(train_rows)
                    predictions.append(rows.loc[:, list(PREDICTION_COLUMNS)])
                    record["status"] = "success"
                except Exception as exc:
                    if record[f"{phase}_seconds"] is None:
                        record[f"{phase}_seconds"] = perf_counter() - phase_start
                    record["error"] = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
                record["warnings"] = [
                    {"category": warning.category.__name__, "message": str(warning.message)}
                    for warning in caught
                ]
                record["convergence_warning"] = any(
                    issubclass(warning.category, ConvergenceWarning) for warning in caught
                )
            diagnostics.append(record)
    complete = all(record["status"] == "success" for record in diagnostics)
    prediction_table = (pd.concat(predictions, ignore_index=True) if predictions
                        else pd.DataFrame(columns=PREDICTION_COLUMNS))
    if complete:
        if (len(prediction_table) != len(development) * len(prototypes)
                or prediction_table.duplicated(["model", "row_id"]).any()):
            raise RuntimeError("Internal error: incomplete or duplicated out-of-fold predictions.")
        prediction_table = prediction_table.sort_values(["model", "row_id"]).reset_index(drop=True)
    summaries, conditions, folds = [], [], []
    if complete:
        for name, rows in prediction_table.groupby("model", sort=True):
            summaries.append({"model": name, **regression_metrics(
                rows["truth_db"], rows["prediction_db"], rows["condition_id"],
            )})
            per_condition = condition_metrics(rows["truth_db"], rows["prediction_db"], rows["condition_id"])
            per_condition.insert(0, "model", name)
            conditions.append(per_condition)
            for fold, subset in rows.groupby("fold", sort=True):
                folds.append({"model": name, "fold": int(fold), **regression_metrics(
                    subset["truth_db"], subset["prediction_db"], subset["condition_id"],
                )})
    result = {
        "status": "success" if complete else "failed",
        "predictions": prediction_table,
        "summary": pd.DataFrame(summaries),
        "condition_metrics": pd.concat(conditions, ignore_index=True) if conditions else pd.DataFrame(),
        "fold_metrics": pd.DataFrame(folds),
        "diagnostics": diagnostics,
        "run": {
            "schema_version": 2, "kind": "development_cross_validation",
            "status": "success" if complete else "failed", "started_utc": started,
            "evaluation_seconds": perf_counter() - start,
            "models": settings, "model_methods": methods, "feature_columns": list(FEATURE_COLUMNS),
            "feature_transformations": preprocessing, "target_transformation": "none",
            "model_random_seeds": seeds,
            "seed_explanation": "Regressor random_state; null for deterministic methods. Split seeds are separate provenance.",
            "primary_metric": "equal_condition_mae_db over all out-of-fold rows",
            "r2_policy": "Undefined for n < 2 or exactly constant truth; CSV blank / JSON null.",
            "n_observations": len(development), "n_conditions": development["condition_id"].nunique(),
            "n_folds": aligned["fold"].nunique(), "fits_attempted": len(diagnostics),
            "fits_succeeded": sum(record["status"] == "success" for record in diagnostics),
            "fits_failed": sum(record["status"] == "failed" for record in diagnostics),
            "n_configurations": len(prototypes),
            "configurations_per_model": {method: list(methods.values()).count(method) for method in sorted(set(methods.values()))},
            "search_performed": False,
            "training_predictions_recorded": record_training,
            "configuration_selection": "None: only the explicitly supplied configurations are evaluated.",
            "timing_note": "Single fit/predict timings, without warmup or repetitions; preliminary diagnostics only.",
            "environment": environment_details(),
        },
    }
    if record_training:
        result["training_predictions"] = pd.concat(training, ignore_index=True) if training else pd.DataFrame()
        result["training_summary"] = pd.DataFrame([
            {"model": name, **regression_metrics(rows["truth_db"], rows["prediction_db"], rows["condition_id"])}
            for name, rows in result["training_predictions"].groupby("model")
        ]) if complete else pd.DataFrame()
    return result


def environment_details() -> dict:
    """Return software, machine and thread settings for an experiment record.

    Returns
    -------
    dict
        Python executable/version, OS, CPU/logical count, dependency versions,
        relevant thread environment and currently loaded numerical thread pools.
        These records describe the run; they do not measure peak RAM.
    """
    variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    packages = {}
    for package in ("mma3001-airfoil-noise", "numpy", "pandas", "scipy", "scikit-learn", "pytest", "pdoc", "matplotlib", "threadpoolctl"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None  # Development extras are optional at runtime.
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "packages": packages,
        "thread_environment": {name: os.environ.get(name) for name in variables},
        "thread_pools": threadpool_info(),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_development_results(results: dict, directory: str | Path, *, provenance: dict) -> None:
    """Save a new traceable development run, refusing to replace any directory.

    Parameters
    ----------
    results : dict
        Output of evaluate_development. Failed runs retain partial predictions
        and diagnostic records, but no metric tables are written.
    directory : str or pathlib.Path
        New run directory. A quick check must use a distinct directory from a
        full experiment; this function never labels a development run 'final'.
    provenance : dict
        Source checksum and approved frozen-split metadata supplied by caller.

    Raises
    ------
    FileExistsError
        Output directory already exists, even if it is empty.
    """
    destination = Path(directory)
    # Validate JSON before creating files. Complete pipeline settings must
    # already be structured records, never live sklearn objects or repr strings.
    json_records = {
        "diagnostics": json.dumps(_json_ready(results["diagnostics"]), indent=2, allow_nan=False),
        "run": json.dumps(_json_ready({**results["run"], "provenance": provenance}), indent=2, allow_nan=False),
    }
    destination.mkdir(parents=True, exist_ok=False)
    tables = ["predictions"]
    if results["status"] == "success":
        tables += ["summary", "condition_metrics", "fold_metrics"]
        if "training_predictions" in results:
            tables += ["training_predictions", "training_summary"]
    for name in tables:
        with (destination / f"{name}.csv").open("x", encoding="utf-8", newline="") as stream:
            results[name].to_csv(stream, index=False, lineterminator="\n", float_format="%.17g")
    for name, content in json_records.items():
        with (destination / f"{name}.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content + "\n")
