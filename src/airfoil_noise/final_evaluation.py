"""One-pass evaluation of frozen models, with reporting from saved records."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from airfoil_noise.data import (
    FEATURE_COLUMNS, SOURCE_SHA256, TARGET_COLUMN, feature_matrix, file_sha256,
    load_airfoil, validate_identifiers,
)
from airfoil_noise.development import check_threads, development_signature, write_json, write_table
from airfoil_noise.metrics import condition_metrics, regression_metrics
from airfoil_noise.models import model_configuration, model_from_configuration
from airfoil_noise.split import apply_manifest, read_frozen_splits
from airfoil_noise.validation import _fitted_diagnostics, environment_details


MODEL_ORDER = ("training_mean", "linear", "tree", "svr", "mlp")
APPROVED_HASHES = {
    "selection": "93a740f1111ee554c67fc54f246cfc41f2d49216e5fad7dff00a1f913dad7d2e",
    "recommendation": "75afe0565d0314d8a9f284fa7f96582a1970adf1745e4951f9a00e3b77258a8c",
    "configurations": {
        "training_mean": "8207df3c3fc204d71b91e7039541845039422cddde5d2af65f42d70673b4b8b6",
        "linear": "528ec3949ab9dcc0941bc4a5c16f3f603b921b0c94ba771fc16ba9d5265caa04",
        "tree": "bfe382800ab16898a39e6ca119a92c212c3ce70b815dedc52ca0eb7be11c27c2",
        "svr": "3e35b4c034c9bd7ee7ea4d9888e68bf6e2fd3a120aa35493b777c3debb0fc6f0",
        "mlp": "c340318227047df9d71ce8ef07c39f23c936b7a534ad8cc3d853b95a2fb76f6a",
    },
}
PREDICTION_COLUMNS = (
    "model", "configuration_id", "row_id", "condition_id", "truth_db", "prediction_db",
)
RECORD_FILES = (
    "run.json", "selection.json", "recommendation.json", "holdout_inputs.csv",
    "predictions.csv", "diagnostics.json",
)


def configuration_hash(configuration: dict) -> str:
    """Hash sorted compact UTF-8 JSON, retaining classes and complete parameters."""
    content = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_frozen_choices(selection_path: str | Path, recommendation_path: str | Path,
                        *, expected_hashes: dict | None = None) -> dict:
    """Check frozen file/configuration hashes and reconstruct settings without fitting.

    Parameters
    ----------
    selection_path, recommendation_path : str or pathlib.Path
        Saved Milestone 4 records. Original bytes are preserved in each new run.
    expected_hashes : dict, optional
        Defaults to the approved real-study anchors. Synthetic tests explicitly
        supply fixture hashes; the real CLI exposes no override for these anchors.

    Returns
    -------
    dict
        Validated records, paths and hashes. Every model is present; primary SVR,
        seed-17 MLP, disabled automatic stopping and complete JSON are checked.
    """
    expected = APPROVED_HASHES if expected_hashes is None else expected_hashes
    paths = {"selection": Path(selection_path), "recommendation": Path(recommendation_path)}
    records = {}
    for name, path in paths.items():
        if file_sha256(path) != expected[name]:
            raise ValueError(f"Frozen {name} checksum differs from the approved record.")
        records[name] = json.loads(path.read_text(encoding="utf-8"))
    selection, recommendation = records["selection"], records["recommendation"]
    if (selection["kind"] != "frozen_development_selection"
            or selection["final_test_evaluated"] is not False
            or set(selection["selected"]) != set(MODEL_ORDER)
            or recommendation["kind"] != "frozen_development_engineering_recommendation"
            or recommendation["recommended_model"] != "svr"
            or recommendation["final_test_evaluated"] is not False
            or recommendation["selection_sha256"] != expected["selection"]):
        raise ValueError("Require all five frozen choices and the preselected SVR recommendation.")
    hashes = {}
    for name in MODEL_ORDER:
        item = selection["selected"][name]
        configuration = item["configuration"]
        hashes[name] = configuration_hash(configuration)
        if (item["model"] != name or item["eligible"] is not True
                or hashes[name] != expected["configurations"][name]):
            raise ValueError(f"Frozen configuration mismatch for {name}.")
        estimator = model_from_configuration(configuration)
        if model_configuration(estimator) != configuration:
            raise ValueError("Configuration reconstruction must preserve every setting.")
        regressor = estimator[-1] if hasattr(estimator, "steps") else estimator
        seed = regressor.get_params(deep=False).get("random_state")
        if seed != item["representative_seed"] or (name in ("tree", "mlp") and seed != 17):
            raise ValueError("Use the predetermined representative seeds, not a best seed.")
    if (recommendation["configuration_id"] != selection["selected"]["svr"]["configuration_id"]
            or recommendation["configuration"] != selection["selected"]["svr"]["configuration"]):
        raise ValueError("Recommendation and selected SVR configuration disagree.")
    return {
        **records, "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "file_sha256": {name: expected[name] for name in paths},
        "configuration_sha256": hashes,
    }


def _numeric_vector(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Missing numeric column {column}.")
    values = frame[column].to_numpy()
    if (not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_)
            or np.issubdtype(values.dtype, np.complexfloating)
            or not np.isfinite(values).all()):
        raise ValueError(f"{column} must contain finite real numbers.")
    return values.astype(float)


def _separated_inputs(development: pd.DataFrame, holdout_inputs: pd.DataFrame) -> tuple:
    validate_identifiers(development)
    validate_identifiers(holdout_inputs)
    if TARGET_COLUMN in holdout_inputs:
        raise ValueError("The inference boundary accepts holdout inputs/IDs, never holdout targets.")
    if (set(development["row_id"]) & set(holdout_inputs["row_id"])
            or set(development["condition_id"]) & set(holdout_inputs["condition_id"])):
        raise ValueError("Development and holdout rows and conditions must be disjoint.")
    development = development.sort_values("row_id").reset_index(drop=True)
    holdout_inputs = holdout_inputs.sort_values("row_id").reset_index(drop=True)
    return (development, holdout_inputs, feature_matrix(development),
            _numeric_vector(development, TARGET_COLUMN), feature_matrix(holdout_inputs))


def _valid_fitted_diagnostics(model: str, fitted: dict) -> None:
    if model == "mlp":
        details = fitted["steps"]["regressor"]
        losses = details.get("loss_curve_", [])
        if (not losses or len(losses) != details.get("n_iter_")
                or details.get("loss_") is None
                or not np.isfinite(np.asarray(losses, dtype=float)).all()
                or not np.isfinite(details["loss_"])):
            raise ValueError("MLP loss diagnostics must be complete and finite.")


def fit_predict_frozen(development: pd.DataFrame, holdout_inputs: pd.DataFrame,
                       frozen: dict) -> dict:
    """Fit each saved configuration once; infer without receiving holdout truth.

    Parameters
    ----------
    development : pandas.DataFrame
        Training observations, target and stable identifiers.
    holdout_inputs : pandas.DataFrame
        Disjoint conditions, five original inputs and identifiers. A target
        column is rejected. Input preparation is outside diagnostic timers.
    frozen : dict
        Output of read_frozen_choices. Fresh complete estimators are constructed
        before fitting; no search, CV, retry or training prediction occurs.

    Returns
    -------
    dict
        Input-only prediction records and diagnostics. A local failure is
        retained while other families are attempted once. Finite convergence-
        warning predictions remain eligible; any failed model makes status failed.
    """
    _, inputs, x_train, y_train, x_test = _separated_inputs(development, holdout_inputs)
    estimators = {}
    for name in MODEL_ORDER:
        configuration = frozen["selection"]["selected"][name]["configuration"]
        if configuration_hash(configuration) != frozen["configuration_sha256"][name]:
            raise ValueError("Frozen configuration was changed after preflight.")
        estimators[name] = model_from_configuration(configuration)
    predictions, diagnostics = [], []
    for name, estimator in estimators.items():
        record = {
            "model": name, "status": "failed", "fit_calls": 0, "prediction_calls": 0,
            "fit_seconds": None, "prediction_seconds": None, "warnings": [],
            "convergence_warning": False, "error": None, "fitted": {},
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            phase = "fit"
            start = perf_counter()
            try:
                record["fit_calls"] += 1
                estimator.fit(x_train, y_train)
                record["fit_seconds"] = perf_counter() - start
                phase = "prediction"
                start = perf_counter()
                record["prediction_calls"] += 1
                predicted = np.asarray(estimator.predict(x_test))
                record["prediction_seconds"] = perf_counter() - start
                if (predicted.shape != (len(inputs),) or not np.issubdtype(predicted.dtype, np.number)
                        or np.iscomplexobj(predicted) or not np.isfinite(predicted).all()):
                    raise ValueError("Require one finite real prediction per holdout observation.")
                phase = "diagnostics"
                record["fitted"] = _fitted_diagnostics(estimator)
                _valid_fitted_diagnostics(name, record["fitted"])
                rows = inputs[["row_id", "condition_id"]].copy()
                rows.insert(0, "configuration_id", frozen["selection"]["selected"][name]["configuration_id"])
                rows.insert(0, "model", name)
                rows["prediction_db"] = predicted
                predictions.append(rows)
                record["status"] = "success"
            except Exception as exc:
                if phase in ("fit", "prediction") and record[f"{phase}_seconds"] is None:
                    record[f"{phase}_seconds"] = perf_counter() - start
                record["error"] = {"phase": phase, "type": type(exc).__name__, "message": str(exc)}
            record["warnings"] = [
                {"category": item.category.__name__, "message": str(item.message)} for item in caught
            ]
            record["convergence_warning"] = any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            )
        diagnostics.append(record)
    columns = [column for column in PREDICTION_COLUMNS if column != "truth_db"]
    return {
        "status": "success" if all(row["status"] == "success" for row in diagnostics) else "failed",
        "predictions": pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(columns=columns),
        "diagnostics": diagnostics,
    }


def _check_real_provenance(development: pd.DataFrame, holdout: pd.DataFrame,
                           frozen: dict, provenance: dict) -> None:
    expected = {"development": (1194, 84), "holdout": (309, 22)}
    for name, frame in (("development", development), ("holdout", holdout)):
        if (len(frame), frame["condition_id"].nunique()) != expected[name]:
            raise ValueError(f"Unexpected {name} coverage for the approved final experiment.")
    if (version("scikit-learn") != "1.7.2"
            or provenance["source_sha256"] != SOURCE_SHA256
            or frozen["file_sha256"] != {key: APPROVED_HASHES[key] for key in ("selection", "recommendation")}
            or frozen["configuration_sha256"] != APPROVED_HASHES["configurations"]
            or development_signature(development) != frozen["selection"]["development_sha256"]
            or provenance["frozen_split_metadata"] != frozen["selection"]["provenance"]["frozen_split_metadata"]):
        raise ValueError("Final provenance/version differs from the frozen development design.")


def run_final_evaluation(development: pd.DataFrame, holdout: pd.DataFrame, frozen: dict,
                         directory: str | Path, *, provenance: dict,
                         run_kind: str = "final_holdout") -> dict:
    """Save one bounded evaluation and derive its report from saved predictions.

    Parameters
    ----------
    development, holdout : pandas.DataFrame
        Already partitioned, disjoint observations including truth. Only the
        development targets enter fit_predict_frozen; truth is attached by IDs.
    frozen : dict
        Validated choices returned by read_frozen_choices.
    directory : str or pathlib.Path
        New protected run directory. Existing directories are rejected before fits.
    provenance : dict
        Source/split checksums, implementation hashes and command supplied by CLI.
    run_kind : {'final_holdout', 'synthetic_verification'}
        Real mode enforces the approved hashes/counts/version. Synthetic mode
        is for small fixtures and explicitly rejects the official dataset.

    Returns
    -------
    dict
        Evaluation status and attempted call counts. Only a fully validated
        five-model run/report receives completion.json. Failures retain partial
        records without summary/figures; no replacement fits are made.
    """
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(destination)
    inputs = holdout.drop(columns=[TARGET_COLUMN])
    _separated_inputs(development, inputs)
    _numeric_vector(holdout, TARGET_COLUMN)
    if run_kind == "final_holdout":
        _check_real_provenance(development, holdout, frozen, provenance)
    elif (run_kind != "synthetic_verification"
          or provenance.get("source_sha256") == SOURCE_SHA256
          or development.attrs.get("source_sha256") == SOURCE_SHA256):
        raise ValueError("Synthetic verification must not use the official source.")
    # Validate record serialization and unchanged external anchors before fitting.
    json.dumps(provenance, allow_nan=False)
    for name, path in frozen["paths"].items():
        if file_sha256(path) != frozen["file_sha256"][name]:
            raise ValueError("Frozen file changed after it was loaded.")
    with threadpool_limits(limits=1):
        check_threads()
        destination.mkdir(parents=True, exist_ok=False)
        for name, path in frozen["paths"].items():
            with (destination / f"{name}.json").open("xb") as stream:
                stream.write(Path(path).read_bytes())
        run = {
            "schema_version": 1, "kind": run_kind, "partition": "final_test" if run_kind == "final_holdout" else "synthetic_test",
            "started_utc": datetime.now(timezone.utc).isoformat(), "primary_model": "svr",
            "feature_columns": list(FEATURE_COLUMNS), "target_transformation": "none",
            "configuration_sha256": frozen["configuration_sha256"], "frozen_file_sha256": frozen["file_sha256"],
            "n_development": len(development), "n_holdout": len(holdout),
            "n_holdout_conditions": holdout["condition_id"].nunique(),
            "development_row_ids": sorted(development["row_id"]),
            "development_condition_ids": sorted(development["condition_id"].unique()),
            "models": {name: frozen["selection"]["selected"][name] for name in MODEL_ORDER},
            "environment": environment_details(), "provenance": provenance,
            "fit_budget": 5, "expected_prediction_records": 5 * len(holdout),
            "timing_note": "One fit and predict per model: diagnostics, not repeated benchmarks.",
            "post_evaluation_policy": "No tuning, new seeds/configurations or switching primary SVR.",
        }
        write_json(destination / "run.json", run)
        write_table(destination / "holdout_inputs.csv", inputs.sort_values("row_id").reset_index(drop=True))
        result = fit_predict_frozen(development, inputs, frozen)
        predictions = result["predictions"]
        truth = holdout.set_index("row_id")[TARGET_COLUMN]
        predictions["truth_db"] = truth.loc[predictions["row_id"]].to_numpy()
        write_json(destination / "diagnostics.json", result["diagnostics"])
        write_table(destination / "predictions.csv", predictions[list(PREDICTION_COLUMNS)])
        evaluation = {
            "status": "predictions_complete" if result["status"] == "success" else "failed",
            "fits_attempted": sum(row["fit_calls"] for row in result["diagnostics"]),
            "prediction_calls": sum(row["prediction_calls"] for row in result["diagnostics"]),
            "models_failed": sum(row["status"] != "success" for row in result["diagnostics"]),
            "fits_failed": sum(row["error"] is not None and row["error"]["phase"] == "fit" for row in result["diagnostics"]),
            "record_sha256": {name: file_sha256(destination / name) for name in RECORD_FILES},
        }
        for name, path in frozen["paths"].items():
            if file_sha256(path) != frozen["file_sha256"][name]:
                evaluation["status"] = "failed"
                evaluation["error"] = f"Frozen {name} changed during evaluation."
        write_json(destination / "evaluation.json", evaluation)
        if evaluation["status"] == "failed":
            return evaluation
        regenerate_final_report(destination, destination / "report")
        completion = {
            **evaluation, "status": "success_warning_qualified" if any(
                row["convergence_warning"] for row in result["diagnostics"]
            ) else "success",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "report_manifest_sha256": file_sha256(destination / "report" / "manifest.json"),
            "evaluation_sha256": file_sha256(destination / "evaluation.json"),
        }
        write_json(destination / "completion.json", completion)
    return completion


def load_final_records(directory: str | Path) -> dict:
    """Read and validate a complete saved prediction set without model calls.

    Returns
    -------
    dict
        Run metadata, inputs, predictions, diagnostics and metric tables.
        Input hashes/coverage/configurations and all cross-model truth/ID
        alignment are checked before any successful report can be written.
    """
    source = Path(directory)
    evaluation = json.loads((source / "evaluation.json").read_text(encoding="utf-8"))
    if (evaluation["status"] != "predictions_complete"
            or set(evaluation["record_sha256"]) != set(RECORD_FILES)):
        raise ValueError("A failed/incomplete evaluation cannot produce a success report.")
    for name, digest in evaluation["record_sha256"].items():
        if file_sha256(source / name) != digest:
            raise ValueError(f"Saved record checksum mismatch: {name}.")
    run = json.loads((source / "run.json").read_text(encoding="utf-8"))
    selected = json.loads((source / "selection.json").read_text(encoding="utf-8"))
    recommendation = json.loads((source / "recommendation.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((source / "diagnostics.json").read_text(encoding="utf-8"))
    inputs = pd.read_csv(source / "holdout_inputs.csv", float_precision="round_trip")
    predictions = pd.read_csv(source / "predictions.csv", float_precision="round_trip")
    if (run["primary_model"] != "svr" or run["feature_columns"] != list(FEATURE_COLUMNS)
            or set(run["models"]) != set(MODEL_ORDER)
            or recommendation["recommended_model"] != "svr"
            or len(diagnostics) != 5 or {row["model"] for row in diagnostics} != set(MODEL_ORDER)
            or any(row["status"] != "success" or row["fit_calls"] != 1 or row["prediction_calls"] != 1
                   for row in diagnostics)
            or evaluation["models_failed"] != 0 or evaluation["fits_failed"] != 0
            or evaluation["fits_attempted"] != 5 or evaluation["prediction_calls"] != 5):
        raise ValueError("Require five complete single-fit models and the unchanged primary recommendation.")
    for name in ("selection", "recommendation"):
        if file_sha256(source / f"{name}.json") != run["frozen_file_sha256"][name]:
            raise ValueError("Saved frozen choices differ from their original anchors.")
    validate_identifiers(inputs)
    feature_matrix(inputs)
    if (len(inputs) != run["n_holdout"] or inputs["condition_id"].nunique() != run["n_holdout_conditions"]
            or set(inputs["row_id"]) & set(run["development_row_ids"])
            or set(inputs["condition_id"]) & set(run["development_condition_ids"])
            or tuple(predictions.columns) != PREDICTION_COLUMNS
            or set(predictions["model"]) != set(MODEL_ORDER)
            or len(predictions) != run["expected_prediction_records"]
            or predictions.duplicated(["model", "row_id"]).any()):
        raise ValueError("Saved holdout coverage/identifiers are incomplete or conflicting.")
    reference = inputs.set_index("row_id")
    summaries, condition_tables, comparisons = [], [], []
    measured = None
    for name in MODEL_ORDER:
        item = run["models"][name]
        if (item != selected["selected"][name]
                or configuration_hash(item["configuration"]) != run["configuration_sha256"][name]):
            raise ValueError("Saved configuration records disagree.")
        rows = predictions[predictions["model"] == name].sort_values("row_id")
        if (set(rows["row_id"]) != set(reference.index) or len(rows) != len(reference)
                or rows["condition_id"].tolist() != reference.loc[rows["row_id"], "condition_id"].tolist()
                or not rows["configuration_id"].eq(item["configuration_id"]).all()):
            raise ValueError("Prediction rows do not align with saved holdout inputs/configurations.")
        truth, predicted = _numeric_vector(rows, "truth_db"), _numeric_vector(rows, "prediction_db")
        if measured is not None and not np.array_equal(truth, measured):
            raise ValueError("Truth must be identical across model prediction records.")
        measured = truth
        diagnostic = next(record for record in diagnostics if record["model"] == name)
        _valid_fitted_diagnostics(name, diagnostic["fitted"])
        summaries.append({
            "model": name, "configuration_id": item["configuration_id"],
            "primary": name == "svr", "seed": item["representative_seed"],
            "warning_qualified": diagnostic["convergence_warning"],
            "fit_seconds": diagnostic["fit_seconds"], "prediction_seconds": diagnostic["prediction_seconds"],
            **regression_metrics(truth, predicted, rows["condition_id"]),
        })
        table = condition_metrics(truth, predicted, rows["condition_id"])
        table.insert(0, "model", name)
        condition_tables.append(table)
        for stage, score in (
            ("development_oof_representative", item["representative_oof_equal_condition_mae_db"]),
            (run["partition"], summaries[-1]["equal_condition_mae_db"]),
        ):
            comparisons.append({
                "model": name, "stage": stage, "seed": item["representative_seed"],
                "equal_condition_mae_db": score, "configuration_selection_score_db": item["selection_score_db"],
                "selection_score_note": "MLP selection averages three seed scores; representative uses seed 17.",
            })
    return {
        "run": run, "inputs": inputs, "predictions": predictions, "diagnostics": diagnostics,
        "summary": pd.DataFrame(summaries), "condition_metrics": pd.concat(condition_tables, ignore_index=True),
        "development_vs_final": pd.DataFrame(comparisons),
    }


def regenerate_final_report(run_directory: str | Path, directory: str | Path) -> dict:
    """Regenerate tables and figures using only validated saved predictions.

    Parameters
    ----------
    run_directory : str or pathlib.Path
        Completed prediction records. Original raw data, estimators and source
        selection paths are not read or needed for regeneration.
    directory : str or pathlib.Path
        New report destination; existing reports are never overwritten.

    Returns
    -------
    dict
        Report file hashes and zero fit/prediction-call counters.
    """
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(destination)
    records = load_final_records(run_directory)
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("summary", "condition_metrics", "development_vs_final"):
        write_table(destination / f"{name}.csv", records[name])
    from airfoil_noise.plotting import plot_final
    plot_final(run_directory, destination / "figures")
    manifest = {
        "kind": "saved_prediction_report", "fits_attempted": 0, "prediction_calls": 0,
        "source_prediction_sha256": file_sha256(Path(run_directory) / "predictions.csv"),
        "primary_model": "svr", "files": {
            str(path.relative_to(destination)).replace("\\", "/"): file_sha256(path)
            for path in sorted(destination.rglob("*")) if path.is_file()
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def execute_final_command(args) -> int:
    """CLI adapter for the explicitly approved real holdout run; never called by checks."""
    if Path(args.output_dir).exists():
        raise FileExistsError(args.output_dir)
    frozen = read_frozen_choices(args.selection, args.recommendation)
    if version("scikit-learn") != "1.7.2":
        raise ValueError("Use the recorded scikit-learn 1.7.2 environment.")
    data = load_airfoil(args.data, expected_sha256=SOURCE_SHA256)
    manifest, _, metadata = read_frozen_splits(data, args.directory)
    development, holdout = apply_manifest(data, manifest)
    git = {}
    for name, arguments in (("HEAD", ["rev-parse", "HEAD"]), ("status", ["status", "--short"])):
        result = subprocess.run(["git", "--no-optional-locks", *arguments], capture_output=True, text=True)
        git[name] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    provenance = {
        "source_path": str(Path(args.data).resolve()), "source_sha256": data.attrs["source_sha256"],
        "split_directory": str(Path(args.directory).resolve()), "frozen_split_metadata": metadata,
        "split_file_sha256": {path.name: file_sha256(path) for path in sorted(Path(args.directory).glob("*")) if path.is_file()},
        "implementation_sha256": {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "command": args.invocation, "git": git,
    }
    result = run_final_evaluation(development, holdout, frozen, args.output_dir, provenance=provenance)
    if result["status"] == "failed":
        raise RuntimeError(f"Final evaluation failed; partial records retained in {args.output_dir}. No retry.")
    print(f'Final records saved: {result["status"]}. Primary recommendation remains SVR.')
    return 0
