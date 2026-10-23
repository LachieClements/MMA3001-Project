"""Separate repeated fit, fixed-batch prediction and individual-call timings."""

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from airfoil_noise.data import TARGET_COLUMN, feature_matrix, file_sha256, validate_identifiers
from airfoil_noise.development import check_threads, development_signature, protocol_signature, write_json, write_table
from airfoil_noise.models import model_from_configuration
from airfoil_noise.validation import _fitted_diagnostics, environment_details


def prepared_inputs(development: pd.DataFrame, batch_rows: int, single_rows: int) -> tuple:
    """Prepare fixed, evenly spaced batch and one-row frames outside timers.

    Returns
    -------
    tuple
        Five-column training matrix/targets, one batch frame, list of one-row
        frames, and exact selected row IDs. No final-test observations are used.
    """
    validate_identifiers(development)
    if min(batch_rows, single_rows) < 1 or max(batch_rows, single_rows) > len(development):
        raise ValueError("Timing sizes must be positive and no larger than development data.")
    ordered = development.sort_values("row_id")
    x = feature_matrix(ordered)
    y = ordered[TARGET_COLUMN].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Benchmark development targets must be finite.")
    batch_positions = np.linspace(0, len(x) - 1, batch_rows, dtype=int)
    single_positions = np.linspace(0, len(x) - 1, single_rows, dtype=int)
    batch = x.iloc[batch_positions].copy()
    singles = [x.iloc[[position]].copy() for position in single_positions]
    identifiers = {
        "batch_row_ids": ordered.iloc[batch_positions]["row_id"].tolist(),
        "single_row_ids": ordered.iloc[single_positions]["row_id"].tolist(),
    }
    return x, y, batch, singles, identifiers


def prediction_block(estimator, inputs: list[pd.DataFrame], rounds: int, counter: dict) -> float:
    """Time separate predict calls, cycling through prepared frames each round.

    The timer includes Python dispatch/loop/counting and pipeline preprocessing.
    DataFrame creation, indexing, loading and output writing happen outside it.
    counter records actual attempted predict calls even when a call raises.
    """
    start = perf_counter()
    for _ in range(rounds):
        for frame in inputs:
            counter["calls"] += 1
            estimator.predict(frame)
    return perf_counter() - start


def prediction_timings(estimators: dict, batch: pd.DataFrame, singles: list,
                       settings: dict) -> tuple[pd.DataFrame, dict]:
    """Warm up, calibrate and cyclically measure batch and single-call timings.

    Each single round makes len(singles) separate calls (32 in the full study).
    Reported latency is the median of block means, not an individual-call p95.
    Raw warmup/calibration/measured durations and actual call counts are saved.
    """
    records, failures = [], {}
    for mode, inputs in (("batch", [batch]), ("single", singles)):
        active, rounds_by_model = [], {}
        for name, estimator in estimators.items():
            counter = {"calls": 0}
            try:
                for stage in ("warmup", "calibration"):
                    rounds = settings["prediction_warmup_rounds"] if stage == "warmup" else 1
                    while True:
                        before = counter["calls"]
                        duration = prediction_block(estimator, inputs, rounds, counter)
                        records.append({
                            "model": name, "mode": mode, "stage": stage, "block": None,
                            "rounds": rounds, "calls_per_round": len(inputs),
                            "prediction_calls": counter["calls"] - before,
                            "seconds": duration,
                            "seconds_per_call": duration / (counter["calls"] - before),
                        })
                        if stage == "warmup" or duration >= settings["minimum_block_seconds"]:
                            break
                        if rounds >= settings["maximum_rounds_per_block"]:
                            raise RuntimeError("Calibration cap reached before minimum block duration.")
                        rounds = min(2 * rounds, settings["maximum_rounds_per_block"])
                rounds_by_model[name] = rounds
                active.append(name)
            except Exception as exc:
                failures[f"{name}:{mode}"] = {
                    "phase": "warmup/calibration", "type": type(exc).__name__,
                    "message": str(exc), "actual_attempted_calls": counter["calls"],
                }
        for block in range(settings["blocks"]):
            order = active[block % len(active):] + active[:block % len(active)] if active else []
            for name in order:
                if f"{name}:{mode}" in failures:
                    continue
                counter = {"calls": 0}
                try:
                    rounds = rounds_by_model[name]
                    duration = prediction_block(estimators[name], inputs, rounds, counter)
                    records.append({
                        "model": name, "mode": mode, "stage": "measured", "block": block + 1,
                        "rounds": rounds, "calls_per_round": len(inputs),
                        "prediction_calls": counter["calls"], "seconds": duration,
                        "seconds_per_call": duration / counter["calls"],
                    })
                except Exception as exc:
                    failures[f"{name}:{mode}"] = {
                        "phase": "measured", "block": block + 1, "type": type(exc).__name__,
                        "message": str(exc), "actual_attempted_calls": counter["calls"],
                    }
    return pd.DataFrame(records), failures


def _interval(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"median": np.median(values), "q25": np.quantile(values, .25), "q75": np.quantile(values, .75)} if len(values) else {
        "median": np.nan, "q25": np.nan, "q75": np.nan,
    }


def benchmark_selected(development: pd.DataFrame, selection_path: str | Path,
                       protocol: dict, directory: str | Path, *, provenance: dict) -> dict:
    """Benchmark frozen family choices with no search or replacement fits.

    Parameters
    ----------
    development : pandas.DataFrame
        Complete development data only, from the frozen split loader.
    selection_path : str or pathlib.Path
        Already frozen family choices and accuracy leader.
    protocol : dict
        Approved protocol containing repeat counts and 265-fit total budget.
    directory : str or pathlib.Path
        New benchmark directory, never overwriting search or prior results.
    provenance : dict
        Current dataset and frozen-bundle checksums.

    Returns
    -------
    dict
        Fit budget used and failures. Detailed records and medians/IQR are saved.
        No hyperparameters or engineering recommendation are selected here.
    """
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(destination)
    selection_path = Path(selection_path)
    selection_sha = file_sha256(selection_path)
    frozen = json.loads(selection_path.read_text(encoding="utf-8"))
    if frozen["kind"] != "frozen_development_selection" or frozen["final_test_evaluated"]:
        raise ValueError("Require a development-only frozen selection.")
    if protocol_signature(protocol) != frozen["protocol_sha256"]:
        raise ValueError("Benchmark protocol differs from the frozen search protocol.")
    for key in ("source_sha256", "frozen_split_metadata"):
        if provenance[key] != frozen["provenance"][key]:
            raise ValueError(f"Benchmark provenance differs from selection: {key}.")
    if development_signature(development) != frozen["development_sha256"]:
        raise ValueError("Benchmark development rows differ from the frozen selection.")
    if (len(development) != protocol["expected_development_rows"]
            or development["condition_id"].nunique() != protocol["expected_development_conditions"]):
        raise ValueError("Benchmark development coverage differs from protocol.")
    settings = protocol["benchmark"]
    if settings["warmup_fits"] != 1 or settings["measured_fits"] != 5:
        raise ValueError("Approved benchmark requires one warmup and five measured fits.")
    if protocol["cv_fit_budget"] + len(frozen["selected"]) * 6 > protocol["total_fit_budget"]:
        raise ValueError("Benchmark would exceed the approved total fit budget.")
    # Preflight reconstruction before creating output or calling fit.
    for chosen in frozen["selected"].values():
        model_from_configuration(chosen["configuration"])
    x, y, batch, singles, identifiers = prepared_inputs(
        development, settings["batch_rows"], settings["single_rows"],
    )
    with threadpool_limits(limits=1):
        check_threads()
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination / "selected_configurations.json", frozen)
        write_json(destination / "protocol.json", protocol)
        write_json(destination / "environment.json", environment_details())
        write_json(destination / "provenance.json", provenance)
        write_json(destination / "prepared_inputs.json", identifiers)
        start = perf_counter()
        names = list(frozen["selected"])
        fit_records, estimators = [], {}
        for repeat in range(6):
            # Warmup has fixed order; measured rounds rotate through all families.
            offset = max(0, repeat - 1) % len(names) if names else 0
            order = names[offset:] + names[:offset]
            for name in order:
                estimator = model_from_configuration(frozen["selected"][name]["configuration"])
                record = {"model": name, "stage": "warmup" if repeat == 0 else "measured",
                          "repeat": repeat, "status": "failed", "warnings": [], "error": None}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit_start = perf_counter()
                    try:
                        estimator.fit(x, y)
                        record["seconds"] = perf_counter() - fit_start
                        record["fitted"] = _fitted_diagnostics(estimator)
                        record["status"] = "success"
                        if repeat > 0:
                            estimators[name] = estimator
                    except Exception as exc:
                        record["seconds"] = perf_counter() - fit_start
                        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    record["warnings"] = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
                    record["convergence_warning"] = any(issubclass(item.category, ConvergenceWarning) for item in caught)
                fit_records.append(record)
                write_json(destination / f"fit-{len(fit_records):02d}.json", record)
                print(f'Benchmark {repeat}/5 {name}: {record["status"]}, {record["seconds"]:.4f}s', flush=True)
        # Retain each call's warnings without emitting them into timing output.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predictions, failures = prediction_timings(estimators, batch, singles, settings)
        write_table(destination / "prediction_repeats.csv", predictions)
        summaries = []
        for name in names:
            measured = [item for item in fit_records if item["model"] == name and item["stage"] == "measured" and item["status"] == "success"]
            summary = {
                "model": name, "measured_fits_succeeded": len(measured),
                "measured_fit_convergence_warnings": sum(item["convergence_warning"] for item in measured),
                "representative_oof_equal_condition_mae_db": frozen["selected"][name]["representative_oof_equal_condition_mae_db"],
                "selection_score_db": frozen["selected"][name]["selection_score_db"],
                "serialized_model_bytes": len(pickle.dumps(estimators[name], protocol=5)) if name in estimators else None,
            }
            summary.update({f"fit_seconds_{key}": value for key, value in _interval([item["seconds"] for item in measured]).items()})
            for mode in ("batch", "single"):
                subset = predictions[(predictions["model"] == name) & (predictions["mode"] == mode) & (predictions["stage"] == "measured")] if not predictions.empty else pd.DataFrame()
                summary[f"{mode}_blocks"] = len(subset)
                summary.update({f"{mode}_seconds_{key}": value for key, value in _interval(
                    subset["seconds_per_call"] if not subset.empty else [],
                ).items()})
            summary["complete"] = len(measured) == 5 and all(
                summary[f"{mode}_blocks"] == settings["blocks"] for mode in ("batch", "single")
            )
            summaries.append(summary)
        write_table(destination / "summary.csv", pd.DataFrame(summaries))
        manifest = {
            "kind": "repeated_development_benchmark", "finished_utc": datetime.now(timezone.utc).isoformat(),
            "selection_sha256": selection_sha, "fits_attempted": len(fit_records),
            "fits_failed": sum(item["status"] != "success" for item in fit_records),
            "prediction_failures": failures,
            "prediction_warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught],
            "actual_prediction_calls_completed_blocks": int(predictions["prediction_calls"].sum()) if not predictions.empty else 0,
            "prediction_call_note": "Completed blocks include warmup/calibration/measured. Failed-block attempted counts are additional in prediction_failures.",
            "single_loop": protocol["benchmark"]["single_loop"],
            "latency_statistic": "Median and IQR of block mean seconds per separately dispatched predict call.",
            "fit_order": "One warmup, then five measured rounds rotating family order.",
            "prediction_order": "Per mode calibrate each model, then rotate model order across measured blocks.",
            "serialization": settings["size"], "wall_seconds": perf_counter() - start,
            "timing_scope": "Fit/predict timers include preprocessing and exclude preparation, diagnostics, serialization and writes.",
            "repeats_are_seed_sensitivity": False, "final_test_evaluated": False,
        }
        write_json(destination / "benchmark.json", manifest)
        if file_sha256(selection_path) != selection_sha:
            raise RuntimeError("Frozen configurations changed during benchmarking.")
    return manifest
