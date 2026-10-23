"""Bounded grouped search, failure isolation and pre-benchmark selection."""

from datetime import datetime, timezone
from itertools import product
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from airfoil_noise.data import FEATURE_COLUMNS, TARGET_COLUMN, file_sha256
from airfoil_noise.models import build_model, model_configuration
from airfoil_noise.validation import (
    _json_ready, _validated_inputs, environment_details, evaluate_development,
    save_development_results,
)


def write_json(path: Path, record: dict | list) -> None:
    """Write strict JSON to a new file, rejecting accidental replacement."""
    content = json.dumps(_json_ready(record), indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Write a new UTF-8/LF table with enough digits to trace predictions."""
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        table.to_csv(stream, index=False, float_format="%.17g", lineterminator="\n")


def development_signature(development: pd.DataFrame) -> str:
    """Bind selections to the exact development IDs, original inputs and truth."""
    columns = ["row_id", "condition_id", *FEATURE_COLUMNS, TARGET_COLUMN]
    content = development.sort_values("row_id")[columns].to_csv(
        index=False, float_format="%.17g", lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def protocol_signature(protocol: dict) -> str:
    """Hash the same canonical JSON bytes saved with each protected run."""
    content = json.dumps(_json_ready(protocol), indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def check_threads() -> list[dict]:
    """Require the declared process environment and verify loaded pools."""
    variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    if any(os.environ.get(key) != "1" for key in variables):
        raise ValueError("Set all four documented thread variables to 1 before Python starts.")
    pools = threadpool_info()
    if any(pool["num_threads"] != 1 for pool in pools):
        raise ValueError("Numerical thread pools must be limited to one thread.")
    return pools


def expand_protocol(protocol: dict) -> list[dict]:
    """Expand declared Cartesian grids and preflight complete model settings.

    Parameters
    ----------
    protocol : dict
        Machine-readable protocol including fixed settings, grids, seeds,
        expected counts and budgets. Synthetic checks use a distinct run_kind.

    Returns
    -------
    list of dict
        Ordered variants with hyperparameter ID, seed, specification and full
        reconstructable estimator configuration. No data or fitting is used.
    """
    variants, configuration_count = [], 0
    families = protocol["families"]
    if not families or len({family["model"] for family in families}) != len(families):
        raise ValueError("Specify distinct model families.")
    for family in families:
        grid, fixed = family["grid"], family["fixed"]
        if set(grid) & set(fixed) or "random_state" in fixed:
            raise ValueError("Grid and fixed settings must be disjoint; seeds are separate.")
        if not family["seeds"] or len(set(family["seeds"])) != len(family["seeds"]):
            raise ValueError("Seeds must be nonempty and distinct.")
        if any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError("Every grid dimension needs an explicit nonempty list.")
        for index, values in enumerate(product(*grid.values()), start=1):
            configuration_count += 1
            parameters = {**fixed, **dict(zip(grid, values))}
            if "hidden_layer_sizes" in parameters:
                parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
            config_id = f'{family["model"]}_{index:02d}'
            for seed in family["seeds"]:
                specification = {"model": family["model"], "parameters": parameters, "seed": seed}
                estimator = build_model(**{"name": specification["model"],
                                           "parameters": parameters, "seed": seed})
                regressor = estimator[-1] if hasattr(estimator, "steps") else estimator
                expected_keys = set(regressor.get_params(deep=False)) - {"random_state"}
                if set(parameters) != expected_keys:
                    raise ValueError(f"Every non-seed constructor setting must be explicit: {config_id}.")
                if hasattr(estimator, "steps"):
                    if estimator["scaler"].get_params(deep=False) != protocol["scaler"]:
                        raise ValueError("Scaler settings differ from the protocol.")
                    pipeline = {k: v for k, v in estimator.get_params(deep=False).items() if k != "steps"}
                    if pipeline != protocol["pipeline"]:
                        raise ValueError("Pipeline settings differ from the protocol.")
                variants.append({
                    "variant_id": config_id + (f"_seed{seed}" if seed is not None else ""),
                    "configuration_id": config_id, "model": family["model"], "seed": seed,
                    "grid_order": configuration_count, "specification": specification,
                    "configuration": model_configuration(estimator),
                })
    expected_fits = len(variants) * protocol["n_folds"]
    benchmark_fits = len(families) * (
        protocol["benchmark"]["warmup_fits"] + protocol["benchmark"]["measured_fits"]
    )
    if (configuration_count != protocol["configuration_count"]
            or len(variants) != protocol["variant_count"]
            or expected_fits != protocol["cv_fit_budget"]
            or expected_fits + benchmark_fits != protocol["total_fit_budget"]
            or protocol["total_fit_budget"] > 265):
        raise ValueError("Protocol counts do not match the declared bounded fit budget.")
    if protocol["selection"]["representative_mlp_seed"] not in (17,):
        raise ValueError("The predefined representative MLP seed is 17.")
    return variants


def _regressor_details(record: dict) -> dict:
    fitted = record["fitted"]
    return fitted.get("steps", {}).get("regressor", fitted)


def variant_score(result: dict, variant: dict, n_folds: int) -> dict:
    """Check complete variant eligibility and extract OOF and complexity scores."""
    diagnostics = result["diagnostics"]
    reasons = []
    if (result["status"] != "success" or len(diagnostics) != n_folds
            or any(item["status"] != "success" for item in diagnostics)):
        reasons.append("incomplete or failed fold fits")
    if variant["model"] == "mlp":
        for item in diagnostics:
            fitted = _regressor_details(item)
            losses = fitted.get("loss_curve_", [])
            if (not losses or fitted.get("loss_") is None
                    or not np.isfinite(np.asarray(losses, dtype=float)).all()
                    or not np.isfinite(float(fitted.get("loss_", np.nan)))
                    or len(losses) != fitted.get("n_iter_")):
                reasons.append("missing/nonfinite MLP loss diagnostics")
                break
    eligible = not reasons
    score = {
        **{key: variant[key] for key in ("variant_id", "configuration_id", "model", "seed", "grid_order")},
        "eligible": eligible, "ineligibility_reason": "; ".join(reasons),
        "convergence_warning_fits": sum(item["convergence_warning"] for item in diagnostics),
        "fits_attempted": len(diagnostics),
        "fits_failed": sum(item["status"] != "success" for item in diagnostics),
    }
    for metric in ("equal_condition_mae_db", "mae_db", "rmse_db", "r2"):
        score[metric] = result["summary"].iloc[0][metric] if eligible else np.nan
    score["training_equal_condition_mae_db"] = (
        result["training_summary"].iloc[0]["equal_condition_mae_db"] if eligible else np.nan
    )
    structural_key = {"tree": "node_count", "svr": "support_vector_count", "mlp": "parameter_count"}.get(variant["model"])
    score["mean_structural_count"] = (
        np.mean([_regressor_details(item)[structural_key] for item in diagnostics])
        if structural_key and eligible else 0
    )
    score["mean_iterations"] = (
        np.mean([_regressor_details(item)["n_iter_"] for item in diagnostics])
        if variant["model"] == "mlp" and eligible else np.nan
    )
    return score


def _choose(rows: list[dict], tolerance: float, *, cross_family: bool, family_order: list[str]) -> dict:
    minimum = min(row["selection_score_db"] for row in rows)
    tied = [row for row in rows if row["selection_score_db"] <= minimum + tolerance]
    return min(tied, key=lambda row: (
        row["warning_fraction"],
        family_order.index(row["model"]) if cross_family else row["mean_structural_count"],
        row["grid_order"],
    ))


def select_configurations(scores: pd.DataFrame, variants: list[dict], protocol: dict) -> tuple[pd.DataFrame, dict]:
    """Freeze eligible family winners and accuracy leader without timing input.

    All seeds must complete. MLP selection averages seed-level overall OOF
    scores, never predictions. Converged-only eligibility checks every fit,
    not just a representative seed. Numerical ties follow the declared order.
    """
    configurations = []
    for config_id, group in scores.groupby("configuration_id", sort=False):
        expected = [item for item in variants if item["configuration_id"] == config_id]
        eligible = len(group) == len(expected) and bool(group["eligible"].all())
        complete_seeds = set(group["variant_id"]) == {item["variant_id"] for item in expected}
        eligible = eligible and complete_seeds
        warning_fits = int(group["convergence_warning_fits"].sum())
        configurations.append({
            "configuration_id": config_id, "model": expected[0]["model"],
            "grid_order": expected[0]["grid_order"], "eligible": eligible,
            "selection_score_db": group["equal_condition_mae_db"].mean() if eligible else np.nan,
            "seed_score_sd_db": group["equal_condition_mae_db"].std(ddof=1) if eligible and len(group) > 1 else 0.0,
            "seed_score_min_db": group["equal_condition_mae_db"].min() if eligible else np.nan,
            "seed_score_max_db": group["equal_condition_mae_db"].max() if eligible else np.nan,
            "warning_fits": warning_fits,
            "warning_fraction": warning_fits / (len(expected) * protocol["n_folds"]),
            "converged_only_eligible": eligible and warning_fits == 0,
            "mean_structural_count": group["mean_structural_count"].mean() if eligible else np.nan,
        })
    tolerance = protocol["selection"]["numerical_tie_db"]
    order = protocol["selection"]["family_tie_order"]
    selected = {}
    for family in protocol["families"]:
        eligible = [row for row in configurations if row["model"] == family["model"] and row["eligible"]]
        if not eligible:
            continue
        winner = _choose(eligible, tolerance, cross_family=False, family_order=order)
        candidates = [item for item in variants if item["configuration_id"] == winner["configuration_id"]]
        representative = (next(item for item in candidates if item["seed"] == 17)
                          if family["model"] == "mlp" else candidates[0])
        representative_score = scores.set_index("variant_id").loc[representative["variant_id"]]
        selected[family["model"]] = {
            **winner, "representative_variant_id": representative["variant_id"],
            "representative_seed": representative["seed"],
            "representative_oof_equal_condition_mae_db": representative_score["equal_condition_mae_db"],
            "configuration": representative["configuration"],
        }
    leader = (_choose(list(selected.values()), tolerance, cross_family=True, family_order=order)["model"]
              if selected else None)
    converged_mlp = [row for row in configurations if row["model"] == "mlp" and row["converged_only_eligible"]]
    sensitivity = (_choose(converged_mlp, tolerance, cross_family=False, family_order=order)
                   if converged_mlp else None)
    return pd.DataFrame(configurations), {
        "schema_version": 1, "kind": "frozen_development_selection",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": protocol["selection"], "selected": selected,
        "accuracy_leader": leader, "converged_only_mlp_selection": sensitivity,
        "engineering_recommendation": "Pending separate benchmarks; timings cannot change these configurations.",
        "final_test_evaluated": False,
    }


def run_study(development: pd.DataFrame, row_folds: pd.DataFrame, protocol: dict,
              directory: str | Path, *, provenance: dict) -> dict:
    """Run the bounded search, saving each variant before starting the next.

    Parameters
    ----------
    development, row_folds : pandas.DataFrame
        Development observations and the saved mapping; no final-test data.
    protocol : dict
        Approved explicit settings. Preflight completes before any fitting.
    directory : str or pathlib.Path
        New protected search directory.
    provenance : dict
        Dataset, frozen assignments and implementation checksum records.

    Returns
    -------
    dict
        Frozen family selections, accuracy leader and converged-only sensitivity.
        Fit failures remain in individual runs and make configurations ineligible.
    """
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(destination)
    variants = expand_protocol(protocol)
    _, _, aligned = _validated_inputs(development, row_folds)
    if (len(development) != protocol["expected_development_rows"]
            or development["condition_id"].nunique() != protocol["expected_development_conditions"]
            or aligned["fold"].nunique() != protocol["n_folds"]):
        raise ValueError("Development coverage differs from the protocol.")
    with threadpool_limits(limits=1):
        check_threads()
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination / "protocol.json", protocol)
        write_json(destination / "variants.json", variants)
        write_json(destination / "environment.json", environment_details())
        write_json(destination / "provenance.json", provenance)
        start = perf_counter()
        scores, costs = [], []
        for variant in variants:
            variant_start = perf_counter()
            result = evaluate_development(
                development, row_folds,
                configurations={variant["variant_id"]: variant["specification"]},
                record_training=True,
            )
            score = variant_score(result, variant, protocol["n_folds"])
            save_development_results(result, destination / "runs" / variant["variant_id"],
                                     provenance={**provenance, "search_protocol_sha256": file_sha256(destination / "protocol.json")})
            scores.append(score)
            write_json(destination / f'progress-{len(scores):02d}.json', score)
            costs.append({
                "variant_id": variant["variant_id"], "model": variant["model"],
                "fit_seconds": sum(item["fit_seconds"] or 0 for item in result["diagnostics"]),
                "prediction_seconds": sum(item["prediction_seconds"] or 0 for item in result["diagnostics"]),
                "training_prediction_seconds": sum(item["training_prediction_seconds"] or 0 for item in result["diagnostics"]),
                "evaluation_seconds": result["run"]["evaluation_seconds"],
                "evaluation_and_save_seconds": perf_counter() - variant_start,
                "fits_attempted": result["run"]["fits_attempted"],
                "fits_failed": result["run"]["fits_failed"],
            })
            print(f'{len(scores)}/{len(variants)} {variant["variant_id"]}: '
                  f'eligible={score["eligible"]}, MAE={score["equal_condition_mae_db"]:.6f}, '
                  f'convergence warnings={score["convergence_warning_fits"]}', flush=True)
        table = pd.DataFrame(scores)
        configurations, frozen = select_configurations(table, variants, protocol)
        frozen["provenance"] = provenance
        frozen["development_sha256"] = development_signature(development)
        frozen["protocol_sha256"] = file_sha256(destination / "protocol.json")
        write_table(destination / "variant_scores.csv", table)
        write_table(destination / "configuration_scores.csv", configurations)
        write_table(destination / "search_costs.csv", pd.DataFrame(costs))
        write_json(destination / "selected_configurations.json", frozen)
        write_json(destination / "search.json", {
            "kind": protocol["run_kind"], "wall_seconds": perf_counter() - start,
            "fits_attempted": sum(item["fits_attempted"] for item in costs),
            "fits_failed": sum(item["fits_failed"] for item in costs),
            "cv_fit_budget": protocol["cv_fit_budget"],
            "timing_scope": "Sequential evaluation and per-variant saving plus selection; excludes preflight/loading and plotting.",
            "status": "completed", "final_test_evaluated": False,
        })
    return frozen
