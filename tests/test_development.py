"""Bounded grids, complete-seed selection and configuration failure isolation."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airfoil_noise import development
from airfoil_noise.development import expand_protocol, run_study, select_configurations, variant_score
from airfoil_noise.models import model_configuration, model_from_configuration
from test_validation import observations


@pytest.fixture
def small_protocol(observations, monkeypatch):
    """All five families, tiny synthetic grids, explicitly distinct from full study."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    protocol = json.loads(Path("configs/milestone4.json").read_text())
    for family in protocol["families"]:
        family["grid"] = {key: values[:1] for key, values in family["grid"].items()}
        if family["model"] == "mlp":
            family["fixed"]["max_iter"] = 2
    protocol.update(run_kind="synthetic_verification", expected_development_rows=len(observations[0]),
                    expected_development_conditions=11, configuration_count=5,
                    variant_count=7, cv_fit_budget=35, total_fit_budget=65)
    protocol["benchmark"].update(batch_rows=16, single_rows=4, blocks=3,
                                 minimum_block_seconds=0, prediction_warmup_rounds=1)
    return protocol


def provenance():
    return {"source_sha256": "synthetic", "frozen_split_metadata": {"seed": "synthetic saved folds"}}


def test_full_approved_grid_has_exact_budget_defaults_and_roundtrip():
    protocol = json.loads(Path("configs/milestone4.json").read_text())
    variants = expand_protocol(protocol)
    assert len(variants) == 47
    assert pd.Series([item["model"] for item in variants]).value_counts().to_dict() == {
        "training_mean": 1, "linear": 1, "tree": 9, "svr": 18, "mlp": 18,
    }
    assert len({item["configuration_id"] for item in variants}) == 35
    assert 47 * 5 + 5 * 6 == protocol["total_fit_budget"] == 265
    for item in variants:
        encoded = json.loads(json.dumps(item["configuration"], allow_nan=False))
        assert model_configuration(model_from_configuration(encoded)) == encoded
        estimator = model_from_configuration(encoded)
        if item["model"] == "mlp":
            assert estimator[-1].max_iter == 2000
            assert estimator[-1].early_stopping is False
            assert estimator[-1].warm_start is False
    assert {item["seed"] for item in variants if item["model"] == "mlp"} == {17, 29, 43}
    assert {item["seed"] for item in variants if item["model"] == "tree"} == {17}


@pytest.mark.parametrize("fault", ["budget", "fixed", "seed", "grid", "overlap"])
def test_protocol_faults_rejected_before_fits(small_protocol, fault):
    protocol = deepcopy(small_protocol)
    if fault == "budget":
        protocol["total_fit_budget"] = 266
    elif fault == "fixed":
        del protocol["families"][0]["fixed"]["strategy"]
    elif fault == "seed":
        protocol["families"][0]["seeds"] = [17]
    elif fault == "grid":
        protocol["families"][2]["grid"]["max_depth"] = []
    else:
        protocol["families"][2]["fixed"]["max_depth"] = 3
    with pytest.raises(ValueError):
        expand_protocol(protocol)


def test_small_study_saves_all_variants_and_freezes_before_benchmark(observations, small_protocol, tmp_path):
    output = tmp_path / "study"
    frozen = run_study(*observations, small_protocol, output, provenance=provenance())
    assert set(frozen["selected"]) == {"training_mean", "linear", "tree", "svr", "mlp"}
    assert frozen["accuracy_leader"] == "linear"  # Exact constructed linear truth.
    assert frozen["selected"]["mlp"]["representative_seed"] == 17
    assert frozen["converged_only_mlp_selection"] is None  # All synthetic two-epoch fits warn.
    assert frozen["selected"]["mlp"]["warning_fraction"] == 1
    assert frozen["final_test_evaluated"] is False
    search = json.loads((output / "search.json").read_text())
    assert search["fits_attempted"] == 35
    assert search["fits_failed"] == 0
    assert search["kind"] == "synthetic_verification"
    for run in (output / "runs").iterdir():
        predictions = pd.read_csv(run / "predictions.csv")
        training = pd.read_csv(run / "training_predictions.csv")
        assert len(predictions) == len(observations[0])
        assert len(training) == 4 * len(predictions)
        assert (training["fit_fold"] != training["fold"]).all()
    with pytest.raises(FileExistsError):
        run_study(*observations, small_protocol, output, provenance=provenance())


def test_one_failed_variant_preserves_success_and_excludes_configuration(observations, small_protocol, tmp_path, monkeypatch):
    from sklearn.svm import SVR
    def fail(*args, **kwargs):
        raise RuntimeError("Deliberate synthetic solver failure")
    monkeypatch.setattr(SVR, "fit", fail)
    output = tmp_path / "failure"
    frozen = run_study(*observations, small_protocol, output, provenance=provenance())
    assert "svr" not in frozen["selected"]
    assert len(frozen["selected"]) == 4
    run = output / "runs" / "svr_01"
    assert (run / "diagnostics.json").exists() and not (run / "summary.csv").exists()
    diagnostics = json.loads((run / "diagnostics.json").read_text())
    assert len(diagnostics) == 5
    assert all("Deliberate" in item["error"]["message"] for item in diagnostics)
    scores = pd.read_csv(output / "variant_scores.csv")
    assert scores.loc[scores.model == "svr", "eligible"].tolist() == [False]
    assert json.loads((output / "search.json").read_text())["fits_attempted"] == 35
    assert json.loads((output / "search.json").read_text())["fits_failed"] == 5


def _selection_fixture():
    protocol = json.loads(Path("configs/milestone4.json").read_text())
    variants = [item for item in expand_protocol(protocol) if item["model"] == "mlp" and item["configuration_id"] in ("mlp_01", "mlp_02")]
    records = []
    for item, score in zip(variants, [1., 5., 6., 3., 3., 3.]):
        records.append({**{key: item[key] for key in ("variant_id", "configuration_id", "model", "seed", "grid_order")},
                        "eligible": True, "convergence_warning_fits": 0, "fits_attempted": 5,
                        "equal_condition_mae_db": score, "mean_structural_count": 71})
    return protocol, variants, pd.DataFrame(records)


def test_mlp_selection_averages_complete_seed_scores_never_best_seed():
    protocol, variants, scores = _selection_fixture()
    configurations, frozen = select_configurations(scores, variants, protocol)
    assert frozen["selected"]["mlp"]["configuration_id"] == "mlp_02"
    first = configurations.set_index("configuration_id").loc["mlp_01"]
    assert first["selection_score_db"] == 4
    assert first["seed_score_sd_db"] == pytest.approx(np.std([1, 5, 6], ddof=1))
    # Losing seed 43 makes the whole second configuration ineligible.
    scores.loc[scores["variant_id"] == "mlp_02_seed43", "eligible"] = False
    _, frozen = select_configurations(scores, variants, protocol)
    assert frozen["selected"]["mlp"]["configuration_id"] == "mlp_01"


@pytest.mark.parametrize("warning_position", range(6))
def test_converged_only_checks_all_seeds_and_every_fold(warning_position):
    protocol, variants, scores = _selection_fixture()
    scores.loc[warning_position, "convergence_warning_fits"] = 1
    configs, frozen = select_configurations(scores, variants, protocol)
    invalid_id = scores.loc[warning_position, "configuration_id"]
    assert not configs.set_index("configuration_id").loc[invalid_id, "converged_only_eligible"]
    assert frozen["converged_only_mlp_selection"]["configuration_id"] != invalid_id
    scores["convergence_warning_fits"] = 1
    _, frozen = select_configurations(scores, variants, protocol)
    assert frozen["converged_only_mlp_selection"] is None
    assert frozen["selected"]["mlp"]["eligible"]  # Warnings do not invalidate primary results.


def test_declared_numerical_ties_use_warnings_then_complexity_then_order():
    protocol, variants, scores = _selection_fixture()
    scores["equal_condition_mae_db"] = 3.
    scores.loc[scores.configuration_id == "mlp_01", "convergence_warning_fits"] = 1
    _, frozen = select_configurations(scores, variants, protocol)
    assert frozen["selected"]["mlp"]["configuration_id"] == "mlp_02"
    scores["convergence_warning_fits"] = 0
    scores.loc[scores.configuration_id == "mlp_02", "mean_structural_count"] = 70
    _, frozen = select_configurations(scores, variants, protocol)
    assert frozen["selected"]["mlp"]["configuration_id"] == "mlp_02"
    scores["mean_structural_count"] = 71
    _, frozen = select_configurations(scores, variants, protocol)
    assert frozen["selected"]["mlp"]["configuration_id"] == "mlp_01"


@pytest.mark.parametrize("fault", ["coverage", "threads"])
def test_global_preflight_failure_creates_no_results(observations, small_protocol, tmp_path, monkeypatch, fault):
    output = tmp_path / "bad"
    if fault == "coverage":
        small_protocol["expected_development_rows"] += 1
    else:
        monkeypatch.delenv("OMP_NUM_THREADS")
    with pytest.raises(ValueError):
        run_study(*observations, small_protocol, output, provenance=provenance())
    assert not output.exists()


def test_nonfinite_loss_diagnostics_ineligible_even_with_finite_predictions(observations, small_protocol):
    variant = next(item for item in expand_protocol(small_protocol) if item["model"] == "mlp")
    result = development.evaluate_development(*observations, configurations={variant["variant_id"]: variant["specification"]},
                                               record_training=True)
    result["diagnostics"][0]["fitted"]["steps"]["regressor"]["loss_curve_"][0] = None
    assert not variant_score(result, variant, 5)["eligible"]


def test_withheld_target_changes_cannot_change_study_scores_or_selection(observations, small_protocol, tmp_path):
    from airfoil_noise.data import COLUMNS, load_airfoil
    from airfoil_noise.split import write_frozen_splits, load_development, read_frozen_splits
    source = tmp_path / "source.dat"
    observations[0][list(COLUMNS)].to_csv(source, sep="\t", index=False, header=False)
    all_data = load_airfoil(source)
    splits = tmp_path / "splits"
    write_frozen_splits(all_data, splits)
    dev, folds = load_development(all_data, splits)
    small_protocol.update(expected_development_rows=len(dev),
                          expected_development_conditions=dev["condition_id"].nunique())
    first = run_study(dev, folds, small_protocol, tmp_path / "first", provenance=provenance())
    manifest, _, _ = read_frozen_splits(all_data, splits)
    withheld = manifest.loc[manifest["split"] == "test", "condition_id"]
    changed = all_data.copy()
    changed.loc[changed["condition_id"].isin(withheld), "spl_db"] = np.nan
    alternate_dev, alternate_folds = load_development(changed, splits)
    second = run_study(alternate_dev, alternate_folds, small_protocol, tmp_path / "second", provenance=provenance())
    assert first["selected"] == second["selected"]
    assert first["accuracy_leader"] == second["accuracy_leader"]
    for filename in ("variant_scores.csv", "configuration_scores.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "first" / filename),
                                      pd.read_csv(tmp_path / "second" / filename))
