"""Synthetic verification of frozen one-pass inference and saved-record reports."""

from copy import deepcopy
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVR

from airfoil_noise import final_evaluation as final
from airfoil_noise.data import FEATURE_COLUMNS, SOURCE_SHA256, file_sha256
from airfoil_noise.development import development_signature, write_json, write_table
from airfoil_noise.models import build_model, model_configuration
from test_validation import observations


@pytest.fixture
def final_fixture(observations, tmp_path, monkeypatch):
    """Explicitly synthetic choices; never edit the real frozen settings."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(variable, "1")
    data = observations[0]
    development = data[data["angle_deg"] < 8].copy()
    holdout = data[data["angle_deg"] >= 8].copy()
    selection = deepcopy(json.loads(Path("reports/development/milestone-4-search/selected_configurations.json").read_text()))
    for name, item in selection["selected"].items():
        parameters = {"max_iter": 2, "hidden_layer_sizes": (6,)} if name == "mlp" else {}
        item.update(configuration_id=f"synthetic_{name}", representative_variant_id=f"synthetic_{name}",
                    configuration=model_configuration(build_model(name, parameters=parameters)),
                    representative_oof_equal_condition_mae_db=0., selection_score_db=0.)
    selection["provenance"] = {"source_sha256": "synthetic"}
    selection["development_sha256"] = development_signature(development)
    paths = tmp_path / "choices"
    paths.mkdir()
    write_json(paths / "selection.json", selection)
    recommendation = {
        "kind": "frozen_development_engineering_recommendation", "recommended_model": "svr",
        "configuration_id": selection["selected"]["svr"]["configuration_id"],
        "configuration": selection["selected"]["svr"]["configuration"],
        "selection_sha256": file_sha256(paths / "selection.json"), "final_test_evaluated": False,
    }
    write_json(paths / "recommendation.json", recommendation)
    expected = {
        "selection": file_sha256(paths / "selection.json"),
        "recommendation": file_sha256(paths / "recommendation.json"),
        "configurations": {name: final.configuration_hash(item["configuration"]) for name, item in selection["selected"].items()},
    }
    frozen = final.read_frozen_choices(paths / "selection.json", paths / "recommendation.json", expected_hashes=expected)
    return development, holdout, frozen, expected


def no_figure_rendering(monkeypatch):
    """Keep table/record tests fast; genuine rendering is tested in test_plotting."""
    from airfoil_noise import plotting
    def placeholder(run_directory, directory):
        Path(directory).mkdir()
        write_json(Path(directory) / "manifest.json", {"synthetic_test_placeholder": True})
        return []
    monkeypatch.setattr(plotting, "plot_final", placeholder)


def save_fixture(final_fixture, tmp_path, monkeypatch):
    no_figure_rendering(monkeypatch)
    dev, held, frozen, _ = final_fixture
    directory = tmp_path / "saved"
    result = final.run_final_evaluation(dev, held, frozen, directory,
        provenance={"source_sha256": "synthetic", "command": ["synthetic_test"]},
        run_kind="synthetic_verification")
    return directory, result


def rehash(directory, filename):
    """Fixture-only integrity update, to exercise validation below the hash layer."""
    path = directory / "evaluation.json"
    record = json.loads(path.read_text())
    record["record_sha256"][filename] = file_sha256(directory / filename)
    path.write_text(json.dumps(record), encoding="utf-8")


def test_real_configuration_preflight_does_not_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Frozen-choice inspection must not fit.")
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neural_network import MLPRegressor
    for cls in (DummyRegressor, LinearRegression, DecisionTreeRegressor, SVR, MLPRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)
    frozen = final.read_frozen_choices(
        "reports/development/milestone-4-search/selected_configurations.json",
        "reports/development/milestone-4-benchmarks/recommendation.json",
    )
    assert frozen["configuration_sha256"] == final.APPROVED_HASHES["configurations"]
    assert frozen["selection"]["selected"]["mlp"]["representative_seed"] == 17
    assert frozen["selection"]["selected"]["mlp"]["configuration"]["parameters"]["steps"]
    assert frozen["recommendation"]["recommended_model"] == "svr"


@pytest.mark.parametrize("which", ["selection", "recommendation"])
def test_stale_frozen_file_rejected_before_any_fits(final_fixture, which):
    frozen, expected = final_fixture[2:]
    path = Path(frozen["paths"][which])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        final.read_frozen_choices(frozen["paths"]["selection"], frozen["paths"]["recommendation"],
                                 expected_hashes=expected)


def test_changed_in_memory_configuration_rejected_before_fit(final_fixture, monkeypatch):
    dev, held, frozen, _ = final_fixture
    frozen["selection"]["selected"]["linear"]["configuration"]["parameters"]["fit_intercept"] = False
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid settings must not reach a fit.")
    monkeypatch.setattr(SVR, "fit", forbidden)
    with pytest.raises(ValueError, match="changed after preflight"):
        final.fit_predict_frozen(dev, held.drop(columns="spl_db"), frozen)


def test_five_fresh_fits_five_predictions_and_development_only_scaling(final_fixture, monkeypatch):
    dev, held, frozen, _ = final_fixture
    original = final.model_from_configuration
    captured = []
    def reconstruct(record):
        estimator = original(record)
        fit, predict = estimator.fit, estimator.predict
        state = {"estimator": estimator, "fit": 0, "predict": 0}
        def checked_fit(x, y):
            assert not hasattr(estimator, "n_features_in_")
            assert list(x.columns) == list(FEATURE_COLUMNS)
            np.testing.assert_allclose(x, dev.sort_values("row_id")[list(FEATURE_COLUMNS)])
            np.testing.assert_allclose(y, dev.sort_values("row_id")["spl_db"])
            state["fit"] += 1
            return fit(x, y)
        def checked_predict(x):
            assert list(x.columns) == list(FEATURE_COLUMNS)
            assert len(x) == len(held) and state["fit"] == 1
            state["predict"] += 1
            return predict(x)
        estimator.fit, estimator.predict = checked_fit, checked_predict
        captured.append(state)
        return estimator
    monkeypatch.setattr(final, "model_from_configuration", reconstruct)
    result = final.fit_predict_frozen(dev, held.drop(columns="spl_db"), frozen)
    assert result["status"] == "success"
    assert len(captured) == len({id(item["estimator"]) for item in captured}) == 5
    assert all(item["fit"] == item["predict"] == 1 for item in captured)
    assert len(result["predictions"]) == 5 * len(held)
    for state in captured:
        estimator = state["estimator"]
        if hasattr(estimator, "steps"):
            np.testing.assert_allclose(estimator["scaler"].mean_, dev[list(FEATURE_COLUMNS)].mean())
            np.testing.assert_allclose(estimator["scaler"].var_, dev[list(FEATURE_COLUMNS)].var(ddof=0))
            assert estimator["scaler"].n_samples_seen_ == len(dev)
    linear = result["predictions"].query("model == 'linear'").set_index("row_id")
    np.testing.assert_allclose(linear.prediction_db, held.set_index("row_id").loc[linear.index, "spl_db"], atol=1e-9)


def test_truth_and_extreme_test_inputs_cannot_change_training(final_fixture):
    dev, held, frozen, _ = final_fixture
    expected = final.fit_predict_frozen(dev, held.drop(columns="spl_db"), frozen)
    changed = held.copy()
    changed["spl_db"] = np.nan
    alternate = final.fit_predict_frozen(dev, changed.drop(columns="spl_db"), frozen)
    pd.testing.assert_frame_equal(expected["predictions"], alternate["predictions"])
    extreme = held.drop(columns="spl_db").copy()
    extreme["frequency_hz"] *= 1000
    different_inputs = final.fit_predict_frozen(dev, extreme, frozen)
    assert [row["fitted"] for row in expected["diagnostics"]] == [row["fitted"] for row in different_inputs["diagnostics"]]
    with pytest.raises(ValueError, match="never holdout targets"):
        final.fit_predict_frozen(dev, held, frozen)


@pytest.mark.parametrize("fault", ["overlap", "duplicate", "condition", "nonfinite", "missing"])
def test_invalid_inference_inputs_rejected(final_fixture, fault):
    dev, held, frozen, _ = final_fixture
    inputs = held.drop(columns="spl_db").copy()
    if fault == "overlap":
        inputs = dev.drop(columns="spl_db").copy()
    elif fault == "duplicate":
        inputs.iloc[1, inputs.columns.get_loc("row_id")] = inputs.iloc[0]["row_id"]
    elif fault == "condition":
        inputs["condition_id"] = "wrong condition"
    elif fault == "nonfinite":
        inputs["frequency_hz"] = np.inf
    else:
        inputs = inputs.drop(columns="velocity_m_s")
    with pytest.raises(ValueError):
        final.fit_predict_frozen(dev, inputs, frozen)


def test_saved_end_to_end_complete_and_aligned_with_shuffled_inputs(final_fixture, tmp_path, monkeypatch):
    dev, held, frozen, _ = final_fixture
    no_figure_rendering(monkeypatch)
    dev = dev.sample(frac=1, random_state=12)
    held = held.sample(frac=1, random_state=13)
    held.index = [0] * len(held)
    directory = tmp_path / "workflow"
    result = final.run_final_evaluation(dev, held, frozen, directory,
        provenance={"source_sha256": "synthetic"}, run_kind="synthetic_verification")
    assert result["status"] == "success_warning_qualified"
    assert result["fits_attempted"] == result["prediction_calls"] == 5
    records = final.load_final_records(directory)
    assert len(records["predictions"]) == 5 * len(held)
    assert not records["predictions"].duplicated(["model", "row_id"]).any()
    assert len(records["condition_metrics"]) == 5 * held.condition_id.nunique()
    assert list(records["summary"].model) == list(final.MODEL_ORDER)
    assert records["summary"].loc[records["summary"].primary, "model"].tolist() == ["svr"]
    for _, group in records["predictions"].groupby("model"):
        np.testing.assert_allclose(group.truth_db, held.set_index("row_id").loc[group.row_id, "spl_db"])
    assert (directory / "completion.json").exists()
    assert json.loads((directory / "selection.json").read_text()) == frozen["selection"]
    assert json.loads((directory / "run.json").read_text())["kind"] == "synthetic_verification"
    with pytest.raises(FileExistsError):
        final.run_final_evaluation(dev, held, frozen, directory, provenance={}, run_kind="synthetic_verification")


@pytest.mark.parametrize("failure", ["fit", "predict", "nonfinite", "shape"])
def test_local_failure_preserves_partial_records_without_success_summary(final_fixture, tmp_path, monkeypatch, failure):
    dev, held, frozen, _ = final_fixture
    if failure == "fit":
        def fail(*args, **kwargs): raise RuntimeError("deliberate fit failure")
        monkeypatch.setattr(SVR, "fit", fail)
    else:
        def predict(self, x):
            if failure == "predict": raise RuntimeError("deliberate prediction failure")
            return np.full(len(x), np.nan) if failure == "nonfinite" else np.zeros((len(x), 1))
        monkeypatch.setattr(SVR, "predict", predict)
    output = tmp_path / "failed"
    result = final.run_final_evaluation(dev, held, frozen, output,
        provenance={"source_sha256": "synthetic"}, run_kind="synthetic_verification")
    assert result["status"] == "failed" and result["fits_attempted"] == 5 and result["models_failed"] == 1
    assert result["fits_failed"] == (1 if failure == "fit" else 0)
    assert result["prediction_calls"] == (4 if failure == "fit" else 5)
    assert len(pd.read_csv(output / "predictions.csv")) == 4 * len(held)
    assert not (output / "completion.json").exists() and not (output / "report").exists()
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    assert next(row for row in diagnostics if row["model"] == "svr")["error"]
    with pytest.raises(ValueError, match="failed/incomplete"):
        final.regenerate_final_report(output, tmp_path / "bad-report")
    assert not (tmp_path / "bad-report").exists()


def test_convergence_warning_and_loss_diagnostics_retained(final_fixture, tmp_path, monkeypatch):
    directory, result = save_fixture(final_fixture, tmp_path, monkeypatch)
    records = final.load_final_records(directory)
    mlp = next(row for row in records["diagnostics"] if row["model"] == "mlp")
    assert mlp["convergence_warning"]
    assert any(item["category"] == "ConvergenceWarning" for item in mlp["warnings"])
    assert mlp["fitted"]["steps"]["regressor"]["n_iter_"] == 2
    assert len(mlp["fitted"]["steps"]["regressor"]["loss_curve_"]) == 2
    assert result["status"] == "success_warning_qualified"


def test_bad_mlp_loss_diagnostics_fail_without_retries(final_fixture, tmp_path, monkeypatch):
    original = final._fitted_diagnostics
    def bad_diagnostics(estimator):
        result = original(estimator)
        if hasattr(estimator, "steps") and hasattr(estimator[-1], "loss_curve_"):
            result["steps"]["regressor"]["loss_curve_"][0] = None
        return result
    monkeypatch.setattr(final, "_fitted_diagnostics", bad_diagnostics)
    dev, held, frozen, _ = final_fixture
    result = final.run_final_evaluation(dev, held, frozen, tmp_path / "bad-loss",
        provenance={"source_sha256": "synthetic"}, run_kind="synthetic_verification")
    assert result["status"] == "failed" and result["fits_attempted"] == 5


def test_saved_metrics_match_hand_calculation_with_unequal_conditions(final_fixture, tmp_path, monkeypatch):
    directory, _ = save_fixture(final_fixture, tmp_path, monkeypatch)
    table = pd.read_csv(directory / "predictions.csv")
    # Conditions at angles 8,9,10 have sizes 4,2,3; errors 1,3,5.
    condition_by_angle = final_fixture[1].groupby("angle_deg").condition_id.first()
    error_by_condition = {condition_by_angle.loc[angle]: error for angle, error in [(8, 1.), (9, 3.), (10, 5.)]}
    table["prediction_db"] = table["truth_db"] - table.condition_id.map(error_by_condition)
    table.to_csv(directory / "predictions.csv", index=False, float_format="%.17g")
    rehash(directory, "predictions.csv")
    summary = final.load_final_records(directory)["summary"]
    assert summary.equal_condition_mae_db.to_list() == pytest.approx([3.] * 5)
    assert summary.mae_db.to_list() == pytest.approx([25 / 9] * 5)
    assert summary.rmse_db.to_list() == pytest.approx([np.sqrt(97 / 9)] * 5)


def test_constant_truth_r2_remains_undefined_in_regenerated_csv(final_fixture, tmp_path, monkeypatch):
    directory, _ = save_fixture(final_fixture, tmp_path, monkeypatch)
    table = pd.read_csv(directory / "predictions.csv")
    table["truth_db"] = table["prediction_db"] = 42.
    table.to_csv(directory / "predictions.csv", index=False)
    rehash(directory, "predictions.csv")
    destination = tmp_path / "constant"
    final.regenerate_final_report(directory, destination)
    summary = pd.read_csv(destination / "summary.csv")
    assert (summary.mae_db == 0).all() and (summary.rmse_db == 0).all()
    assert summary.r2.isna().all()
    assert summary.r2_undefined_reason.eq("constant_truth").all()


@pytest.mark.parametrize("fault", ["checksum", "missing", "duplicate", "truth", "condition"])
def test_saved_record_corruption_rejected(final_fixture, tmp_path, monkeypatch, fault):
    directory, _ = save_fixture(final_fixture, tmp_path, monkeypatch)
    table = pd.read_csv(directory / "predictions.csv")
    if fault == "missing": table = table.iloc[:-1]
    elif fault == "duplicate": table.iloc[1] = table.iloc[0]
    elif fault == "truth": table.loc[0, "truth_db"] += 1
    elif fault == "condition": table.loc[0, "condition_id"] = "wrong"
    else: table.loc[0, "prediction_db"] += 1
    table.to_csv(directory / "predictions.csv", index=False, float_format="%.17g")
    if fault != "checksum": rehash(directory, "predictions.csv")
    with pytest.raises(ValueError):
        final.regenerate_final_report(directory, tmp_path / "corrupted")
    assert not (tmp_path / "corrupted").exists()


def test_reporting_uses_only_saved_records_without_construct_fit_or_predict(final_fixture, tmp_path, monkeypatch):
    directory, _ = save_fixture(final_fixture, tmp_path, monkeypatch)
    original = (directory / "predictions.csv").read_bytes()
    def forbidden(*args, **kwargs): pytest.fail("Reporting must not construct or invoke estimators.")
    monkeypatch.setattr(final, "model_from_configuration", forbidden)
    for name in ("selection", "recommendation"):
        Path(final_fixture[2]["paths"][name]).unlink()  # Original input paths are not needed.
    result = final.regenerate_final_report(directory, tmp_path / "regenerated")
    assert result["fits_attempted"] == result["prediction_calls"] == 0
    assert (directory / "predictions.csv").read_bytes() == original
    pd.testing.assert_frame_equal(pd.read_csv(directory / "report/summary.csv"),
                                  pd.read_csv(tmp_path / "regenerated/summary.csv"))
    with pytest.raises(FileExistsError):
        final.regenerate_final_report(directory, tmp_path / "regenerated")


def test_real_mode_rejects_synthetic_data_and_synthetic_mode_rejects_official_source(final_fixture, tmp_path):
    dev, held, frozen, _ = final_fixture
    with pytest.raises(ValueError, match="coverage"):
        final.run_final_evaluation(dev, held, frozen, tmp_path / "not-real", provenance={})
    with pytest.raises(ValueError, match="official source"):
        final.run_final_evaluation(dev, held, frozen, tmp_path / "not-synthetic",
            provenance={"source_sha256": SOURCE_SHA256}, run_kind="synthetic_verification")
    assert not (tmp_path / "not-real").exists()
    assert not (tmp_path / "not-synthetic").exists()


def test_missing_integrity_record_rejected(final_fixture, tmp_path, monkeypatch):
    directory, _ = save_fixture(final_fixture, tmp_path, monkeypatch)
    path = directory / "evaluation.json"
    record = json.loads(path.read_text())
    del record["record_sha256"]["predictions.csv"]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incomplete"):
        final.regenerate_final_report(directory, tmp_path / "missing-hash")


def test_write_failure_does_not_leave_success_completion(final_fixture, tmp_path, monkeypatch):
    original = final.write_table
    def fail_predictions(path, table):
        if Path(path).name == "predictions.csv":
            raise OSError("Deliberate synthetic output failure")
        return original(path, table)
    monkeypatch.setattr(final, "write_table", fail_predictions)
    dev, held, frozen, _ = final_fixture
    output = tmp_path / "io-failure"
    with pytest.raises(OSError):
        final.run_final_evaluation(dev, held, frozen, output,
            provenance={"source_sha256": "synthetic"}, run_kind="synthetic_verification")
    assert (output / "diagnostics.json").exists()
    assert not (output / "completion.json").exists()
    assert not (output / "report").exists()
