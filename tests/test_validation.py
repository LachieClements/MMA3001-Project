"""Development mapping, estimator isolation, leakage controls and saved records."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

from airfoil_noise import validation
from airfoil_noise.data import COLUMNS, FEATURE_COLUMNS, add_identifiers
from airfoil_noise.metrics import regression_metrics
from airfoil_noise.validation import evaluate_development, save_development_results


@pytest.fixture
def observations():
    rng = np.random.default_rng(51)
    rows = []
    for condition in range(11):
        chord, velocity, thickness = rng.uniform(.05, .3), rng.uniform(20, 70), rng.uniform(.001, .009)
        for frequency in np.geomspace(200, 8000, 2 + condition % 3):
            inputs = [frequency, float(condition), chord, velocity, thickness]
            target = 120 + np.dot(inputs, [-.001, .7, 25, .1, -1500])
            rows.append(inputs + [target])
    data = add_identifiers(pd.DataFrame(rows, columns=COLUMNS))
    folds = data[["row_id", "condition_id"]].copy()
    folds["fold"] = data["angle_deg"].astype(int) % 5 + 1
    return data, folds


def test_complete_predictions_identical_folds_and_exact_linear_recovery(observations) -> None:
    data, folds = observations
    result = evaluate_development(data, folds)
    predictions = result["predictions"]
    assert result["status"] == "success"
    assert result["run"]["fits_attempted"] == result["run"]["fits_succeeded"] == 10
    assert len(predictions) == 2 * len(data)
    assert not predictions.duplicated(["model", "row_id"]).any()
    for _, rows in predictions.groupby("model"):
        assert set(rows["row_id"]) == set(data["row_id"])
        expected = folds.set_index("row_id").loc[rows["row_id"]]
        assert rows["fold"].tolist() == expected["fold"].tolist()
        assert rows["condition_id"].tolist() == expected["condition_id"].tolist()
    linear = predictions[predictions["model"] == "linear"]
    np.testing.assert_allclose(linear["prediction_db"], linear["truth_db"], atol=1e-9)
    assert result["summary"].set_index("model").loc["linear", "mae_db"] < 1e-9
    assert all(record["status"] == "success" and not record["warnings"] for record in result["diagnostics"])
    for record in result["diagnostics"]:
        assert record["fit_seconds"] >= 0
        assert record["prediction_seconds"] >= 0
        assert record["fitted"]["n_features_in_"] == 5


def test_overall_primary_score_is_not_unweighted_fold_average(observations) -> None:
    data, folds = observations
    result = evaluate_development(data, folds, model_names=("training_mean",))
    overall = result["summary"].iloc[0]["equal_condition_mae_db"]
    assert overall == pytest.approx(result["condition_metrics"]["mae_db"].mean())
    assert overall != pytest.approx(result["fold_metrics"]["equal_condition_mae_db"].mean())


def test_alignment_by_id_and_consistent_feature_order(observations) -> None:
    data, folds = observations
    expected = evaluate_development(data, folds)
    shuffled = data.sample(frac=1, random_state=12).loc[:, list(reversed(data.columns))]
    shuffled.index = [0] * len(shuffled)  # DataFrame index is not a stable row ID.
    alternate = evaluate_development(shuffled, folds.sample(frac=1, random_state=13))
    pd.testing.assert_frame_equal(expected["predictions"], alternate["predictions"], atol=1e-9, rtol=1e-12)


def test_fresh_estimator_per_fold_and_no_metadata_features(observations, monkeypatch) -> None:
    data, folds = observations
    estimators = []

    class FitOnce(DummyRegressor):
        def fit(self, x, y):
            assert not hasattr(self, "constant_"), "Estimator state was reused."
            assert x.columns.tolist() == list(FEATURE_COLUMNS)
            assert x["frequency_hz"].min() >= 200  # No log transform.
            self.training_rows = x.index.tolist()
            return super().fit(x, y)

    def factory(name):
        model = FitOnce(strategy="mean")
        estimators.append(model)
        return model

    monkeypatch.setattr(validation, "build_model", factory)
    result = evaluate_development(data, folds)
    assert result["status"] == "success"
    assert len(estimators) == len({id(model) for model in estimators}) == 10
    for model, record in zip(estimators, result["diagnostics"]):
        expected_indices = data.index[folds["fold"] != record["fold"]].tolist()
        assert model.training_rows == expected_indices


def test_validation_truth_does_not_affect_that_folds_fit(observations) -> None:
    data, folds = observations
    original = evaluate_development(data, folds)
    changed = data.copy()
    changed.loc[folds["fold"] == 1, "spl_db"] += 1000
    alternate = evaluate_development(changed, folds)
    for name in ("training_mean", "linear"):
        before = original["predictions"].query("model == @name and fold == 1")
        after = alternate["predictions"].query("model == @name and fold == 1")
        np.testing.assert_allclose(before["prediction_db"], after["prediction_db"], rtol=0, atol=0)
        fit_before = next(r for r in original["diagnostics"] if r["model"] == name and r["fold"] == 1)
        fit_after = next(r for r in alternate["diagnostics"] if r["model"] == name and r["fold"] == 1)
        assert fit_before["fitted"] == fit_after["fitted"]


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate", "condition", "overlap", "gap", "float", "null", "extra_column"])
def test_invalid_fold_assignments_rejected(observations, fault) -> None:
    data, folds = observations
    if fault == "missing":
        folds = folds.iloc[:-1]
    elif fault == "extra":
        extra = folds.iloc[[0]].copy()
        extra["row_id"] = "row_999999"
        folds = pd.concat([folds, extra])
    elif fault == "duplicate":
        folds = pd.concat([folds, folds.iloc[[0]]])
    elif fault == "condition":
        folds.loc[0, "condition_id"] = "other"
    elif fault == "overlap":
        folds.loc[0, "fold"] = 2
    elif fault == "gap":
        folds["fold"] += 1
    elif fault == "float":
        folds["fold"] = folds["fold"].astype(float)
    elif fault == "null":
        folds.loc[0, "row_id"] = None
    else:
        folds["spl_db"] = 0
    with pytest.raises(ValueError):
        evaluate_development(data, folds)


@pytest.mark.parametrize("fault", ["empty", "nan", "infinite", "target_missing", "text_target", "boolean_target", "bad_feature", "row_id", "condition_id"])
def test_invalid_development_inputs_rejected(observations, fault) -> None:
    data, folds = observations
    if fault == "empty":
        data = data.iloc[:0]
    elif fault == "nan":
        data.loc[0, "spl_db"] = np.nan
    elif fault == "infinite":
        data.loc[0, "spl_db"] = np.inf
    elif fault == "target_missing":
        data = data.drop(columns="spl_db")
    elif fault == "text_target":
        data["spl_db"] = data["spl_db"].astype(str)
    elif fault == "boolean_target":
        data["spl_db"] = True
    elif fault == "bad_feature":
        data.loc[0, "frequency_hz"] = -1
    elif fault == "row_id":
        data.loc[0, "row_id"] = data.loc[1, "row_id"]
    else:
        data.loc[0, "condition_id"] = "other"
    with pytest.raises(ValueError):
        evaluate_development(data, folds)


@pytest.mark.parametrize("names", [(), ("linear", "linear"), ("linear_log_frequency",), ("tree",)])
def test_invalid_model_names_rejected(observations, names) -> None:
    with pytest.raises(ValueError):
        evaluate_development(*observations, model_names=names)


@pytest.mark.parametrize("fault", ["fit_failure", "nan_prediction", "wrong_shape"])
def test_warnings_and_failures_are_recorded_without_partial_scores(observations, monkeypatch, tmp_path, fault) -> None:
    class Faulty(DummyRegressor):
        def fit(self, x, y):
            warnings.warn("Deliberate diagnostic test", ConvergenceWarning)
            if fault == "fit_failure":
                raise RuntimeError("Deliberate failed fit")
            return super().fit(x, y)

        def predict(self, x):
            if fault == "wrong_shape":
                return np.ones((len(x), 1))
            return np.full(len(x), np.nan)

    monkeypatch.setattr(validation, "build_model", lambda name: Faulty())
    result = evaluate_development(*observations, model_names=("training_mean",))
    assert result["status"] == "failed"
    assert result["run"]["fits_failed"] == 5
    assert result["summary"].empty and result["fold_metrics"].empty
    for record in result["diagnostics"]:
        assert record["warnings"][0]["category"] == "ConvergenceWarning"
        assert record["error"]["type"] in ("RuntimeError", "ValueError")
    output = tmp_path / "failed-check"
    save_development_results(result, output, provenance={"kind": "synthetic_check"})
    assert not (output / "summary.csv").exists()
    assert json.loads((output / "run.json").read_text())["status"] == "failed"


def test_successful_warnings_retained(observations, monkeypatch) -> None:
    class Warns(DummyRegressor):
        def fit(self, x, y):
            warnings.warn("Deliberate nonfatal warning", UserWarning)
            return super().fit(x, y)

    monkeypatch.setattr(validation, "build_model", lambda name: Warns())
    result = evaluate_development(*observations, model_names=("training_mean",))
    assert result["status"] == "success"
    assert all(record["warnings"][0]["category"] == "UserWarning" for record in result["diagnostics"])


def test_saved_records_recompute_scores_and_refuse_overwrite(observations, tmp_path) -> None:
    result = evaluate_development(*observations)
    output = tmp_path / "synthetic-check"
    save_development_results(result, output, provenance={"kind": "synthetic_check"})
    assert {path.name for path in output.iterdir()} == {
        "predictions.csv", "summary.csv", "condition_metrics.csv", "fold_metrics.csv", "run.json", "diagnostics.json",
    }
    records = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    saved_summary = pd.read_csv(output / "summary.csv", float_precision="round_trip").set_index("model")
    for name, rows in records.groupby("model"):
        computed = regression_metrics(rows["truth_db"], rows["prediction_db"], rows["condition_id"])
        assert saved_summary.loc[name, "equal_condition_mae_db"] == pytest.approx(computed["equal_condition_mae_db"])
    run = json.loads((output / "run.json").read_text())
    assert run["feature_columns"] == list(FEATURE_COLUMNS)
    assert run["models"]["training_mean"]["parameters"]["strategy"] == "mean"
    assert run["provenance"]["kind"] == "synthetic_check"
    previous = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        save_development_results(result, output, provenance={})
    assert previous == {path.name: path.read_bytes() for path in output.iterdir()}


def test_constant_truth_serialization_uses_blank_csv_and_strict_json(observations, tmp_path) -> None:
    data, folds = observations
    data["spl_db"] = 123.0
    result = evaluate_development(data, folds)
    assert result["summary"]["r2"].isna().all()
    output = tmp_path / "constant-check"
    save_development_results(result, output, provenance={"undefined_example": float("nan")})
    assert pd.read_csv(output / "summary.csv")["r2"].isna().all()
    assert json.loads((output / "run.json").read_text())["provenance"]["undefined_example"] is None
