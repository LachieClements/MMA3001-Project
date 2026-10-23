"""Development mapping, estimator isolation, leakage controls and saved records."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

from airfoil_noise import validation
from airfoil_noise.data import COLUMNS, FEATURE_COLUMNS, add_identifiers, feature_matrix
from airfoil_noise.metrics import regression_metrics
from airfoil_noise.models import MODEL_NAMES, model_configuration, model_from_configuration
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


@pytest.fixture
def five_configurations():
    return {
        "training_mean": {"model": "training_mean"},
        "linear": {"model": "linear"},
        "tree": {"model": "tree", "seed": 17, "parameters": {"max_depth": 3, "min_samples_leaf": 2}},
        "svr": {"model": "svr", "parameters": {"C": 2.0}},
        "mlp": {"model": "mlp", "seed": 17, "parameters": {
            "hidden_layer_sizes": (6,), "max_iter": 5, "tol": 0.0,
        }},
    }


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
            estimators.append(self)
            return super().fit(x, y)

    def factory(name):
        return FitOnce(strategy="mean")

    monkeypatch.setattr(validation, "build_model", factory)
    result = evaluate_development(data, folds)
    assert result["status"] == "success"
    assert len(estimators) == len({id(model) for model in estimators}) == 10
    for model, record in zip(estimators, result["diagnostics"]):
        expected_indices = data.index[folds["fold"] != record["fold"]].tolist()
        assert model.training_rows == expected_indices


def test_validation_truth_does_not_affect_that_folds_fit(observations, five_configurations) -> None:
    data, folds = observations
    original = evaluate_development(data, folds, configurations=five_configurations)
    changed = data.copy()
    changed.loc[folds["fold"] == 1, "spl_db"] += 1000
    alternate = evaluate_development(changed, folds, configurations=five_configurations)
    for name in MODEL_NAMES:
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


@pytest.mark.parametrize("names", [(), ("linear", "linear"), ("linear_log_frequency",), ("forest",)])
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
        assert record["convergence_warning"] is True
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


def test_all_models_share_folds_and_save_reconstructable_records(observations, five_configurations, tmp_path) -> None:
    data, folds = observations
    result = evaluate_development(data, folds, configurations=five_configurations)
    assert result["status"] == "success"
    assert len(result["predictions"]) == 5 * len(data)
    assert result["run"]["fits_attempted"] == result["run"]["fits_succeeded"] == 25
    assert not result["predictions"].duplicated(["model", "row_id"]).any()
    output = tmp_path / "five-model-synthetic-check"
    save_development_results(result, output, provenance={"kind": "synthetic_check"})
    run = json.loads((output / "run.json").read_text())
    records = pd.read_csv(output / "predictions.csv", float_precision="round_trip")
    assert run["schema_version"] == 2
    assert run["model_methods"] == {name: name for name in MODEL_NAMES}
    assert run["model_random_seeds"] == {
        "training_mean": None, "linear": None, "tree": 17, "svr": None, "mlp": 17,
    }
    assert run["feature_transformations"]["linear"] == []
    assert run["feature_transformations"]["tree"] == []
    assert run["target_transformation"] == "none"
    for name in ("svr", "mlp"):
        steps = run["feature_transformations"][name]
        assert steps[0]["step_name"] == "scaler"
        assert steps[0]["estimator_class"].endswith(".StandardScaler")
        assert steps[0]["parameters"]["with_mean"] is True
    for name, rows in records.groupby("model"):
        assert set(rows["row_id"]) == set(data["row_id"])
        np.testing.assert_array_equal(rows["fold"], folds.set_index("row_id").loc[rows["row_id"], "fold"])
        configuration = run["models"][name]
        for fold in range(1, 6):
            estimator = model_from_configuration(configuration)
            assert model_configuration(estimator) == configuration
            training = folds["fold"].to_numpy() != fold
            expected = rows[rows["fold"] == fold]
            validation_data = data.set_index("row_id").loc[expected["row_id"]]
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                estimator.fit(feature_matrix(data.iloc[np.flatnonzero(training)]), data.loc[training, "spl_db"])
            np.testing.assert_allclose(
                estimator.predict(feature_matrix(validation_data)), expected["prediction_db"], rtol=1e-12, atol=1e-12,
            )
    assert len(json.loads((output / "diagnostics.json").read_text())) == 25


def test_nested_diagnostics_and_training_only_scaler_statistics(observations, five_configurations) -> None:
    data, folds = observations
    result = evaluate_development(data, folds, configurations=five_configurations)
    for record in result["diagnostics"]:
        fitted = record["fitted"]
        assert fitted["n_features_in_"] == 5
        if record["model"] == "tree":
            assert 0 <= fitted["depth"] <= 3
            assert fitted["node_count"] == 2 * fitted["leaf_count"] - 1
        if record["model"] in ("svr", "mlp"):
            training = folds["fold"].to_numpy() != record["fold"]
            x = feature_matrix(data.iloc[np.flatnonzero(training)]).to_numpy()
            scaler = fitted["steps"]["scaler"]
            assert scaler["n_samples_seen_"] == record["n_train"]
            assert scaler["feature_names_in_"] == list(FEATURE_COLUMNS)
            np.testing.assert_allclose(scaler["mean_"], x.mean(axis=0), rtol=1e-13)
            np.testing.assert_allclose(scaler["var_"], x.var(axis=0), rtol=1e-13)
            np.testing.assert_allclose(scaler["scale_"], np.sqrt(x.var(axis=0)), rtol=1e-13)
            regressor = fitted["steps"]["regressor"]
            if record["model"] == "svr":
                assert regressor["support_vector_count"] == sum(regressor["n_support_"])
                assert 0 < regressor["support_vector_count"] <= record["n_train"]
                assert regressor["fit_status_"] == 0
            else:
                assert regressor["n_iter_"] == len(regressor["loss_curve_"]) == 5
                assert regressor["loss_"] == pytest.approx(regressor["loss_curve_"][-1])
                assert regressor["parameter_count"] == 43  # 5*6 + 6 biases + 6*1 + 1 bias.
                assert np.isfinite(regressor["loss_curve_"]).all()


def test_validation_features_cannot_change_training_scaler(observations, five_configurations) -> None:
    data, folds = observations
    configurations = {name: five_configurations[name] for name in ("svr", "mlp")}
    original = evaluate_development(data, folds, configurations=configurations)
    changed = data.copy()
    changed.loc[folds["fold"] == 1, "frequency_hz"] *= 100
    alternate = evaluate_development(changed, folds, configurations=configurations)
    for name in ("svr", "mlp"):
        before = next(r for r in original["diagnostics"] if r["model"] == name and r["fold"] == 1)
        after = next(r for r in alternate["diagnostics"] if r["model"] == name and r["fold"] == 1)
        assert before["fitted"] == after["fitted"]


def test_every_fold_has_fresh_nested_pipeline_state(observations, five_configurations, monkeypatch) -> None:
    instances, prototypes = [], []
    original_clone = validation.clone

    def spy_clone(prototype):
        fresh = original_clone(prototype)
        if hasattr(fresh, "named_steps"):
            assert not hasattr(fresh.named_steps["scaler"], "mean_")
            assert fresh.named_steps["scaler"] is not prototype.named_steps["scaler"]
            assert fresh.named_steps["regressor"] is not prototype.named_steps["regressor"]
        instances.append(fresh)
        prototypes.append(prototype)
        return fresh

    monkeypatch.setattr(validation, "clone", spy_clone)
    result = evaluate_development(*observations, configurations=five_configurations)
    assert result["status"] == "success"
    assert len(instances) == len({id(instance) for instance in instances}) == 25
    pipelines = [instance for instance in instances if hasattr(instance, "named_steps")]
    assert len({id(model.named_steps["scaler"]) for model in pipelines}) == 10
    assert len({id(model.named_steps["regressor"]) for model in pipelines}) == 10
    assert all(not hasattr(prototype, "n_features_in_") for prototype in prototypes)


def test_multiple_seed_configuration_ids_remain_distinct(observations) -> None:
    configurations = {
        f"mlp_seed{seed}": {"model": "mlp", "seed": seed, "parameters": {"max_iter": 2}}
        for seed in (17, 29)
    }
    result = evaluate_development(*observations, configurations=configurations)
    assert result["status"] == "success"
    assert set(result["predictions"]["model"]) == set(configurations)
    assert result["run"]["configurations_per_model"] == {"mlp": 2}
    assert result["run"]["model_random_seeds"] == {"mlp_seed17": 17, "mlp_seed29": 29}
    assert result["run"]["configuration_selection"].startswith("None")


@pytest.mark.parametrize("configurations", [
    {}, [], {"": {"model": "tree"}}, {"tree": {}},
    {"tree": {"model": "forest"}}, {"tree": {"model": "tree", "unknown": 1}},
    {"valid": {"model": "linear"}, "invalid": {"model": "svr", "parameters": {"C": -1}}},
])
def test_invalid_configurations_fail_before_any_fold_fit(observations, configurations, monkeypatch) -> None:
    def forbidden_clone(*args, **kwargs):
        pytest.fail("All configurations must be validated before evaluating any fold.")

    monkeypatch.setattr(validation, "clone", forbidden_clone)
    with pytest.raises(ValueError):
        evaluate_development(*observations, configurations=configurations)


def test_names_and_configurations_are_mutually_exclusive(observations) -> None:
    with pytest.raises(ValueError, match="not both"):
        evaluate_development(*observations, model_names=("tree",), configurations={"tree": {"model": "tree"}})


def test_real_mlp_iteration_limit_warning_is_recorded(observations) -> None:
    result = evaluate_development(*observations, configurations={
        "mlp_limit_check": {"model": "mlp", "seed": 17, "parameters": {"max_iter": 1, "tol": 0}},
    })
    assert result["status"] == "success"  # A finite prediction with a warning remains traceable.
    for record in result["diagnostics"]:
        assert record["convergence_warning"] is True
        assert any(warning["category"] == "ConvergenceWarning" for warning in record["warnings"])
        assert record["fitted"]["steps"]["regressor"]["n_iter_"] == 1


def test_unknown_json_objects_rejected_before_output_creation(observations, tmp_path) -> None:
    result = evaluate_development(*observations)
    output = tmp_path / "invalid-record"
    with pytest.raises(TypeError):
        save_development_results(result, output, provenance={"opaque": object()})
    assert not output.exists()
