"""Independent prediction-call counts, fit budget and frozen configuration checks."""

import json
import numpy as np
import pytest

from airfoil_noise import benchmarking
from airfoil_noise.benchmarking import benchmark_selected, prediction_block, prediction_timings, prepared_inputs
from airfoil_noise.development import run_study
from test_development import small_protocol, provenance
from test_validation import observations


def test_prepared_inputs_are_deterministic_five_feature_frames(observations):
    data = observations[0]
    _, _, batch, singles, ids = prepared_inputs(data.sample(frac=1), 16, 32)
    assert batch.shape == (16, 5)
    assert len(singles) == 32 and all(frame.shape == (1, 5) for frame in singles)
    assert ids == prepared_inputs(data, 16, 32)[-1]
    assert len(set(ids["single_row_ids"])) == 32
    assert all("row_id" not in frame and "spl_db" not in frame for frame in singles)


@pytest.mark.parametrize("batch,single", [(0, 1), (1, 0), (99, 1), (1, 99)])
def test_invalid_prediction_input_sizes(observations, batch, single):
    with pytest.raises(ValueError):
        prepared_inputs(observations[0], batch, single)


def test_single_loop_separately_calls_same_32_prepared_rows_every_round(observations):
    singles = prepared_inputs(observations[0], 16, 32)[3]
    seen = []
    class Spy:
        def predict(self, frame):
            seen.append(frame)
            assert frame.shape == (1, 5)
            return np.zeros(1)
    counter = {"calls": 0}
    prediction_block(Spy(), singles, 3, counter)
    assert counter["calls"] == 96
    assert all(frame is singles[index % 32] for index, frame in enumerate(seen))


def test_prediction_calibration_and_measured_counts_and_order(observations, small_protocol, monkeypatch):
    _, _, batch, singles, _ = prepared_inputs(observations[0], 16, 4)
    calls = []
    class Spy:
        def __init__(self, name): self.name = name
        def predict(self, frame):
            calls.append((self.name, len(frame)))
            return np.zeros(len(frame))
    # Every synthetic block is exactly one second; no calibration doubling needed.
    clock = iter(range(1000))
    monkeypatch.setattr(benchmarking, "perf_counter", lambda: next(clock))
    settings = {**small_protocol["benchmark"], "minimum_block_seconds": .5}
    records, errors = prediction_timings({"a": Spy("a"), "b": Spy("b")}, batch, singles, settings)
    assert not errors
    assert records["prediction_calls"].sum() == len(calls)
    assert (records["prediction_calls"] == records["rounds"] * records["calls_per_round"]).all()
    measured = records[records.stage == "measured"]
    assert measured[measured["mode"] == "single"]["prediction_calls"].tolist() == [4] * 6
    assert measured[measured["mode"] == "batch"]["model"].tolist() == ["a", "b", "b", "a", "a", "b"]


def test_calibration_doubles_rounds_and_counts_every_attempt(observations, small_protocol, monkeypatch):
    _, _, batch, singles, _ = prepared_inputs(observations[0], 16, 4)
    seen = []
    class Spy:
        def predict(self, frame):
            seen.append(frame)
            return np.zeros(len(frame))
    monkeypatch.setattr(benchmarking, "perf_counter", lambda: len(seen) * .01)
    settings = {**small_protocol["benchmark"], "minimum_block_seconds": .055}
    records, failures = prediction_timings({"a": Spy()}, batch, singles, settings)
    assert not failures
    assert records["prediction_calls"].sum() == len(seen)
    assert records.query("mode == 'batch' and stage == 'calibration'")["rounds"].tolist() == [1, 2, 4, 8]
    assert records.query("mode == 'single' and stage == 'calibration'")["rounds"].tolist() == [1, 2]
    assert records.query("stage == 'measured'")["prediction_calls"].tolist() == [8] * 6


def test_prediction_failure_is_counted_without_replacement(observations, small_protocol):
    _, _, batch, singles, _ = prepared_inputs(observations[0], 16, 4)
    class Fail:
        def predict(self, frame): raise RuntimeError("deliberate prediction failure")
    records, failures = prediction_timings({"bad": Fail()}, batch, singles, small_protocol["benchmark"])
    assert records.empty
    assert len(failures) == 2
    assert all(item["actual_attempted_calls"] == 1 for item in failures.values())


def test_benchmark_uses_frozen_settings_fresh_repeated_fits_and_no_extra_refit(observations, small_protocol, tmp_path, monkeypatch):
    search, output = tmp_path / "search", tmp_path / "benchmark"
    run_study(*observations, small_protocol, search, provenance=provenance())
    frozen_path = search / "selected_configurations.json"
    before = frozen_path.read_bytes()
    original = benchmarking.model_from_configuration
    models = []
    def reconstruct(record):
        model = original(record)
        models.append(model)
        return model
    monkeypatch.setattr(benchmarking, "model_from_configuration", reconstruct)
    result = benchmark_selected(observations[0], frozen_path, small_protocol, output, provenance=provenance())
    assert result["fits_attempted"] == 30 and result["fits_failed"] == 0
    assert result["final_test_evaluated"] is False
    fitted = [model for model in models if hasattr(model, "n_features_in_") or hasattr(model, "constant_")]
    assert len(fitted) == 30 and len({id(model) for model in fitted}) == 30
    assert frozen_path.read_bytes() == before
    assert len(list(output.glob("fit-*.json"))) == 30
    assert json.loads((output / "selected_configurations.json").read_text()) == json.loads(before)
    with pytest.raises(FileExistsError):
        benchmark_selected(observations[0], frozen_path, small_protocol, output, provenance=provenance())
    changed = observations[0].copy()
    changed.loc[changed.index[0], "spl_db"] += 1
    with pytest.raises(ValueError, match="development rows differ"):
        benchmark_selected(changed, frozen_path, small_protocol, tmp_path / "changed", provenance=provenance())
    changed_protocol = json.loads(json.dumps(small_protocol))
    changed_protocol["benchmark"]["blocks"] += 1
    with pytest.raises(ValueError, match="protocol differs"):
        benchmark_selected(observations[0], frozen_path, changed_protocol, tmp_path / "changed-protocol", provenance=provenance())


def test_benchmark_failed_fits_are_retained_without_replacement(observations, small_protocol, tmp_path, monkeypatch):
    from sklearn.svm import SVR
    search, output = tmp_path / "search", tmp_path / "failed-benchmark"
    run_study(*observations, small_protocol, search, provenance=provenance())
    def failed(*args, **kwargs):
        raise RuntimeError("Deliberate benchmark fit failure")
    monkeypatch.setattr(SVR, "fit", failed)
    result = benchmark_selected(observations[0], search / "selected_configurations.json",
                                small_protocol, output, provenance=provenance())
    assert result["fits_attempted"] == 30
    assert result["fits_failed"] == 6
    import pandas as pd
    summary = pd.read_csv(output / "summary.csv").set_index("model")
    assert not summary.loc["svr", "complete"]
    assert summary.loc["svr", "measured_fits_succeeded"] == 0
    assert summary.loc["linear", "complete"]
