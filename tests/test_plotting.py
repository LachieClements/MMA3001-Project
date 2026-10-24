"""Figures consume records without repeating experiments."""

from pathlib import Path
import json
import pytest

from airfoil_noise.development import run_study
from airfoil_noise.benchmarking import benchmark_selected
from airfoil_noise.plotting import plot_development
from test_development import small_protocol, provenance
from test_validation import observations
from test_final_evaluation import final_fixture


def test_all_development_figures_from_saved_synthetic_records(observations, small_protocol, tmp_path, monkeypatch):
    search, benchmark, figures = [tmp_path / name for name in ("search", "benchmark", "figures")]
    run_study(*observations, small_protocol, search, provenance=provenance())
    benchmark_selected(observations[0], search / "selected_configurations.json",
                       small_protocol, benchmark, provenance=provenance())
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    def forbidden(*args, **kwargs):
        pytest.fail("Plotting must not fit models.")
    for cls in (DummyRegressor, LinearRegression, DecisionTreeRegressor, SVR, MLPRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)
    stems = plot_development(search, benchmark, figures)
    assert len(stems) == 7
    for stem in stems:
        assert (figures / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (figures / f"{stem}.svg").read_text(encoding="utf-8")
    manifest = json.loads((figures / "manifest.json").read_text())
    assert manifest["fits_attempted"] == 0 and manifest["source_sha256"]
    with pytest.raises(FileExistsError):
        plot_development(search, benchmark, figures)


def test_final_figures_and_reports_regenerate_without_any_model_calls(final_fixture, tmp_path, monkeypatch):
    from airfoil_noise import final_evaluation as final
    from airfoil_noise.plotting import plot_final
    from airfoil_noise.__main__ import main
    dev, held, frozen, _ = final_fixture
    run = tmp_path / "synthetic-final"
    result = final.run_final_evaluation(dev, held, frozen, run,
        provenance={"source_sha256": "synthetic", "command": ["synthetic_full_workflow"]},
        run_kind="synthetic_verification")
    assert result["fits_attempted"] == result["prediction_calls"] == 5
    assert result["status"] == "success_warning_qualified"
    original = (run / "predictions.csv").read_bytes()
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    def forbidden(*args, **kwargs):
        pytest.fail("Saved-record reporting/plotting must not construct, fit or predict.")
    for cls in (DummyRegressor, LinearRegression, DecisionTreeRegressor, SVR, MLPRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)
        monkeypatch.setattr(cls, "predict", forbidden)
    monkeypatch.setattr(final, "model_from_configuration", forbidden)
    regenerated = tmp_path / "regenerated-final"
    assert main(["report-final", "--run-dir", str(run), "--output-dir", str(regenerated)]) == 0
    for name in ("summary.csv", "condition_metrics.csv", "development_vs_final.csv"):
        assert (run / "report" / name).read_bytes() == (regenerated / name).read_bytes()
    manifest = json.loads((regenerated / "figures/manifest.json").read_text())
    assert len(manifest["figures"]) == 4
    assert manifest["condition_ids_in_order"] == sorted(held.condition_id.unique())
    assert manifest["fits_attempted"] == manifest["prediction_calls"] == 0
    for stem in manifest["figures"]:
        assert (regenerated / "figures" / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (regenerated / "figures" / f"{stem}.svg").read_text(encoding="utf-8")
    assert (run / "predictions.csv").read_bytes() == original
    with pytest.raises(FileExistsError):
        plot_final(run, regenerated / "figures")


def test_final_sweeps_show_every_one_of_22_synthetic_conditions(final_fixture, tmp_path):
    import pandas as pd
    from airfoil_noise.data import COLUMNS, add_identifiers
    from airfoil_noise import final_evaluation as final
    # Exactly22 test conditions, but synthetic measurements and short-epoch settings.
    rows = [[frequency, angle, .1, 40., .002, 120 + .5 * angle - .001 * frequency]
            for angle in range(30) for frequency in (500., 2000.)]
    source = add_identifiers(pd.DataFrame(rows, columns=COLUMNS))
    development = source[source.angle_deg < 8].copy()
    holdout = source[source.angle_deg >= 8].copy()
    directory = tmp_path / "all-22-synthetic"
    final.run_final_evaluation(development, holdout, final_fixture[2], directory,
        provenance={"source_sha256": "synthetic"}, run_kind="synthetic_verification")
    manifest = json.loads((directory / "report/figures/manifest.json").read_text())
    assert len(manifest["condition_ids_in_order"]) == 22
    assert set(manifest["condition_ids_in_order"]) == set(holdout.condition_id)
    assert len(pd.read_csv(directory / "report/condition_metrics.csv")) == 110
    assert len(pd.read_csv(directory / "predictions.csv")) == 220
    pages = [stem for stem in manifest["figures"] if stem.startswith("frequency_sweeps_")]
    assert pages == ["frequency_sweeps_01", "frequency_sweeps_02"]
    assert "22: c=" in (directory / "report/figures/frequency_sweeps_02.svg").read_text()
