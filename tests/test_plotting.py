"""Figures consume records without repeating experiments."""

from pathlib import Path
import json
import pytest

from airfoil_noise.development import run_study
from airfoil_noise.benchmarking import benchmark_selected
from airfoil_noise.plotting import plot_development
from test_development import small_protocol, provenance
from test_validation import observations


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
