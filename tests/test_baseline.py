"""Check grouped baseline behaviour without consulting final-test targets."""

from pathlib import Path

import numpy as np

from airfoil_noise.baseline import development_baselines
from airfoil_noise.data import load_airfoil
from airfoil_noise.split import apply_manifest, make_manifest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/airfoil_self_noise.dat"


def test_baselines_do_not_depend_on_test_targets() -> None:
    """Changing every test label must leave development CV results unchanged."""
    data = load_airfoil(SOURCE)
    manifest = make_manifest(data)
    development, test = apply_manifest(data, manifest)
    original = development_baselines(development)

    changed = data.copy()
    changed.loc[test.index, "spl_db"] = 999.0
    changed_development, _ = apply_manifest(changed, manifest)
    alternate = development_baselines(changed_development)
    assert np.array_equal(original["mae_db"], alternate["mae_db"])
    assert np.array_equal(original["rmse_db"], alternate["rmse_db"])
    assert set(original["model"]) == {"training_mean", "linear_log_frequency"}
    assert len(original) == 10

