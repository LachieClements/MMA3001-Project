"""The baseline wrapper uses frozen development folds and excludes test targets."""

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.baseline import development_baselines
from airfoil_noise.data import COLUMNS, load_airfoil
from airfoil_noise.split import load_development, read_frozen_splits, write_frozen_splits
from airfoil_noise.validation import evaluate_development


@pytest.mark.parametrize("additional_model", [None, "tree", "svr", "mlp"])
def test_frozen_workflow_is_independent_of_final_test_targets(tmp_path, additional_model) -> None:
    source = tmp_path / "synthetic.dat"
    rows = [
        [frequency, angle, .1, 40, .002, 120 + angle - .001 * frequency]
        for angle in range(15) for frequency in [500, 1000, 2000]
    ]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(source, sep="\t", header=False, index=False)
    data = load_airfoil(source)
    directory = tmp_path / "splits"
    write_frozen_splits(data, directory)
    original_bytes = {path.name: path.read_bytes() for path in directory.iterdir()}
    development, row_folds = load_development(data, directory)
    def evaluate(observations, assignments):
        if additional_model is None:
            return development_baselines(observations, assignments)
        parameters = {"hidden_layer_sizes": (6,), "max_iter": 5} if additional_model == "mlp" else {}
        return evaluate_development(observations, assignments, configurations={
            additional_model: {"model": additional_model, "parameters": parameters},
        })

    original = evaluate(development, row_folds)
    if additional_model is None:
        shared = evaluate_development(development, row_folds)
        pd.testing.assert_frame_equal(original["predictions"], shared["predictions"])
        pd.testing.assert_frame_equal(original["summary"], shared["summary"])

    manifest, _, _ = read_frozen_splits(data, directory)
    test_ids = manifest.loc[manifest["split"] == "test", "condition_id"]
    changed = data.copy()
    changed.loc[changed["condition_id"].isin(test_ids), "spl_db"] = np.nan
    alternate_development, alternate_folds = load_development(changed, directory)
    alternate = evaluate(alternate_development, alternate_folds)
    for table in ("predictions", "summary", "condition_metrics", "fold_metrics"):
        pd.testing.assert_frame_equal(original[table], alternate[table])
    assert original["run"]["models"] == alternate["run"]["models"]
    assert set(original["predictions"]["condition_id"]).isdisjoint(test_ids)
    assert original_bytes == {path.name: path.read_bytes() for path in directory.iterdir()}

