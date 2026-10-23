"""Synthetic split/fold tests plus verification of the frozen source bundle."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.data import COLUMNS, SOURCE_SHA256, condition_table, load_airfoil
from airfoil_noise.split import (
    ASSIGNMENT_FILES, apply_manifest, load_development, make_development_folds,
    make_manifest, read_frozen_splits, read_manifest, split_coverage,
    validate_folds, validate_manifest, write_frozen_splits,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def observations(tmp_path: Path) -> pd.DataFrame:
    """Fifteen unequal sweeps; no original dataset needed."""
    rows = [
        [500 * (frequency + 1), group, 0.1, 40, 0.002 + group * 0.00001, float(110 + frequency + group)]
        for group in range(15)
        for frequency in range(3 + group % 4)
    ]
    source = tmp_path / "synthetic.dat"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(source, sep="\t", header=False, index=False)
    return load_airfoil(source)


def test_complete_repeatable_holdout_and_input_only_construction(observations: pd.DataFrame) -> None:
    manifest = make_manifest(observations)
    pd.testing.assert_frame_equal(manifest, make_manifest(observations.iloc[::-1]))
    pd.testing.assert_frame_equal(manifest, make_manifest(observations.drop(columns="spl_db")))
    development, test = apply_manifest(observations, manifest)
    assert set(development["condition_id"]).isdisjoint(test["condition_id"])
    assert set(development["row_id"]) | set(test["row_id"]) == set(observations["row_id"])
    assert len(development) + len(test) == len(observations)
    assert development["condition_id"].nunique() == 12
    assert test["condition_id"].nunique() == 3


@pytest.mark.parametrize("fraction", [0, -0.2, 1, 1.1, np.nan, True])
def test_invalid_fraction(observations: pd.DataFrame, fraction: float) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        make_manifest(observations, test_fraction=fraction)


@pytest.mark.parametrize("seed", [-1, 2**32, 1.5, True])
def test_invalid_seed(observations: pd.DataFrame, seed: int) -> None:
    with pytest.raises(ValueError, match="seed"):
        make_manifest(observations, seed=seed)


def test_too_few_conditions_or_no_development(observations: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="conditions"):
        make_manifest(observations.iloc[:3])
    with pytest.raises(ValueError, match="nonempty development"):
        make_manifest(observations, test_fraction=0.99)


def test_index_regression_and_no_input_mutation(observations: pd.DataFrame) -> None:
    observations.index = pd.Index([f"original-{i}" for i in range(len(observations))], name="source")
    original = observations.copy(deep=True)
    manifest = make_manifest(observations)
    for partition in apply_manifest(observations, manifest):
        expected = observations.loc[observations["row_id"].isin(partition["row_id"])]
        pd.testing.assert_frame_equal(partition, expected)
    pd.testing.assert_frame_equal(observations, original)


@pytest.mark.parametrize("fault", [
    "missing", "duplicate", "conflicting", "unknown", "missing_key",
    "bad_label", "only_test", "wrong_key", "schema",
])
def test_reject_bad_manifest(observations: pd.DataFrame, fault: str) -> None:
    manifest = make_manifest(observations)
    if fault == "missing":
        manifest = manifest.iloc[:-1]
    elif fault in ("duplicate", "conflicting"):
        extra = manifest.iloc[[0]].copy()
        if fault == "conflicting":
            extra["split"] = "test" if extra["split"].iloc[0] == "development" else "development"
        manifest = pd.concat([manifest, extra], ignore_index=True)
    elif fault == "unknown":
        extra = manifest.iloc[[0]].copy()
        extra["condition_id"] = "unknown"
        extra["angle_deg"] = 99
        manifest = pd.concat([manifest, extra], ignore_index=True)
    elif fault == "missing_key":
        manifest.loc[0, "chord_m"] = np.nan
    elif fault == "bad_label":
        manifest.loc[0, "split"] = "validation"
    elif fault == "only_test":
        manifest["split"] = "test"
    elif fault == "wrong_key":
        manifest["angle_deg"] = manifest["angle_deg"].astype(float)
        manifest.loc[0, "angle_deg"] += 0.5
    else:
        manifest = manifest.loc[:, list(reversed(manifest.columns))]
    with pytest.raises(ValueError):
        validate_manifest(observations, manifest)


def test_read_manifest_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("condition_id,split\nx,test\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_manifest(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_manifest(path)


def test_folds_are_complete_grouped_and_repeatable(observations: pd.DataFrame) -> None:
    manifest = make_manifest(observations)
    folds = make_development_folds(observations, manifest)
    pd.testing.assert_frame_equal(folds, make_development_folds(observations.iloc[::-1], manifest))
    pd.testing.assert_frame_equal(folds, make_development_folds(
        observations.drop(columns="spl_db"), manifest
    ))
    development, test = apply_manifest(observations, manifest)
    mapped = development["condition_id"].map(folds.set_index("condition_id")["fold"])
    assert mapped.notna().all()
    assert set(folds["condition_id"]).isdisjoint(test["condition_id"])
    visits = np.zeros(len(development), dtype=int)
    for fold in range(1, 6):
        validation = mapped.eq(fold).to_numpy()
        visits += validation
        assert set(development.loc[validation, "condition_id"]).isdisjoint(
            development.loc[~validation, "condition_id"]
        )
    np.testing.assert_array_equal(visits, np.ones(len(development), dtype=int))


@pytest.mark.parametrize("count", [0, 1, 99, 2.5, True])
def test_invalid_fold_count(observations: pd.DataFrame, count: int) -> None:
    with pytest.raises(ValueError, match="n_splits"):
        make_development_folds(observations, make_manifest(observations), n_splits=count)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "test", "unknown", "zero", "fraction", "empty_fold", "null"])
def test_invalid_fold_assignments(observations: pd.DataFrame, fault: str) -> None:
    manifest = make_manifest(observations)
    folds = make_development_folds(observations, manifest)
    if fault == "missing":
        folds = folds.iloc[:-1]
    elif fault == "duplicate":
        folds = pd.concat([folds, folds.iloc[[0]]], ignore_index=True)
    elif fault in ("test", "unknown"):
        folds.loc[0, "condition_id"] = (
            manifest.loc[manifest["split"].eq("test"), "condition_id"].iloc[0]
            if fault == "test" else "unknown"
        )
    elif fault == "zero":
        folds.loc[0, "fold"] = 0
    elif fault == "fraction":
        folds["fold"] = folds["fold"].astype(float)
        folds.loc[0, "fold"] = 1.5
    elif fault == "empty_fold":
        folds.loc[folds["fold"].eq(5), "fold"] = 4
    else:
        folds.loc[0, "condition_id"] = None
    with pytest.raises(ValueError):
        validate_folds(observations, manifest, folds)


def test_bundle_roundtrip_and_development_boundary(tmp_path: Path, observations: pd.DataFrame) -> None:
    directory = tmp_path / "splits"
    saved = write_frozen_splits(observations, directory)
    manifest, folds, metadata = read_frozen_splits(observations, directory)
    assert saved == metadata
    pd.testing.assert_frame_equal(manifest, make_manifest(observations))
    pd.testing.assert_frame_equal(folds, make_development_folds(observations, manifest))
    development, row_folds = load_development(observations, directory)
    assert row_folds["row_id"].tolist() == development["row_id"].tolist()
    assert row_folds["row_id"].is_unique
    assert row_folds["fold"].notna().all()
    assert len(row_folds) == metadata["counts"]["development"]["rows"]

    # Even unavailable/nonfinite held-out labels cannot enter development checks.
    changed = observations.copy()
    held_out = manifest.loc[manifest["split"].eq("test"), "condition_id"]
    changed.loc[changed["condition_id"].isin(held_out), "spl_db"] = np.nan
    changed_development, changed_folds = load_development(changed, directory)
    pd.testing.assert_frame_equal(development, changed_development)
    pd.testing.assert_frame_equal(row_folds, changed_folds)
    pd.testing.assert_frame_equal(split_coverage(observations, manifest),
                                  split_coverage(changed.drop(columns="spl_db"), manifest))


@pytest.mark.parametrize("filename", [*ASSIGNMENT_FILES, "metadata.json"])
def test_no_bundle_file_can_be_overwritten(tmp_path: Path, observations: pd.DataFrame, filename: str) -> None:
    directory = tmp_path / "splits"
    directory.mkdir()
    path = directory / filename
    path.write_bytes(b"preserve this work")
    with pytest.raises(FileExistsError, match="not be overwritten"):
        write_frozen_splits(observations, directory)
    assert path.read_bytes() == b"preserve this work"
    assert list(directory.iterdir()) == [path]


@pytest.mark.parametrize("filename", ASSIGNMENT_FILES)
def test_assignment_byte_changes_rejected(tmp_path: Path, observations: pd.DataFrame, filename: str) -> None:
    directory = tmp_path / "splits"
    write_frozen_splits(observations, directory)
    with (directory / filename).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_frozen_splits(observations, directory)


@pytest.mark.parametrize("fault", ["frequency", "thickness", "row_ids", "source"])
def test_stale_observations_rejected(tmp_path: Path, observations: pd.DataFrame, fault: str) -> None:
    directory = tmp_path / "splits"
    write_frozen_splits(observations, directory)
    if fault == "source":
        observations.attrs["source_sha256"] = "0" * 64
    elif fault == "row_ids":
        observations.loc[[0, 1], "row_id"] = observations.loc[[1, 0], "row_id"].to_numpy()
    else:
        column = "frequency_hz" if fault == "frequency" else "displacement_thickness_m"
        observations.loc[0, column] *= 1.01
    with pytest.raises(ValueError, match="checksum|fingerprint"):
        read_frozen_splits(observations, directory)


@pytest.mark.parametrize("fault", ["missing", "json", "schema", "counts", "folds", "seed", "valid_seed", "scheme"])
def test_incomplete_or_conflicting_metadata(tmp_path: Path, observations: pd.DataFrame, fault: str) -> None:
    directory = tmp_path / "splits"
    metadata = write_frozen_splits(observations, directory)
    path = directory / "metadata.json"
    if fault == "missing":
        path.unlink()
    elif fault == "json":
        path.write_text("{broken", encoding="utf-8")
    else:
        if fault == "schema":
            metadata["schema_version"] = 99
        elif fault == "counts":
            metadata["counts"]["test"]["rows"] += 1
        elif fault == "folds":
            metadata["folds"]["n_splits"] = 4
        elif fault == "seed":
            metadata["holdout"]["seed"] = -1
        elif fault == "valid_seed":
            metadata["holdout"]["seed"] = 3002
        else:
            metadata["condition_columns"] = ["displacement_thickness_m"]
        path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        read_frozen_splits(observations, directory)


def test_no_source_identity_no_freeze(tmp_path: Path, observations: pd.DataFrame) -> None:
    observations.attrs.clear()
    with pytest.raises(ValueError, match="source SHA-256"):
        write_frozen_splits(observations, tmp_path / "splits")


def test_thickness_conflict_stops_splitting(observations: pd.DataFrame) -> None:
    observations.loc[0, "displacement_thickness_m"] *= 2
    with pytest.raises(ValueError, match="thickness varies"):
        make_manifest(observations)


def test_coverage_reports_unsupported_inputs_without_targets() -> None:
    # Explicitly withheld highest-angle condition; no labels needed.
    from airfoil_noise.data import add_identifiers

    data = add_identifiers(pd.DataFrame([
        [1000, angle, 0.1, 40, 0.002, 120] for angle in [0, 1, 10]
    ], columns=COLUMNS)).drop(columns="spl_db")
    manifest = condition_table(data)
    manifest["split"] = ["development", "development", "test"]
    coverage = split_coverage(data, manifest)
    angle_test = coverage.loc[(coverage["feature"] == "angle_deg") & (coverage["split"] == "test")].iloc[0]
    assert angle_test["rows_outside_development_range"] == 1
    assert angle_test["minimum"] == 10
    assert "spl_db" not in set(coverage["feature"])


def test_committed_bundle_matches_original_source() -> None:
    """Integration check of the persisted approved experimental design."""
    data = load_airfoil(ROOT / "data/raw/airfoil_self_noise.dat", expected_sha256=SOURCE_SHA256)
    _, folds, metadata = read_frozen_splits(data, ROOT / "data/splits")
    assert metadata["holdout"]["seed"] == 3001
    assert metadata["counts"]["development"] == {"conditions": 84, "rows": 1194}
    assert metadata["counts"]["test"] == {"conditions": 22, "rows": 309}
    assert len(folds) == 84
    assert [item["rows"] for item in metadata["counts"]["validation_folds"].values()] == [
        241, 241, 240, 240, 232,
    ]

