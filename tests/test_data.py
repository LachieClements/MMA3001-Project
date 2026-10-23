"""Data checks: synthetic unit cases and one real-source integrity check."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airfoil_noise.data import (
    COLUMNS, FEATURE_COLUMNS, ID_COLUMNS, SOURCE_SHA256, add_identifiers,
    audit_data, condition_table, feature_matrix, file_sha256, input_fingerprint,
    load_airfoil, validate_identifiers,
)


SOURCE = Path(__file__).resolve().parents[1] / "data/raw/airfoil_self_noise.dat"
VALID = "1000\t2\t0.1\t40\t0.002\t120\n"


@pytest.fixture
def observations() -> pd.DataFrame:
    return add_identifiers(pd.DataFrame([
        [1000, 2, 0.1, 40, 0.002, 120],
        [2000, 2, 0.1, 40, 0.002, 121],
        [1000, -1, 0.2, 50, 0.003, 119],
    ], columns=COLUMNS))


def test_source_file_integrity_and_audit() -> None:
    """Integration check of original bytes and known experimental grouping."""
    assert file_sha256(SOURCE) == SOURCE_SHA256
    data = load_airfoil(SOURCE, expected_sha256=SOURCE_SHA256)
    summary = audit_data(data)
    assert summary["rows"] == 1503
    assert summary["conditions"] == 106
    assert summary["exact_duplicate_rows"] == 0
    assert summary["thickness_conflicts"] == []
    assert summary["repeated_condition_frequency_rows"] == 0
    assert (summary["smallest_sweep"], summary["largest_sweep"]) == (8, 19)
    assert data["row_id"].iloc[[0, -1]].tolist() == ["row_000001", "row_001503"]


def test_loading_schema_numeric_order_identity_and_unchanged_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.dat"
    original = (VALID + "2000\t-2\t0.2\t50\t0.003\t119\n").encode()
    source.write_bytes(original)
    data = load_airfoil(source, expected_sha256=file_sha256(source))
    assert tuple(data.columns) == (*COLUMNS, *ID_COLUMNS)
    np.testing.assert_array_equal(data.loc[0, list(COLUMNS)].to_numpy(dtype=float),
                                  [1000, 2, 0.1, 40, 0.002, 120])
    assert source.read_bytes() == original
    pd.testing.assert_frame_equal(data, load_airfoil(source))
    assert data.attrs["source_sha256"] == file_sha256(source)


@pytest.mark.parametrize(("rows", "message"), [
    ("", "nonempty"),
    ("1000\t2\t0.1\t40\t0.002\n", "6 columns"),
    (VALID.rstrip() + "\t7\n", "6 columns"),
    ("frequency\tangle\tchord\tvelocity\tthickness\tspl\n", "numeric"),
    (VALID + "\n" + VALID, "nonfinite"),
    ("1000\t2\t0.1\tbad\t0.002\t120\n", "numeric"),
    ("1000\t2\t0.1\t40\t\t120\n", "nonfinite"),
    (VALID + "2000\t2\t0.1\t40\t0.002\t121\t7\n", "6 columns"),
])
def test_invalid_source_schema(tmp_path: Path, rows: str, message: str) -> None:
    path = tmp_path / "bad.dat"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_airfoil(path)


@pytest.mark.parametrize("column", range(6))
@pytest.mark.parametrize("bad_value", ["NaN", "inf", "-inf"])
def test_every_column_rejects_nonfinite(tmp_path: Path, column: int, bad_value: str) -> None:
    values = VALID.strip().split("\t")
    values[column] = bad_value
    path = tmp_path / "bad.dat"
    path.write_text("\t".join(values) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite"):
        load_airfoil(path)


@pytest.mark.parametrize("column", [0, 2, 3, 4])
@pytest.mark.parametrize("bad_value", ["0", "-1"])
def test_basic_physical_positivity(tmp_path: Path, column: int, bad_value: str) -> None:
    values = VALID.strip().split("\t")
    values[column] = bad_value
    path = tmp_path / "bad.dat"
    path.write_text("\t".join(values) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        load_airfoil(path)


def test_checksum_rejects_one_changed_target_byte(tmp_path: Path) -> None:
    path = tmp_path / "changed.dat"
    path.write_text(VALID, encoding="utf-8")
    expected = file_sha256(path)
    path.write_text(VALID.replace("120", "121"), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_airfoil(path, expected_sha256=expected)


def test_missing_source_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_airfoil(tmp_path / "missing.dat")


def test_ids_survive_reordering_and_filtering(observations: pd.DataFrame) -> None:
    original = observations.copy(deep=True)
    reordered = observations.iloc[[2, 0]].copy()
    validate_identifiers(reordered)
    assert reordered["row_id"].tolist() == ["row_000003", "row_000001"]
    assert observations["condition_id"].iloc[0] == observations["condition_id"].iloc[1]
    assert observations["condition_id"].iloc[0] != observations["condition_id"].iloc[2]
    pd.testing.assert_frame_equal(observations, original)
    assert input_fingerprint(observations) == input_fingerprint(observations.iloc[::-1])


@pytest.mark.parametrize("bad_ids", [
    ["row_000001", "row_000001", "row_000003"],
    ["row_000000", "row_000002", "row_000003"],
    [None, "row_000002", "row_000003"],
    [1, "row_000002", "row_000003"],
    ["row_1", "row_000002", "row_000003"],
])
def test_bad_row_ids_rejected(observations: pd.DataFrame, bad_ids: list) -> None:
    observations["row_id"] = bad_ids
    with pytest.raises(ValueError, match="Row identifiers"):
        validate_identifiers(observations)


def test_condition_ids_must_match_inputs(observations: pd.DataFrame) -> None:
    observations.loc[0, "angle_deg"] = 3
    with pytest.raises(ValueError, match="Condition identifiers"):
        validate_identifiers(observations)


def test_feature_selection_is_target_free_ordered_and_independent(observations: pd.DataFrame) -> None:
    inputs = observations.drop(columns="spl_db").loc[:, list(reversed(
        [*FEATURE_COLUMNS, *ID_COLUMNS]
    ))]
    features = feature_matrix(inputs)
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert features.shape == (3, 5)
    features.iloc[0, 0] = 999
    assert inputs.loc[0, "frequency_hz"] == 1000


@pytest.mark.parametrize("fault", ["missing", "boolean", "empty", "duplicate_columns"])
def test_invalid_feature_tables(observations: pd.DataFrame, fault: str) -> None:
    if fault == "missing":
        observations = observations.drop(columns="angle_deg")
    elif fault == "boolean":
        observations["angle_deg"] = True
    elif fault == "empty":
        observations = observations.iloc[:0]
    else:
        observations.columns = ["frequency_hz", *observations.columns[:-1]]
    with pytest.raises(ValueError):
        feature_matrix(observations)


def test_identifiers_cannot_be_regenerated_on_identified_data(observations: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="six source columns"):
        add_identifiers(observations)
    with pytest.raises(ValueError, match="required"):
        validate_identifiers(observations.drop(columns="row_id"))


def test_audit_counts_measurement_duplicates_despite_unique_ids(observations: pd.DataFrame) -> None:
    duplicate = add_identifiers(pd.concat([
        observations.loc[:, list(COLUMNS)], observations.loc[[0], list(COLUMNS)],
    ], ignore_index=True))
    summary = audit_data(duplicate)
    assert summary["exact_duplicate_rows"] == 1
    assert summary["repeated_condition_frequency_rows"] == 1


def test_thickness_conflict_is_reported_without_redefining_conditions(observations: pd.DataFrame) -> None:
    observations.loc[1, "displacement_thickness_m"] = 0.004
    summary = audit_data(observations)
    assert summary["conditions"] == 2
    assert summary["thickness_conflicts"] == [{
        "condition_id": observations["condition_id"].iloc[0],
        "thicknesses_m": [0.002, 0.004],
    }]
    with pytest.raises(ValueError, match="thickness varies"):
        condition_table(observations)


def test_signed_zero_is_one_condition(observations: pd.DataFrame) -> None:
    measurements = observations.iloc[:2].loc[:, list(COLUMNS)].copy()
    measurements["angle_deg"] = [-0.0, 0.0]
    identified = add_identifiers(measurements)
    assert len(condition_table(identified)) == 1


def test_nullable_missing_feature_is_rejected(observations: pd.DataFrame) -> None:
    observations["angle_deg"] = pd.Series([2.0, pd.NA, -1.0], dtype="Float64")
    with pytest.raises(ValueError, match="nonfinite"):
        feature_matrix(observations)

