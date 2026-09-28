"""Checks for source-file schema and unsupported inputs."""

from pathlib import Path

import pandas as pd
import pytest

from airfoil_noise.data import audit_data, load_airfoil


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/airfoil_self_noise.dat"


def test_source_file_has_expected_observations() -> None:
    """A changed or incomplete source file should fail the reproducibility check."""
    summary = audit_data(load_airfoil(SOURCE))
    assert summary["rows"] == 1503
    assert summary["conditions"] == 106
    assert summary["exact_duplicate_rows"] == 0
    assert (summary["smallest_sweep"], summary["largest_sweep"]) == (8, 19)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("1000\t2\t0.1\t40\t0.002\n", "6 columns"),
        ("1000\t2\t0.1\t40\tNaN\t120\n", "nonfinite"),
        ("0\t2\t0.1\t40\t0.002\t120\n", "positive"),
        ("1000\t2\t0.1\tbad\t0.002\t120\n", "numeric"),
    ],
)
def test_invalid_source_values_are_rejected(
    tmp_path: Path, rows: str, message: str
) -> None:
    """Bad column counts, measurements and frequency cannot enter modelling."""
    path = tmp_path / "bad.dat"
    path.write_text(rows, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_airfoil(path)

