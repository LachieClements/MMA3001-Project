"""Checks that generated holdouts are complete and condition-disjoint."""

from pathlib import Path

import pandas as pd
import pytest

from airfoil_noise.data import load_airfoil
from airfoil_noise.split import (
    CONDITION_COLUMNS, apply_manifest, make_manifest, read_manifest,
    validate_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/airfoil_self_noise.dat"


def test_generated_split_is_repeatable_and_has_no_condition_overlap() -> None:
    """Frequency siblings must never appear in both development and test."""
    data = load_airfoil(SOURCE)
    stored = make_manifest(data)
    pd.testing.assert_frame_equal(stored, make_manifest(data))
    development, test = apply_manifest(data, stored)
    assert (len(development), len(test)) == (1194, 309)
    dev_groups = set(map(tuple, development[list(CONDITION_COLUMNS)].to_numpy()))
    test_groups = set(map(tuple, test[list(CONDITION_COLUMNS)].to_numpy()))
    assert not dev_groups.intersection(test_groups)
    assert len(dev_groups) == 84
    assert len(test_groups) == 22


def test_missing_or_conflicting_condition_assignment_fails() -> None:
    """An edited split cannot silently omit or duplicate an experimental run."""
    data = load_airfoil(SOURCE)
    manifest = make_manifest(data)
    with pytest.raises(ValueError, match="no condition-level assignment"):
        validate_manifest(data, manifest.iloc[:-1].copy())
    duplicate = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="multiple split assignments"):
        validate_manifest(data, duplicate)

