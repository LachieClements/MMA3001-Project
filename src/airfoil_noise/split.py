"""Freeze and verify a holdout of complete operating conditions."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


CONDITION_COLUMNS = ("chord_m", "velocity_m_s", "angle_deg")
SPLIT_COLUMNS = (*CONDITION_COLUMNS, "split")


def make_manifest(
    data: pd.DataFrame, *, test_fraction: float = 0.2, seed: int = 3001
) -> pd.DataFrame:
    """Assign each complete frequency sweep to development or final test.

    Parameters
    ----------
    data : pandas.DataFrame
        Loaded observations in source row order.
    test_fraction : float, default 0.2
        Approximate fraction of distinct operating conditions reserved for
        final testing, strictly between 0 and 1.
    seed : int, default 3001
        Random seed for reproducible group selection.

    Returns
    -------
    pandas.DataFrame
        One row per unique chord (m), velocity (m/s) and angle (degrees),
        sorted by those columns, with ``split`` set to ``development`` or
        ``test``. The manifest contains no targets.

    Raises
    ------
    ValueError
        The test fraction is invalid or there are too few conditions.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")
    groups = data.groupby(list(CONDITION_COLUMNS), sort=True).ngroup()
    if groups.nunique() < 2:
        raise ValueError("At least two operating conditions are needed.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=seed
    )
    development_rows, test_rows = next(splitter.split(data, groups=groups))
    table = data.loc[:, list(CONDITION_COLUMNS)].copy()
    table["split"] = ""
    split_column = table.columns.get_loc("split")
    table.iloc[development_rows, split_column] = "development"
    table.iloc[test_rows, split_column] = "test"
    manifest = table.drop_duplicates().sort_values(list(CONDITION_COLUMNS))
    manifest = manifest.reset_index(drop=True)
    validate_manifest(data, manifest)
    return manifest


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read a condition-level split manifest from a CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV with chord in m, velocity in m/s, angle in degrees and split.

    Returns
    -------
    pandas.DataFrame
        Four-column manifest. Use :func:`validate_manifest` against the data
        before modelling.

    Raises
    ------
    ValueError
        The CSV has an unexpected schema.
    """
    manifest = pd.read_csv(path)
    if tuple(manifest.columns) != SPLIT_COLUMNS:
        raise ValueError(f"Split file must have columns {SPLIT_COLUMNS}.")
    return manifest


def validate_manifest(data: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Check that every observed condition has exactly one split assignment.

    Parameters
    ----------
    data : pandas.DataFrame
        Original loaded observations.
    manifest : pandas.DataFrame
        One row per condition, with ``development`` or ``test`` assignment.

    Raises
    ------
    ValueError
        Schema, values, coverage or uniqueness is incorrect. This guards
        against leakage and stale or edited manifests.
    """
    if tuple(manifest.columns) != SPLIT_COLUMNS:
        raise ValueError(f"Split file must have columns {SPLIT_COLUMNS}.")
    if manifest[list(CONDITION_COLUMNS)].isna().any().any():
        raise ValueError("A condition key is missing.")
    if manifest.duplicated(subset=list(CONDITION_COLUMNS)).any():
        raise ValueError("A condition has multiple split assignments.")
    if set(manifest["split"]) != {"development", "test"}:
        raise ValueError("Both development and test splits are required.")
    merged = data.merge(
        manifest,
        on=list(CONDITION_COLUMNS),
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    if len(merged) != len(data) or not merged["_merge"].eq("both").all():
        raise ValueError("Some data rows have no condition-level assignment.")
    if len(manifest) != len(data[list(CONDITION_COLUMNS)].drop_duplicates()):
        raise ValueError("The manifest includes conditions absent from the data.")


def apply_manifest(
    data: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return development and final-test rows without mixing a condition.

    Parameters
    ----------
    data : pandas.DataFrame
        Loaded observations.
    manifest : pandas.DataFrame
        Valid condition assignment. Validation is repeated before use.

    Returns
    -------
    tuple of pandas.DataFrame
        Development rows, then held-out test rows, both in original row
        order with the six observation columns and original indices.
    """
    validate_manifest(data, manifest)
    assigned = data.merge(
        manifest, on=list(CONDITION_COLUMNS), how="left", sort=False,
        validate="many_to_one"
    )
    development = assigned.loc[assigned["split"].eq("development"), data.columns]
    test = assigned.loc[assigned["split"].eq("test"), data.columns]
    return development, test
