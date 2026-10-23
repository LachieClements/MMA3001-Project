"""Freeze input-only condition assignments and development validation folds."""

from hashlib import sha256
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from airfoil_noise.data import (
    CONDITION_COLUMNS, CONDITION_ID_SCHEME, FEATURE_COLUMNS, ID_COLUMNS,
    ROW_ID_SCHEME, condition_table, feature_matrix, file_sha256, input_fingerprint,
)


SPLIT_COLUMNS = ("condition_id", *CONDITION_COLUMNS, "split")
FOLD_COLUMNS = ("condition_id", "fold")
ASSIGNMENT_FILES = ("conditions.csv", "development_folds.csv")
METADATA_FILE = "metadata.json"


def _metadata_sha256(metadata: dict[str, object]) -> str:
    """Hash record fields except the checksum field itself."""
    fields = {key: value for key, value in metadata.items() if key != "record_sha256"}
    canonical = json.dumps(fields, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def make_manifest(
    data: pd.DataFrame, *, test_fraction: float = 0.2, seed: int = 3001,
) -> pd.DataFrame:
    """Assign complete conditions using inputs and a predeclared seed.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified observations; targets are not inspected.
    test_fraction : float, default 0.2
        Fraction of conditions reserved for testing, rounded up by
        GroupShuffleSplit. This is not a requested fraction of rows.
    seed : int, default 3001
        Fixed holdout seed chosen before model comparisons.

    Returns
    -------
    pandas.DataFrame
        One numerically sorted row per condition, labelled development/test.

    Raises
    ------
    ValueError
        Inputs, fraction, seed or number of conditions are invalid.
    """
    if isinstance(test_fraction, bool) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1.")
    table = condition_table(data)
    if len(table) < 2 or int(np.ceil(test_fraction * len(table))) >= len(table):
        raise ValueError("At least two conditions and a nonempty development split are needed.")
    groups = data.groupby(list(CONDITION_COLUMNS), sort=True).ngroup()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    _, test_rows = next(splitter.split(feature_matrix(data), groups=groups))
    test_ids = set(data["condition_id"].iloc[test_rows])
    table["split"] = np.where(table["condition_id"].isin(test_ids), "test", "development")
    validate_manifest(data, table)
    return table


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Read saved condition assignments without rounding floating-point keys.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV in the exact SPLIT_COLUMNS schema.

    Returns
    -------
    pandas.DataFrame
        Condition assignments; validate against observations before use.

    Raises
    ------
    ValueError
        The file is empty or has an unexpected schema.
    """
    try:
        manifest = pd.read_csv(path, float_precision="round_trip")
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Split file is empty.") from exc
    if tuple(manifest.columns) != SPLIT_COLUMNS:
        raise ValueError(f"Split file must have columns {SPLIT_COLUMNS}.")
    return manifest


def validate_manifest(data: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Reject missing, conflicting or unknown condition assignments.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified inputs; targets are not required.
    manifest : pandas.DataFrame
        One row per condition in SPLIT_COLUMNS order.

    Raises
    ------
    ValueError
        Schema, IDs, keys, labels, coverage or uniqueness is incorrect.
        Source/input fingerprints are checked by read_frozen_splits.
    """
    expected = condition_table(data)
    if tuple(manifest.columns) != SPLIT_COLUMNS:
        raise ValueError(f"Split file must have columns {SPLIT_COLUMNS}.")
    if manifest.isna().any().any():
        raise ValueError("A condition key, identifier or split label is missing.")
    if (manifest["condition_id"].duplicated().any()
            or manifest.duplicated(subset=list(CONDITION_COLUMNS)).any()):
        raise ValueError("A condition has multiple split assignments.")
    if set(manifest["split"]) != {"development", "test"}:
        raise ValueError("Both development and test splits are required, with no other labels.")
    expected_ids = set(expected["condition_id"])
    actual_ids = set(manifest["condition_id"])
    if expected_ids - actual_ids:
        raise ValueError("Some data rows have no condition-level assignment.")
    if actual_ids - expected_ids:
        raise ValueError("The manifest includes conditions absent from the data.")
    keys = expected.set_index("condition_id").sort_index()
    supplied = manifest.set_index("condition_id").loc[keys.index, list(CONDITION_COLUMNS)]
    if any(not pd.api.types.is_numeric_dtype(supplied[c])
           or pd.api.types.is_bool_dtype(supplied[c]) for c in CONDITION_COLUMNS):
        raise ValueError("Condition keys must be numeric.")
    if not np.array_equal(keys.to_numpy(dtype=float), supplied.to_numpy(dtype=float)):
        raise ValueError("Condition identifiers and numeric keys disagree.")


def apply_manifest(
    data: pd.DataFrame, manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition rows, preserving IDs, source order and original indices.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified observations or inputs.
    manifest : pandas.DataFrame
        Complete condition assignments, validated before use.

    Returns
    -------
    tuple of pandas.DataFrame
        Independent development and final-test copies. Development callers
        should prefer load_development, which does not return test targets.
    """
    validate_manifest(data, manifest)
    labels = data["condition_id"].map(manifest.set_index("condition_id")["split"])
    development = data.iloc[np.flatnonzero(labels.eq("development").to_numpy())].copy()
    test = data.iloc[np.flatnonzero(labels.eq("test").to_numpy())].copy()
    return development, test


def make_development_folds(
    data: pd.DataFrame, manifest: pd.DataFrame, *, n_splits: int = 5,
) -> pd.DataFrame:
    """Create target-free GroupKFold assignments for development conditions.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified inputs; targets are not inspected.
    manifest : pandas.DataFrame
        Frozen holdout assignment.
    n_splits : int, default 5
        Fold count. GroupKFold approximately balances row counts, with no
        shuffling or fold seed, while preserving complete conditions.

    Returns
    -------
    pandas.DataFrame
        One condition_id and one one-based validation fold per development
        condition. Final-test conditions have no fold assignment.

    Raises
    ------
    ValueError
        Fold count is invalid or exceeds development condition count.
    """
    inputs = data.loc[:, [*FEATURE_COLUMNS, *ID_COLUMNS]]
    development, _ = apply_manifest(inputs, manifest)
    if (isinstance(n_splits, bool) or not isinstance(n_splits, int)
            or not 2 <= n_splits <= development["condition_id"].nunique()):
        raise ValueError("Need at least n_splits distinct development conditions and n_splits >= 2.")
    groups = development.groupby(list(CONDITION_COLUMNS), sort=True).ngroup()
    assignments = {}
    for fold, (_, validation) in enumerate(
        GroupKFold(n_splits=n_splits).split(feature_matrix(development), groups=groups), start=1,
    ):
        for condition_id in development["condition_id"].iloc[validation].unique():
            assignments[condition_id] = fold
    folds = condition_table(development).loc[:, ["condition_id"]]
    folds["fold"] = folds["condition_id"].map(assignments)
    validate_folds(data, manifest, folds, n_splits=n_splits)
    return folds


def validate_folds(
    data: pd.DataFrame, manifest: pd.DataFrame, folds: pd.DataFrame, *, n_splits: int = 5,
) -> None:
    """Check complete, unique development-only validation assignments.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified inputs.
    manifest : pandas.DataFrame
        Complete condition holdout assignment.
    folds : pandas.DataFrame
        condition_id and integer fold columns, in that order.
    n_splits : int, default 5
        Required labels are exactly 1 through n_splits.

    Raises
    ------
    ValueError
        A fold is invalid, missing, duplicated, unknown or assigned to test.
    """
    validate_manifest(data, manifest)
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two.")
    if tuple(folds.columns) != FOLD_COLUMNS or folds.isna().any().any():
        raise ValueError(f"Fold file must have nonmissing columns {FOLD_COLUMNS}.")
    if folds["condition_id"].duplicated().any():
        raise ValueError("A development condition has multiple fold assignments.")
    expected = set(manifest.loc[manifest["split"].eq("development"), "condition_id"])
    if set(folds["condition_id"]) != expected:
        raise ValueError("Folds must cover every development condition and exclude test/unknown conditions.")
    labels = folds["fold"]
    if (not pd.api.types.is_integer_dtype(labels) or pd.api.types.is_bool_dtype(labels)
            or set(labels) != set(range(1, n_splits + 1))):
        raise ValueError(f"Fold labels must be integers covering 1 through {n_splits}.")


def assignment_summary(
    data: pd.DataFrame, manifest: pd.DataFrame, folds: pd.DataFrame,
) -> dict[str, object]:
    """Count assigned rows and conditions without reading targets.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified inputs.
    manifest : pandas.DataFrame
        Validated holdout assignments.
    folds : pandas.DataFrame
        Validated development-fold assignments.

    Returns
    -------
    dict
        Development/test counts and validation counts for each fold.
    """
    sizes = data.groupby("condition_id").size()
    summary = {}
    for label in ("development", "test"):
        ids = manifest.loc[manifest["split"].eq(label), "condition_id"]
        summary[label] = {"conditions": len(ids), "rows": int(sizes.loc[ids].sum())}
    summary["validation_folds"] = {
        str(fold): {"conditions": len(table), "rows": int(sizes.loc[table["condition_id"]].sum())}
        for fold, table in folds.groupby("fold", sort=True)
    }
    return summary


def write_frozen_splits(
    data: pd.DataFrame, directory: str | Path, *, seed: int = 3001,
    test_fraction: float = 0.2, n_splits: int = 5,
) -> dict[str, object]:
    """Write a new split bundle, refusing to replace existing output files.

    Parameters
    ----------
    data : pandas.DataFrame
        Source observations loaded with load_airfoil, with source digest.
    directory : str or pathlib.Path
        Destination for conditions.csv, development_folds.csv, metadata.json.
    seed : int, default 3001
        Predeclared holdout seed.
    test_fraction : float, default 0.2
        Fraction of conditions reserved for testing.
    n_splits : int, default 5
        Number of development folds.

    Returns
    -------
    dict
        Provenance metadata. Written last as the completion record; reading
        an incomplete bundle fails.

    Raises
    ------
    FileExistsError
        Any bundle output exists. No force-overwrite option is provided.
    ValueError
        Source identity or assignments are invalid.
    """
    directory = Path(directory)
    paths = [directory / name for name in (*ASSIGNMENT_FILES, METADATA_FILE)]
    if any(path.exists() for path in paths):
        raise FileExistsError("Split outputs already exist; frozen assignments will not be overwritten.")
    digest = data.attrs.get("source_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Load source observations before freezing splits; source SHA-256 is missing.")
    manifest = make_manifest(data, test_fraction=test_fraction, seed=seed)
    folds = make_development_folds(data, manifest, n_splits=n_splits)
    texts = [table.to_csv(index=False, float_format="%.17g", lineterminator="\n")
             for table in (manifest, folds)]
    metadata = {
        "schema_version": 1,
        "source_sha256": digest,
        "input_fingerprint": input_fingerprint(data),
        "condition_columns": list(CONDITION_COLUMNS),
        "row_id_scheme": ROW_ID_SCHEME,
        "condition_id_scheme": CONDITION_ID_SCHEME,
        "holdout": {"algorithm": "GroupShuffleSplit", "seed": seed, "test_fraction": test_fraction},
        "folds": {"algorithm": "GroupKFold", "n_splits": n_splits, "shuffle": False, "seed": None},
        "counts": assignment_summary(data, manifest, folds),
        "assignment_sha256": {
            name: sha256(text.encode("utf-8")).hexdigest()
            for name, text in zip(ASSIGNMENT_FILES, texts)
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    metadata["record_sha256"] = _metadata_sha256(metadata)
    texts.append(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    directory.mkdir(parents=True, exist_ok=True)
    for path, text in zip(paths, texts):
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
    return metadata


def read_frozen_splits(
    data: pd.DataFrame, directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Verify source/input identity and assignment checksums before use.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified inputs with loader-recorded source digest. The CLI first
        verifies actual source bytes against the UCI digest.
    directory : str or pathlib.Path
        Complete saved split bundle.

    Returns
    -------
    tuple
        Validated conditions, development folds and metadata.

    Raises
    ------
    ValueError
        Bundle is incomplete, stale, edited, conflicting or malformed.

    Notes
    -----
    Checksums detect accidental changes, not adversarial rewriting of all
    files and metadata. Input validation excludes in-memory targets; source
    target bytes are bound separately by the source checksum.
    """
    directory = Path(directory)
    try:
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        if (not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int
                or metadata["schema_version"] != 1):
            raise ValueError("Unsupported split metadata schema.")
        if metadata["record_sha256"] != _metadata_sha256(metadata):
            raise ValueError("Split metadata checksum mismatch; the frozen record was edited.")
        if (metadata["condition_columns"] != list(CONDITION_COLUMNS)
                or metadata["row_id_scheme"] != ROW_ID_SCHEME
                or metadata["condition_id_scheme"] != CONDITION_ID_SCHEME
                or metadata["holdout"]["algorithm"] != "GroupShuffleSplit"
                or metadata["folds"]["algorithm"] != "GroupKFold"
                or metadata["folds"]["shuffle"] is not False
                or metadata["folds"]["seed"] is not None):
            raise ValueError("Split metadata disagrees with the declared identity/split scheme.")
        seed = metadata["holdout"]["seed"]
        fraction = metadata["holdout"]["test_fraction"]
        if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
                or isinstance(fraction, bool) or not 0 < fraction < 1):
            raise ValueError("Invalid holdout seed or fraction in metadata.")
        digest = metadata["source_sha256"]
        if (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or digest != data.attrs.get("source_sha256")):
            raise ValueError("Source checksum differs from the frozen split source.")
        if metadata["input_fingerprint"] != input_fingerprint(data):
            raise ValueError("Input fingerprint differs: stale or edited observations/row identifiers.")
        for name in ASSIGNMENT_FILES:
            if file_sha256(directory / name) != metadata["assignment_sha256"][name]:
                raise ValueError(f"Assignment checksum mismatch for {name}.")
        manifest = read_manifest(directory / ASSIGNMENT_FILES[0])
        folds = pd.read_csv(directory / ASSIGNMENT_FILES[1])
        validate_folds(data, manifest, folds, n_splits=metadata["folds"]["n_splits"])
        if metadata["counts"] != assignment_summary(data, manifest, folds):
            raise ValueError("Saved row/condition counts disagree with assignments.")
    except (OSError, KeyError, TypeError, json.JSONDecodeError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Incomplete or malformed frozen split bundle.") from exc
    return manifest, folds, metadata


def load_development(
    data: pd.DataFrame, directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return development observations and row folds, excluding test targets.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified source observations, including development targets.
    directory : str or pathlib.Path
        Frozen bundle; source/input identity and assignments are checked.

    Returns
    -------
    tuple of pandas.DataFrame
        Development observations and a separate row_id, condition_id, fold
        table in matching order. Neither contains final-test observations.
    """
    manifest, folds, _ = read_frozen_splits(data, directory)
    ids = set(manifest.loc[manifest["split"].eq("development"), "condition_id"])
    development = data.iloc[np.flatnonzero(data["condition_id"].isin(ids).to_numpy())].copy()
    row_folds = development.loc[:, list(ID_COLUMNS)].copy()
    row_folds["fold"] = row_folds["condition_id"].map(folds.set_index("condition_id")["fold"])
    return development, row_folds


def split_coverage(data: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Describe input ranges/distributions without inspecting test targets.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified inputs; targets are not required.
    manifest : pandas.DataFrame
        Frozen holdout assignment.

    Returns
    -------
    pandas.DataFrame
        Per-feature/per-split counts, quartiles, ranges and row counts outside
        the development range. Marginal overlap does not establish joint
        coverage or physically supported extrapolation.
    """
    inputs = data.loc[:, [*FEATURE_COLUMNS, *ID_COLUMNS]]
    development, test = apply_manifest(inputs, manifest)
    records = []
    for column in FEATURE_COLUMNS:
        low, high = development[column].min(), development[column].max()
        for label, table in (("development", development), ("test", test)):
            values = table[column]
            records.append({
                "feature": column, "split": label, "rows": len(table),
                "conditions": table["condition_id"].nunique(), "distinct_values": values.nunique(),
                "minimum": values.min(), "q25": values.quantile(0.25), "median": values.median(),
                "q75": values.quantile(0.75), "maximum": values.max(),
                "rows_outside_development_range": int(((values < low) | (values > high)).sum()),
            })
    return pd.DataFrame.from_records(records)
