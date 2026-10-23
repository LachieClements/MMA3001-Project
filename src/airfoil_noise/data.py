"""Load, identify and audit the original UCI airfoil observations."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = (
    "frequency_hz", "angle_deg", "chord_m", "velocity_m_s",
    "displacement_thickness_m", "spl_db",
)
"""The six source columns, in file order, with units in their names."""
FEATURE_COLUMNS = COLUMNS[:-1]
TARGET_COLUMN = COLUMNS[-1]
CONDITION_COLUMNS = ("chord_m", "velocity_m_s", "angle_deg")
ID_COLUMNS = ("row_id", "condition_id")
SOURCE_SHA256 = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
ROW_ID_SCHEME = "row_ plus one-based source record number, padded to six digits"
CONDITION_ID_SCHEME = "c=<chord>|v=<velocity>|a=<angle>, using str(float(value))"
POSITIVE_COLUMNS = (
    "frequency_hz", "chord_m", "velocity_m_s", "displacement_thickness_m",
)


def file_sha256(path: str | Path) -> str:
    """Calculate a file's SHA-256 without modifying it.

    Parameters
    ----------
    path : str or pathlib.Path
        File whose exact bytes are to be checked.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    return sha256(Path(path).read_bytes()).hexdigest()


def _check_numeric(data: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if not set(columns).issubset(data.columns):
        raise ValueError(f"Required columns are missing: {columns}.")
    for column in columns:
        if (not pd.api.types.is_numeric_dtype(data[column])
                or pd.api.types.is_bool_dtype(data[column])):
            raise ValueError("Every measurement column must be numeric, not boolean.")
    if (data.loc[:, list(columns)].isna().any().any()
            or not np.isfinite(data.loc[:, list(columns)].to_numpy(dtype=float)).all()):
        raise ValueError("Missing or nonfinite observations are unsupported.")


def feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Select only the original five inputs in their declared order.

    Parameters
    ----------
    data : pandas.DataFrame
        Named measurements; targets and metadata may be present or absent.

    Returns
    -------
    pandas.DataFrame
        Independent feature copy, without targets or identifiers.

    Raises
    ------
    ValueError
        Inputs are empty, missing, nonnumeric, nonfinite or nonpositive where
        required. Negative angles remain permissible.
    """
    if data.empty or not data.columns.is_unique:
        raise ValueError("Expected nonempty observations with unique columns.")
    _check_numeric(data, FEATURE_COLUMNS)
    if (data.loc[:, list(POSITIVE_COLUMNS)] <= 0).any().any():
        raise ValueError("Frequency, chord, velocity and thickness must be positive.")
    return data.loc[:, list(FEATURE_COLUMNS)].copy()


def _condition_ids(data: pd.DataFrame) -> list[str]:
    # Numeric equality treats -0.0 and +0.0 as the same operating condition.
    identifiers = []
    for chord, velocity, angle in data.loc[:, list(CONDITION_COLUMNS)].itertuples(
        index=False, name=None,
    ):
        angle = 0.0 if angle == 0 else float(angle)
        identifiers.append(f"c={float(chord)}|v={float(velocity)}|a={angle}")
    return identifiers


def validate_identifiers(data: pd.DataFrame) -> None:
    """Check row uniqueness and correspondence of condition IDs with inputs.

    Parameters
    ----------
    data : pandas.DataFrame
        Five inputs and both IDs. Subsets and reordered rows are accepted.

    Raises
    ------
    ValueError
        IDs are missing, malformed, duplicated or inconsistent with inputs.
    """
    feature_matrix(data)
    if not set(ID_COLUMNS).issubset(data.columns):
        raise ValueError("Stable row_id and condition_id columns are required.")
    rows = data["row_id"]
    if (rows.isna().any() or rows.duplicated().any()
            or not rows.map(lambda value: isinstance(value, str)).all()
            or not rows.str.fullmatch(r"row_[0-9]{6,}").all()
            or rows.str.removeprefix("row_").map(int).le(0).any()):
        raise ValueError("Row identifiers must be unique positive source record IDs.")
    if data["condition_id"].tolist() != _condition_ids(data):
        raise ValueError("Condition identifiers do not match chord, velocity and angle.")


def add_identifiers(data: pd.DataFrame) -> pd.DataFrame:
    """Attach IDs once, before filtering or reordering source observations.

    Parameters
    ----------
    data : pandas.DataFrame
        Original six measurement columns in source record and column order.

    Returns
    -------
    pandas.DataFrame
        Copy with row_id and condition_id appended. Preserve these columns
        through subsequent operations; never regenerate IDs on a subset.
        Signed zero is normalised so numerically identical keys share an ID.

    Raises
    ------
    ValueError
        Measurements or schema are invalid, or IDs are already present.
    """
    if tuple(data.columns) != COLUMNS:
        raise ValueError(f"Expected the six source columns in order: {COLUMNS}.")
    feature_matrix(data)
    _check_numeric(data, COLUMNS)
    identified = data.copy()
    identified["row_id"] = [f"row_{i:06d}" for i in range(1, len(data) + 1)]
    identified["condition_id"] = _condition_ids(data)
    return identified


def load_airfoil(
    path: str | Path, *, expected_sha256: str | None = None,
) -> pd.DataFrame:
    """Read source measurements with optional integrity enforcement and IDs.

    Parameters
    ----------
    path : str or pathlib.Path
        Headerless, tab-separated file in the six-column source schema.
    expected_sha256 : str, optional
        Required digest of exact bytes. The CLI supplies the UCI digest by
        default; omit only for deliberately supplied other data.

    Returns
    -------
    pandas.DataFrame
        Six numeric columns followed by both IDs. Attributes record the
        source digest and ID schemes. No rows are cleaned, dropped, sorted,
        imputed or transformed.

    Raises
    ------
    FileNotFoundError
        Source file does not exist.
    ValueError
        Source integrity, schema or measurement checks fail.

    Notes
    -----
    Row numbers identify records within a checksum-identified source file.
    Reordering the raw file changes its identity. Thickness is a supplied
    measurement, not a boundary-layer calculation performed by this package.
    """
    content = Path(path).read_bytes()
    digest = sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(f"Source SHA-256 mismatch: expected {expected_sha256}, got {digest}.")
    # Parse the same bytes that were hashed; blank records must not disappear.
    try:
        raw = pd.read_csv(
            BytesIO(content), sep="\t", header=None, skip_blank_lines=False,
            float_precision="round_trip",
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Expected nonempty data with 6 columns.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("Every source record must contain exactly 6 columns.") from exc
    if raw.empty or raw.shape[1] != len(COLUMNS):
        raise ValueError("Expected nonempty data with 6 columns.")
    raw.columns = list(COLUMNS)
    try:
        numeric = raw.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Every airfoil column must be numeric.") from exc
    data = add_identifiers(numeric)
    data.attrs.update({
        "source_sha256": digest, "row_id_scheme": ROW_ID_SCHEME,
        "condition_id_scheme": CONDITION_ID_SCHEME,
    })
    return data


def condition_table(data: pd.DataFrame) -> pd.DataFrame:
    """Extract sorted condition keys, rejecting inconsistent thickness.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified inputs; targets are neither required nor inspected.

    Returns
    -------
    pandas.DataFrame
        Condition IDs and keys sorted numerically by chord, velocity, angle.

    Raises
    ------
    ValueError
        Inputs/IDs are invalid or thickness differs within a condition.
        No alternate grouping or thickness averaging is performed.
    """
    validate_identifiers(data)
    counts = data.groupby("condition_id")["displacement_thickness_m"].nunique()
    conflicts = counts.index[counts.ne(1)].tolist()
    if conflicts:
        raise ValueError(f"Displacement thickness varies within conditions: {conflicts}.")
    return (data.loc[:, ["condition_id", *CONDITION_COLUMNS]].drop_duplicates()
            .sort_values(list(CONDITION_COLUMNS)).reset_index(drop=True))


def input_fingerprint(data: pd.DataFrame) -> str:
    """Bind inputs to row IDs without inspecting targets.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified observations in any row order.

    Returns
    -------
    str
        SHA-256 of a canonical input/ID CSV sorted by row ID. Exact source
        bytes, including targets, are checked separately by the loader.
    """
    validate_identifiers(data)
    canonical = data.loc[:, [*FEATURE_COLUMNS, *ID_COLUMNS]].sort_values("row_id")
    text = canonical.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return sha256(text.encode("utf-8")).hexdigest()


def audit_data(data: pd.DataFrame) -> dict[str, object]:
    """Report source identity, measurements and grouping inconsistencies.

    Parameters
    ----------
    data : pandas.DataFrame
        Identified source observations returned by load_airfoil.

    Returns
    -------
    dict
        JSON-compatible duplicate counts, sweep sizes, thickness conflicts
        and global ranges. Target ranges describe the whole source, never
        final-test predictive performance.
    """
    validate_identifiers(data)
    _check_numeric(data, COLUMNS)
    groups = data.groupby("condition_id", sort=True)
    sizes = groups.size()
    conflicts = []
    for condition_id, observations in groups:
        thicknesses = sorted(observations["displacement_thickness_m"].unique().tolist())
        if len(thicknesses) != 1:
            conflicts.append({"condition_id": condition_id, "thicknesses_m": thicknesses})
    return {
        "source_sha256": data.attrs.get("source_sha256"),
        "row_id_scheme": ROW_ID_SCHEME,
        "condition_id_scheme": CONDITION_ID_SCHEME,
        "columns": list(COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "condition_columns": list(CONDITION_COLUMNS),
        "rows": len(data),
        "exact_duplicate_rows": int(data.duplicated(subset=list(COLUMNS)).sum()),
        "conditions": len(sizes),
        "smallest_sweep": int(sizes.min()),
        "largest_sweep": int(sizes.max()),
        "repeated_condition_frequency_rows": int(
            data.duplicated(subset=["condition_id", "frequency_hz"]).sum()
        ),
        "thickness_conflicts": conflicts,
        "ranges": {
            column: [float(data[column].min()), float(data[column].max())]
            for column in COLUMNS
        },
    }

