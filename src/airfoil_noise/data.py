"""Read and audit the six-column UCI airfoil self-noise data file."""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = (
    "frequency_hz",
    "angle_deg",
    "chord_m",
    "velocity_m_s",
    "displacement_thickness_m",
    "spl_db",
)
"""Columns in the exact order of the source file, with units in names."""

FEATURE_COLUMNS = COLUMNS[:-1]
TARGET_COLUMN = COLUMNS[-1]


def load_airfoil(path: str | Path) -> pd.DataFrame:
    """Load a headerless, tab-separated airfoil observation file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the original six-column ``airfoil_self_noise.dat`` file.
        Inputs are frequency (Hz), angle (degrees), chord (m), velocity (m/s),
        and suction-side displacement thickness (m). The last column is
        scaled sound pressure level (dB).

    Returns
    -------
    pandas.DataFrame
        Numeric observations with the names in :data:`COLUMNS`, in original
        row order. No imputation, transformation or split is performed.

    Raises
    ------
    FileNotFoundError
        The data file does not exist.
    ValueError
        The file has the wrong number of columns, an empty dataset, missing
        or nonfinite values, or nonpositive frequency, chord, velocity or
        displacement thickness.

    Notes
    -----
    Negative angles are allowed by the loader, although none appear in the
    supplied dataset. A valid file format does not prove a measurement is
    physically accurate or within the training domain.
    """
    raw = pd.read_csv(path, sep="\t", header=None)
    if raw.empty or raw.shape[1] != len(COLUMNS):
        raise ValueError(f"Expected nonempty data with {len(COLUMNS)} columns.")
    raw.columns = list(COLUMNS)

    try:
        data = raw.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Every airfoil column must be numeric.") from exc

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite observations are unsupported.")

    positive = (
        "frequency_hz",
        "chord_m",
        "velocity_m_s",
        "displacement_thickness_m",
    )
    if (data.loc[:, list(positive)] <= 0).any().any():
        raise ValueError("Frequency, chord, velocity and thickness must be positive.")
    return data


def audit_data(data: pd.DataFrame) -> dict[str, object]:
    """Summarise dataset size, repeated rows and observed input ranges.

    Parameters
    ----------
    data : pandas.DataFrame
        A table returned by :func:`load_airfoil`.

    Returns
    -------
    dict
        Row count, duplicate-row count, number of operating conditions,
        frequency measurements per condition, and min/max for each column.
        A condition is one chord / velocity / angle combination.
    """
    condition_sizes = data.groupby(
        ["chord_m", "velocity_m_s", "angle_deg"], sort=True
    ).size()
    return {
        "rows": len(data),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "conditions": len(condition_sizes),
        "smallest_sweep": int(condition_sizes.min()),
        "largest_sweep": int(condition_sizes.max()),
        "ranges": {
            column: (float(data[column].min()), float(data[column].max()))
            for column in COLUMNS
        },
    }

