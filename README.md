# Airfoil self-noise prediction

Individual MMA3001 project. The engineering question is: **How accurately can
a model predict NACA 0012 airfoil self-noise across frequency at operating
conditions withheld from model development?**

The program accepts the five measured numerical inputs described in
[`data/README.md`](data/README.md) and estimates scaled sound pressure level
in decibels. Its intended use is a comparison of prediction methods within
the wind-tunnel conditions represented by the dataset. It is not a certified
aircraft noise estimator. Displacement thickness is an input measurement,
not something this project predicts.

## Status and evaluation design

This first version contains a reproducible dataset loader, an audit,
a fixed condition-level split, a development-only baseline comparison,
tests and HTML API documentation. It does **not** report final test accuracy.

Rows from one chord / velocity / angle combination belong to the same
frequency sweep. The fixed split keeps every sweep together: 84 conditions
(1,194 rows) in development, and 22 (309 rows) reserved for final evaluation.
All six chords and four speeds occur on both sides. It was generated with a
20% group holdout and random seed 3001. Development models use five-fold
grouped cross-validation. Do not use the final test labels for feature
selection, tuning or choosing the model.

The current comparison uses a training-mean predictor and a linear regression
with log10 frequency as a transparent starting point. Planned next work:
choose one nonlinear method covered in the unit, compare development-fold
MAE/RMSE in dB and runtime, analyse errors by condition, then evaluate the
selected model once on the held-out conditions. This repository does not yet
support prediction for unmeasured inputs in a production setting.

## Reproduce

Python 3.10 or later is required. From the repository root:

```bash
python -m venv .venv
# On Windows PowerShell: .venv\Scripts\Activate.ps1
# On macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
airfoil-noise audit
airfoil-noise check-split
airfoil-noise baseline-cv
python -m pytest -q
pdoc -o docs/api airfoil_noise airfoil_noise.__main__
```

The commands read `data/raw/airfoil_self_noise.dat` and
`data/splits/conditions.csv` relative to the repository. The data file and
split manifest are committed. The baseline command prints development-fold
metrics only. Run it at the repository root or supply `--data` and
`--manifest` paths to the commands that need them.

The generated HTML documentation is committed in [`docs/api/`](docs/api/).
Regenerate it after changing public functions. Unit and integration checks
are in `tests/`; the last test run is recorded in `docs/test-report.txt`.
Initial development-only baseline results are in
[`reports/development-baseline.md`](reports/development-baseline.md).

## Repository map

| Path | Purpose |
| --- | --- |
| `data/raw/` | Attributed UCI source file |
| `data/splits/` | Frozen condition-level holdout |
| `src/airfoil_noise/data.py` | Schema, units and input checks |
| `src/airfoil_noise/split.py` | Group assignment and manifest checks |
| `src/airfoil_noise/baseline.py` | Development-only grouped comparison |
| `tests/` | Data, leakage and baseline behaviour checks |
| `docs/api/` | Generated HTML function documentation |
| `reports/` | Development results and interpretation |

## Limitations and project record

The 106 conditions come from one experimental campaign. A group holdout
estimates performance on withheld combinations in this dataset; it does not
prove accuracy for a different airfoil, observer position or flow regime.
The raw UCI file has no column headers. The loader rejects missing, nonfinite,
malformed and physically invalid basic inputs. It leaves questions about
measurement uncertainty and dataset provenance for the report.

Keep an `AI_USE.md` record as the project develops. Identify tools, prompts or
tasks, how suggestions were checked, mistakes found and decisions you made.
For this initial scaffold, AI helped draft code and documentation; the
student must review the methods and independently explain the results before
submission. The report's AI reflection should describe actual use, not a
generic declaration.

Code and documentation: MIT licence in [`LICENSE`](LICENSE). Source data:
CC BY 4.0 with attribution in [`data/README.md`](data/README.md).
