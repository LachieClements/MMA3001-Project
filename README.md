# NACA 0012 Aerofoil Self-Noise

MMA3001 Numerical Methods and Machine Learning — Semester 2, 2026
Lachie Clements, 33134529

## Engineering question

How accurately can different regression methods predict airfoil self-noise across
frequency for operating conditions withheld from model development, and what
trade-offs exist between accuracy, training cost and prediction speed?

The output is scaled sound pressure level in dB. The five supplied measurements
are frequency (Hz), angle of attack (degrees), chord (m), free-stream velocity
(m/s) and suction-side displacement thickness (m). Thickness must already be
known: this project does not independently calculate boundary-layer thickness.
See [data definitions and attribution](data/README.md).

The comparison concerns the experimental domain represented by this dataset.
It cannot establish prediction quality for other airfoils, observer positions,
measurement arrangements or unsupported flow regimes.

## Current status: Milestone 1

The raw source is unchanged and passes the recorded SHA-256 check. It contains
1,503 observations, 106 chord–velocity–angle conditions, no exact duplicates and
8–19 distinct frequencies per condition. Thickness is constant within each
condition. The [saved audit](reports/data-audit.json) records these checks.

Following approval of the design, seed **3001** assigns **84 conditions /
1,194 rows to development** and **22 conditions / 309 rows to final testing**.
The [frozen bundle](data/splits/README.md) also records five development folds.
Validation row counts are 241, 241, 240, 240 and 232.

[Input coverage](reports/split-coverage.csv) describes both partitions without
using held-out targets. Every holdout input lies within its corresponding
development feature range; joint combinations and distributions still differ.
Range overlap is not evidence of physical extrapolation capability.

No new model comparison or final-test evaluation has been run. Earlier starter
work remains in history. Its direct Python baseline utility is retained, but
the legacy baseline-cv command is paused: its log-frequency transform and fold
regeneration must be replaced in Milestone 2.

## Setup and checks

Use Python 3.10 or later. From the repository root:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
airfoil-noise --help
airfoil-noise audit
airfoil-noise check-split
python -m airfoil_noise audit
python -m airfoil_noise check-split
python -m pytest -q
pdoc -o docs/api airfoil_noise airfoil_noise.__main__
~~~

The console entry point airfoil-noise offers the same commands as
python -m airfoil_noise. [docs/environment.txt](docs/environment.txt) identifies
the environment actually used for milestone checks. Dependency ranges are not
a complete dependency lock. Install the package before running its commands or
tests; no PYTHONPATH setting or source-path injection is needed. The subprocess
entry-point test runs from outside the checkout and removes any inherited
PYTHONPATH setting.

The repository attributes preserve the original raw .dat bytes and enforce LF
checkout endings for data/splits/*.csv, including when core.autocrlf=true.
Do not normalise the raw file: its approved checksum includes CRLF endings.
The [reproducibility report](docs/reproducibility-report.txt) records the isolated
Git checkout and fresh editable-install checks. Git configuration changes and
fixture commits for that check are confined to temporary repositories.

The complete pytest suite includes synthetic unit/CLI tests, real-source/bundle
integration checks and the retained legacy baseline regression test. A quick
synthetic-only check of Milestone 1 is:

~~~powershell
python -m pytest tests/test_data.py tests/test_split.py tests/test_cli.py -q -k "not source_file and not committed"
~~~

Tests use temporary outputs. Neither test command runs a final evaluation or
a tuning study. [docs/test-report.txt](docs/test-report.txt) contains actual
executed results. Passing tests verifies software behaviour, not physical or
predictive validity.

## Frozen assignment workflow

A normal checkout already includes assignments. Use check-split to validate
them; do not recreate them during model development.

The original generation commands, shown for provenance, were:

~~~powershell
python -m airfoil_noise audit --output reports/data-audit.json
python -m airfoil_noise make-split --seed 3001 --test-fraction 0.2 --folds 5
python -m airfoil_noise check-split --coverage-output reports/split-coverage.csv
~~~

These output-writing commands refuse to replace existing files. There is no
force option. If a justified design revision is approved, use a distinct
directory and record why; do not search seeds for favourable performance.

The loader appends row_id and condition_id after the six measurement columns.
Keep these IDs when selecting or reordering rows. Use feature_matrix to obtain
the original five features in the correct order. IDs and targets are excluded.

Future development evaluation should obtain only development rows and saved
row-fold assignments through load_development. The source digest includes all
original bytes; the separate input fingerprint checks row/input correspondence
without inspecting in-memory held-out targets.

## Remaining milestones

1. Verify training-mean and raw-input linear baselines through shared metrics,
   saved predictions and the frozen folds.
2. Add tree, scaled RBF SVR and small scaled MLP pipelines and predefined
   configurations. Initially disable automatic MLP early stopping.
3. Run grouped development comparisons, tuning, seed sensitivity and separate
   fit/prediction/search cost measurements. Select using equal-condition MAE,
   supported by pooled metrics, diagnostics and engineering trade-offs.
4. Present frozen configurations, rationale and the exact final procedure for
   student approval before opening final-test results.
5. Complete final evaluation, figures, report support, HTML documentation and
   a fresh-environment reproduction/submission check.

## Repository and assessment

| Path | Responsibility |
| --- | --- |
| data/raw/ | Original attributed source bytes |
| data/splits/ | Frozen condition/fold CSVs and provenance metadata |
| src/airfoil_noise/ | Reusable data/split code and retained starter utilities |
| tests/ | Software verification |
| reports/ | Saved audit and input coverage; future experiment records |
| notebooks/ | Future investigation calling package functions |
| docs/api/ | Generated HTML API documentation |
| AI_USE.md | Actual AI assistance, checks, errors and student decisions |

The supplied brief requires functioning code on main, README, comments and
consistent docstrings, HTML documentation, licensing, test functions and an
executed test report. It also requires an individual 5–10 page report
(references/appendices excluded from the 10-page limit), accessible final
repository link, zipped clone via Moodle and a 5-minute presentation plus
3 minutes of questions in Week 14. Obtain dates from the unit overview and
appointment schedule. The brief does not explicitly require a repository rename
or public visibility.

Keep meaningful development history; commits and remote changes require the
student's explicit approval. Teaching notebooks and the assessment brief are
reference material and are not copied into this repository.

Code/documentation use the MIT [licence](LICENSE); source data retains CC BY 4.0
attribution. Student review and explanation of AI-assisted work remain necessary.
Record actual oversight in [AI_USE.md](AI_USE.md), without representing automated
checks as independent student verification.
