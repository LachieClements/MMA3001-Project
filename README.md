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

## Current status: Milestone 2

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

The training-mean predictor and multiple linear regression now share one
development evaluator using these saved folds. Both use the original five
inputs; linear regression has an intercept, without scaling, log-frequency or
added features. The mean predictor averages training observations. Neither
method needs a model seed. The legacy transform/fold-regeneration path has
been removed. No final-test predictions are produced by this workflow.

On the frozen development folds, equal-condition MAE is **5.680 dB** for the
training mean and **3.771 dB** for linear regression (pooled RMSE: 6.944 and
4.855 dB respectively). These are development comparisons, not final-test
estimates. Full results and limitations are in the interpretation linked below.

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

The complete pytest suite includes synthetic unit/CLI tests and real-source/
bundle integration checks. A quick synthetic check of the development evaluator
is:

~~~powershell
python -m pytest tests/test_models.py tests/test_metrics.py tests/test_validation.py tests/test_baseline.py tests/test_cli.py -q -ra
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

Development evaluation obtains only development rows and saved
row-fold assignments through load_development. The source digest includes all
original bytes; the separate input fingerprint checks row/input correspondence
without inspecting in-memory held-out targets.

## Baseline development workflow

After installation and checks, reproduce the two fixed baselines in a **new**
directory (the recorded milestone directory already exists):

~~~powershell
airfoil-noise baseline-cv --directory data/splits --output-dir reports/development/baselines-reproduction
~~~

There are two configurations, five fits each, with no search. A fresh estimator
is constructed per fold. Only development targets enter the evaluator. Overall
metrics use all out-of-fold predictions; every development row is predicted
exactly once per model. The primary score is

\[
\mathrm{MAE}_{\mathrm{condition}} = \frac{1}{G}\sum_{g=1}^{G}
\left(\frac{1}{n_g}\sum_{i=1}^{n_g}|y_{gi}-\hat y_{gi}|\right).
\]

This gives each operating condition equal weight despite unequal frequency
counts. It is calculated over all conditions, rather than by averaging the
five fold scores equally. Pooled MAE/RMSE weight every observation equally;
R² is supplementary and can be negative. R² is undefined for fewer than two
observations or exactly constant truth, even with perfect predictions. It is
NaN internally, blank in CSV and null in JSON, with an explicit reason.

Each run saves predictions.csv (model, row/condition IDs, fold, measured and
predicted dB), summary.csv, condition_metrics.csv and fold_metrics.csv.
diagnostics.json retains separate fit/prediction seconds, warnings, failures
and fitted coefficients/rank or mean. run.json records every estimator setting,
feature order, environment/thread details and the frozen bundle's checksums,
seed and assignment metadata. Fit/predict timings are single measurements,
without warmup or repetitions: they are preliminary diagnostics, not the final
speed comparison. No peak-memory or FLOP claim is made.

Existing run directories are refused before fitting. Failed fits retain
diagnostics and any partial predictions, mark the run failed and produce no
summary tables; the CLI raises an error. Use a distinct directory for a quick
check so it cannot replace a study. Figures and later report tables should
read saved predictions rather than repeat fits.

See the [recorded baseline interpretation](reports/development/milestone-2-baselines/interpretation.md)
for the development results. Cross-validation here is development evidence,
not an independent final performance estimate. It does not establish an
acceptable engineering error threshold or performance beyond this dataset.

## Remaining milestones

1. Add tree, scaled RBF SVR and small scaled MLP pipelines and predefined
   configurations. Initially disable automatic MLP early stopping.
2. Run grouped development comparisons, tuning, seed sensitivity and separate
   fit/prediction/search cost measurements. Select using equal-condition MAE,
   supported by pooled metrics, diagnostics and engineering trade-offs.
3. Present frozen configurations, rationale and the exact final procedure for
   student approval before opening final-test results.
4. Complete final evaluation, figures, report support, HTML documentation and
   a fresh-environment reproduction/submission check.

## Repository and assessment

| Path | Responsibility |
| --- | --- |
| data/raw/ | Original attributed source bytes |
| data/splits/ | Frozen condition/fold CSVs and provenance metadata |
| src/airfoil_noise/ | Data/split checks, estimator builders, metrics and shared validation |
| tests/ | Software verification |
| reports/ | Saved audit, input coverage and traceable development records |
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
