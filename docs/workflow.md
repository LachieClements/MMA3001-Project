# Detailed workflow and milestone history

[Back to the project README](../README.md)

This guide retains the technical workflow and milestone detail previously in the
README. Start with the README for installation, routine verification, principal
results and HTML viewing instructions. Run all commands below from the
repository root after installation.

Milestone-specific descriptions explain the development sequence; the approved
final evaluation is complete. Historical experiment records and settings are
retained, and saved-record reporting recreates outputs without repeating fits.

## Contents

- [Dataset and early milestone history](#dataset-and-early-milestone-history)
- [Setup and checks](#setup-and-checks)
- [Frozen assignment workflow](#frozen-assignment-workflow)
- [Baseline development workflow](#baseline-development-workflow)
- [Configurable models and records](#configurable-models-and-records)
- [Milestone 4 development study](#milestone-4-development-study)
- [Milestone 5 completed final evaluation](#milestone-5-approved-final-evaluation-completed)
- [Submission and final review](#submission-and-final-review)
- [Repository and assessment](#repository-and-assessment)

## Dataset and early milestone history

The raw source is unchanged and passes the recorded SHA-256 check. It contains
1,503 observations, 106 chord–velocity–angle conditions, no exact duplicates and
8–19 distinct frequencies per condition. Thickness is constant within each
condition. The [saved audit](../reports/data-audit.json) records these checks.

Following approval of the design, seed **3001** assigns **84 conditions /
1,194 rows to development** and **22 conditions / 309 rows to final testing**.
The [frozen bundle](../data/splits/README.md) also records five development folds.
Validation row counts are 241, 241, 240, 240 and 232.

[Input coverage](../reports/split-coverage.csv) describes both partitions without
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

Milestone 3 added decision tree, scaled RBF SVR and small scaled MLP pipelines
to the same evaluator. Milestone 4 uses the approved bounded development
search and separate repeated benchmarks described below. The reviewed
Milestone 2 records remain unchanged. The separately approved final holdout run
is complete; its results and warning are recorded below.

## Setup and checks

Use Python 3.10 or later. The declared package range is scikit-learn
**>=1.7,<2**: the saved configurations include LinearRegression.tol and
MLPRegressor.loss, which require version 1.7 or later. This range permits
installation; it does not certify identical results across every allowed
release. The recorded experiments used **scikit-learn 1.7.2**.

For routine installation and software checks, from the repository root:

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

For reproduction of the recorded model configurations, start with a **new**
virtual environment and explicitly constrain scikit-learn to its recorded
version in the installation command:

~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]" "scikit-learn==1.7.2"
python -m pip check
airfoil-noise --help
airfoil-noise audit
airfoil-noise check-split
python -m pytest -q -ra
~~~

Use a different environment directory if .venv already exists. These checks
audit the source and saved assignments and run synthetic model tests; they
do not rerun the real-data development search, benchmarks or final evaluation.
The exact original Python and remaining package versions are recorded in
[docs/environment.txt](environment.txt). Pinning scikit-learn alone is
not a complete environment lock and does not guarantee bit-for-bit numerical
or timing reproduction on another machine. Older saved constructor records
remain immutable; incompatible classes/parameter schemas fail explicitly.

The console entry point airfoil-noise offers the same commands as
python -m airfoil_noise. [docs/environment.txt](environment.txt) identifies
the environment actually used for milestone checks. Dependency ranges are not
a complete dependency lock. Install the package before running its commands or
tests; no PYTHONPATH setting or source-path injection is needed. The subprocess
entry-point test runs from outside the checkout and removes any inherited
PYTHONPATH setting.

The repository attributes preserve the original raw .dat bytes and enforce LF
checkout endings for data/splits/*.csv, including when core.autocrlf=true.
Do not normalise the raw file: its approved checksum includes CRLF endings.
The [reproducibility report](reproducibility-report.txt) records the isolated
Git checkout and fresh editable-install checks. Git configuration changes and
fixture commits for that check are confined to temporary repositories.

The complete pytest suite includes synthetic unit/CLI tests and real-source/
bundle integration checks. A quick synthetic check of the development evaluator
is:

~~~powershell
python -m pytest tests/test_models.py tests/test_metrics.py tests/test_validation.py tests/test_baseline.py tests/test_cli.py -q -ra
~~~

Tests use temporary outputs. Neither test command runs a final evaluation or
a tuning study. [docs/test-report.txt](test-report.txt) contains actual
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
and fitted diagnostics. run.json records every estimator setting,
feature order, environment/thread details and the frozen bundle's checksums,
seed and assignment metadata. Fit/predict timings are single measurements,
without warmup or repetitions: they are preliminary diagnostics, not the final
speed comparison. No peak-memory or FLOP claim is made.

Existing run directories are refused before fitting. Failed fits retain
diagnostics and any partial predictions, mark the run failed and produce no
summary tables; the CLI raises an error. Use a distinct directory for a quick
check so it cannot replace a study. Figures and later report tables should
read saved predictions rather than repeat fits.

See the [recorded baseline interpretation](../reports/development/milestone-2-baselines/interpretation.md)
for the development results. Cross-validation here is development evidence,
not an independent final performance estimate. It does not establish an
acceptable engineering error threshold or performance beyond this dataset.

## Configurable models and records

build_model now accepts parameters (regressor parameter overrides) and seed.
Mean and linear defaults remain unchanged. SVR and MLP have independent
StandardScaler and regressor steps, fitted together inside each training fold.
There are still only five original features, and no target scaling. The
baseline-cv CLI still explicitly runs only the two reviewed baselines.

| Method | Initial construction settings | Preprocessing / seed |
| --- | --- | --- |
| training_mean | strategy="mean" | None / not applicable |
| linear | Intercept, ordinary least squares, n_jobs=1 | None / not applicable |
| tree | max_depth=5, min_samples_leaf=5 | None / random_state=17 |
| svr | RBF, C=10, epsilon=0.1 dB, gamma="scale" | StandardScaler / not applicable |
| mlp | (20,), ReLU, Adam, alpha=0.0001, learning_rate_init=0.001, max_iter=2000, tol=0.0001, n_iter_no_change=20 | StandardScaler / random_state=17 |

These are explicit starting settings, not tuned or selected results. Every
sklearn constructor parameter, including defaults not shown in the table, is
saved in configuration records. Model seed 17 was chosen before new model
comparisons and is independent of holdout seed 3001. Deterministic methods
reject an unused model seed; tree/MLP use the explicit seed or the documented
default. Parameters use regressor names such as C or max_depth, not pipeline
prefixes; preprocessing cannot be replaced through those overrides.

For the small MLP, automatic early_stopping stays False and warm_start stays
False. No row-based internal validation set is used; validation_fraction is
recorded as a constructor default but is inactive. Training-loss tolerance and
n_iter_no_change can still end training before max_iter. This differs from
stopping on a grouped validation set, which is not implemented here. Hidden
layers are limited to one or two layers of at most 64 neurons each, with Adam
as the supported optimiser. Tree sample-count settings use integers. Common
tuning ranges and study controls are checked before fitting; sklearn validates
remaining solver-specific constraints during fit, with failures recorded.

To supply configurations to the existing evaluate_development function, use
unique IDs mapped to model, optional parameters and optional seed:

~~~python
configurations = {
    "tree_depth3": {"model": "tree", "parameters": {"max_depth": 3}, "seed": 17},
    "svr_initial": {"model": "svr", "parameters": {"C": 10.0}},
    "mlp_seed17": {"model": "mlp", "parameters": {"hidden_layer_sizes": (20,)}, "seed": 17},
}
# development and row_folds must come from load_development and the saved bundle.
# results = evaluate_development(development, row_folds, configurations=configurations)
~~~

Alternatively, model_names selects distinct methods at their construction
defaults. Omitting both options preserves the two-baseline workflow. The two
options cannot be combined. Configurations are checked before any fold fit,
then cloned per fold so nested scalers/regressors share no fitted state. The
model column in predictions and metric tables is the configuration ID; run.json
maps it to the underlying method. The evaluator does not generate, search or
select configurations. The Milestone 4 runner expands only the approved explicit grid and invokes
this evaluator once per configuration/seed. These tuned comparisons are
development evidence rather than independent final validation.

New run records use schema version 2. They contain per-configuration methods,
actual preprocessing and random states rather than a blanket "no transforms"
or "no seeds" claim. Constructor records use their own version 1 schema:
qualified estimator_class and complete parameters, recursively including
pipeline steps and typed tuples. They contain settings, not fitted weights.
For example, after installation this round trip performs no fitting:

~~~python
import json
from airfoil_noise.models import build_model, model_configuration, model_from_configuration

original = build_model("mlp", parameters={"hidden_layer_sizes": (20,)}, seed=17)
record = json.loads(json.dumps(model_configuration(original), allow_nan=False))
restored = model_from_configuration(record)
assert model_configuration(restored) == record
~~~

To reconstruct a saved run's configuration, pass run["models"][configuration_id]
to model_from_configuration. Reconstruction creates fresh, unfitted objects
from the seven supported sklearn classes; unknown classes, opaque objects,
incomplete parameters and incompatible schemas/versions fail explicitly.
The original Milestone 2 mean/linear records remain readable. Exact numerical
reproduction also requires the recorded inputs, fold assignments, seeds and
compatible package environment; configuration JSON is not a saved fitted model.

Nested diagnostics retain scaler mean/variance/scale and training-row count;
tree node/leaf counts and depth; SVR support-vector counts, iterations and fit
status; and MLP iterations, parameter count, final/best loss and loss history.
Warnings remain attached to each fit, with an explicit convergence-warning
flag. Finite predictions with a convergence warning remain traceable successful
fits; this does not claim that optimisation converged. Loss-history length and
final loss are checked, without requiring monotone loss or a particular model
ranking. For Adam, MLP n_iter_ counts training epochs; it is not directly
comparable to an SVR solver iteration. Complexity counts describe different structures and are not equivalent
cost units. JSON is checked before output creation so unsupported objects cannot
leave partially written metric tables after a serialization failure.

## Milestone 4 development study

The [approved protocol](milestone-4-protocol.md) defines every search value,
fixed setting, seed, failure rule, tie rule and timing loop. The machine-readable
[configuration](../configs/milestone4.json) expands to 35 parameter configurations,
47 configuration/seed variants and 235 grouped CV fits. All use the saved folds.
MLP uses seeds 17, 29 and 43; its selection score averages the three overall OOF
equal-condition MAEs. Seed 17 represents the chosen MLP in speed measurements;
it is not a best-seed choice or an ensemble. No automatic early stopping is used.

Each variant is saved separately. Failures cannot erase other variants; an
incomplete variant is ineligible, and an MLP configuration needs all three
seeds. Finite complete results with ConvergenceWarnings remain eligible but are
labelled warning-qualified. Converged-only sensitivity requires all 15 MLP fits
to complete without such warnings. There are no replacement fits or expanded
searches after results. Training predictions carry both their saved fold and
fit_fold, appear four times per observation, and never enter selection.

The runner freezes one choice per eligible family and the accuracy leader
before benchmarks. The benchmark uses one warmup plus five measured fresh fits
per family, with identical representative seeds and rotating method order.
There are at most 30 additional fits, or 265 real-data fits in total. Search
cost, selected-model training cost, 256-row batch time and separately dispatched
one-row latency are distinct records. The same 32 prepared one-row frames cycle
in every single-observation round. Warmup, calibration and 20 measured blocks
have explicit call counts. Medians and IQRs describe block mean seconds per
call, not an individual-call p95. Scaling is included in pipeline timings.
Serialized protocol-5 model size is recorded; peak RAM and FLOPs are not measured.

To reproduce intentionally, use **new directories** and the same recorded
package versions. These commands launch the full study, not routine checks:

~~~powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
python -m airfoil_noise development-study --protocol configs/milestone4.json --directory data/splits --output-dir reports/development/search-reproduction
python -m airfoil_noise benchmark-development --protocol configs/milestone4.json --directory data/splits --selection reports/development/search-reproduction/selected_configurations.json --output-dir reports/development/benchmarks-reproduction
python -m airfoil_noise plot-development --search-dir reports/development/search-reproduction --benchmark-dir reports/development/benchmarks-reproduction --output-dir reports/development/figures-reproduction
~~~

Process-local thread variables are required before Python imports. The runner
also applies threadpool_limits(1) and checks loaded libraries. Baseline-cv keeps
its original two-model purpose. Audit/check-split never fit; final-evaluate is a separate explicit command
that requires separate approval before use. Synthetic tests use separately labelled small protocols and
never change the real grid's 2000-epoch MLP limit.

Recorded outputs:

- [Search and selected configurations](../reports/development/milestone-4-search/selected_configurations.json),
  per-variant OOF/training predictions, diagnostics and full pipeline records.
- [Repeated benchmarks](../reports/development/milestone-4-benchmarks/summary.csv),
  raw fit records, prediction blocks and prepared-row identifiers.
- [Development interpretation](../reports/development/milestone-4-search/interpretation.md)
  and [engineering recommendation](../reports/development/milestone-4-benchmarks/recommendation.json).
- [Reviewed figure manifest](../reports/development/milestone-4-figures/reviewed/manifest.json), PNG and
  SVG accuracy/time, complexity, convergence, sensitivity and all-condition plots.

The frozen SVR (C=100, epsilon=0.1, gamma=0.2) has development equal-condition
MAE **2.338 dB**; the tree has **2.850 dB**. MLP's three-seed score is **3.364 dB**
and remains warning-qualified. All 265 approved real-data fits completed with
zero failures; 78/90 MLP CV fits warned and no all-15-fits warning-free MLP
configuration qualified. Full tests: **306 passed**. SVR is recommended after
benchmarking, with its slower batch prediction and 10.502 dB worst-condition
MAE explicitly acknowledged. See the interpretation for timing IQRs and limits.

Original figures are retained; reviewed/ fixes overlapping labels without
repeating fits or timings. Later Milestone 5 additions extend CLI/plotting and
add a final module; the M4 model/development-evaluation modules and historical
source-hash records remain unchanged.

The engineering recommendation considers cost, accuracy, warnings and complexity
**after** benchmarks and was frozen before final evaluation. It does
not alter the previously selected configurations. Development tuning estimates
can be optimistic; no acceptable dB threshold or extrapolation claim is inferred.

## Milestone 5: approved final evaluation completed

The [final-evaluation protocol](milestone-5-protocol.md) fixes the selected
mean, linear, tree, scaled RBF SVR and scaled MLP configurations. SVR stays the
primary engineering recommendation. The complete settings are reconstructed
from the immutable M4 selection record, with file and canonical configuration
hashes checked. MLP uses predetermined seed **17**, not the best seed or an
ensemble; automatic early stopping remains disabled.

Only the explicit final-evaluate command can invoke this path. The approved
real experiment completed **five fresh fits** on all **1,194 development rows**,
five batch predict calls over all **309 held-out rows / 22 conditions**, and **1,545
prediction records**. No CV, new features, tuning, alternate seeds, replacement
fits or benchmark repetitions occurred. The inference function rejects holdout
targets; truth is joined by validated row IDs after prediction. Every pipeline
fits its scaler only on development inputs.

Implementation checks use small labelled synthetic fixtures: **50 focused tests**
and **340 complete-suite tests passed** before the real evaluation. Those tests
did not evaluate the real holdout. Run
installed tests with the documented scikit-learn **1.7.2** environment:

~~~powershell
python -m pytest tests/test_final_evaluation.py tests/test_cli.py tests/test_plotting.py -q -ra
python -m pytest -q -ra
~~~

The following command was executed **once** after separate approval on
24 October 2026. Its output directory is protected; do not rerun the evaluation
or change settings after inspecting the results:

~~~powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
python -m airfoil_noise final-evaluate --data data/raw/airfoil_self_noise.dat --directory data/splits --selection reports/development/milestone-4-search/selected_configurations.json --recommendation reports/development/milestone-4-benchmarks/recommendation.json --output-dir reports/final/milestone-5-frozen-models
~~~

The new directory refuses overwrite before fitting. It saves run metadata,
source/split/configuration hashes, complete settings/seeds, environment/command,
original-byte frozen-choice snapshots, holdout inputs, predictions and diagnostics.
Metric tables and figures in report/ are generated from reloaded predictions.csv.
completion.json is written last only after all five models and saved records
validate and the report succeeds. Finite convergence-warning results remain
warning-qualified. A failed model leaves partial records, a failed status and
no comparison report or completion marker; failures are never silently retried.

The report contains the primary equal-condition MAE, pooled MAE/RMSE/R2 with
existing undefined-value handling, all 110 model/condition error rows, and
stage-labelled representative OOF/final comparisons. MLP's seed-17 OOF score
is distinguished from its three-seed configuration-selection score. Figures
show every observation/condition: measured versus predicted, residuals versus
frequency, an all-condition error heatmap and frequency sweeps for all22
conditions in fixed ID order, paginated eleven per page.

To regenerate reports into a NEW directory using saved records only:

~~~powershell
python -m airfoil_noise report-final --run-dir reports/final/milestone-5-frozen-models --output-dir reports/final/milestone-5-report-reproduction
~~~

This command validates saved checksums/coverage and makes **zero estimator
construction, fit or predict calls**. The Python functions regenerate_final_report
and plot_final support the same separation. Original raw data or fitted model
files are not needed to regenerate reporting. The completed
[final summary](../reports/final/milestone-5-frozen-models/report/summary.csv),
[all condition errors](../reports/final/milestone-5-frozen-models/report/condition_metrics.csv)
and [figures](../reports/final/milestone-5-frozen-models/report/figures/) come from
the protected saved predictions.

The [results narrative](milestone-5-results.md) interprets the final errors
alongside the saved M4 cost evidence. Those batch timings used **256 observations**;
the **32-observation** pool was cycled through separate individual prediction calls.

Final equal-condition MAE is **5.398 dB** (mean), **3.739 dB** (linear),
**2.254 dB** (tree), **2.503 dB** (SVR) and **3.066 dB** (MLP).
The tree has the lowest observed holdout error; SVR remains the primary
recommendation fixed using development evidence. SVR's worst condition MAE is
**11.876 dB**, so its average is not a uniform error guarantee. MLP seed 17
reached 2,000 iterations with a ConvergenceWarning. Its finite complete
predictions remain explicitly warning-qualified; no model failed or was retried.
The completion status is success_warning_qualified. Fit/predict durations in
this run are diagnostics; the prior repeated M4 benchmarks remain the cost
evidence. These final results do not establish performance for other airfoils
or measurement arrangements, or an acceptable engineering error threshold.

After holdout exposure, configurations, seeds and primary SVR recommendation
stay fixed even if another model has lower test error. Unexpected results are
reported honestly. No acceptable engineering dB threshold or extrapolation
claim is inferred.

## Submission and final review

The implementation and approved final evaluation are complete. Review the saved
predictions, diagnostics, tables and figures while preparing the student report,
presentation and submission package. Reproduce reporting from saved records,
without repeating the final fits or changing the frozen settings. An exploratory
notebook is optional; reusable computational logic belongs in the package.

Future repository changes should have a concrete, agreed scope before a commit
or push. The README reorganisation and this guide were explicitly requested.

## Repository and assessment

| Path | Responsibility |
| --- | --- |
| data/raw/ | Original attributed source bytes |
| data/splits/ | Frozen condition/fold CSVs and provenance metadata |
| src/airfoil_noise/ | Data/split checks, estimator builders, metrics and shared validation |
| tests/ | Software verification |
| reports/ | Saved audit, input coverage and traceable development/final records |
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

Code/documentation use the MIT [licence](../LICENSE); source data retains CC BY 4.0
attribution. Student review and explanation of AI-assisted work remain necessary.
Record actual oversight in [AI_USE.md](../AI_USE.md), without representing automated
checks as independent student verification.
