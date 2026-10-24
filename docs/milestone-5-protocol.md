# Milestone 5 protocol and execution record

**Status update — 24 October 2026:** the student separately approved one final
holdout execution after the checkout-integrity and full-suite checks passed.
The run completed at **00:40:44 UTC (11:40:44 AEDT)** with status
**success_warning_qualified**: five fits, five batch prediction calls and
1,545 prediction records. MLP seed 17 retained its convergence warning; SVR
remains the preselected primary recommendation. No retries or tuning occurred.

The procedure below was reviewed before that approval and is preserved as the
pre-run protocol. Its approval-boundary wording records that earlier stage;
the dated execution record at the end documents the later approval/completion.

SVR remains the primary engineering recommendation selected before final evaluation.
The final table may show another method with lower observed holdout error; that
fact will be reported honestly without switching the primary recommendation.

## Frozen inputs to the proposed evaluation

Read the complete configuration records from:
- reports/development/milestone-4-search/selected_configurations.json
  SHA256: 93a740f1111ee554c67fc54f246cfc41f2d49216e5fad7dff00a1f913dad7d2e
- reports/development/milestone-4-benchmarks/recommendation.json
  SHA256: 75afe0565d0314d8a9f284fa7f96582a1970adf1745e4951f9a00e3b77258a8c

| Family | Selected ID | Relevant settings | Model seed |
| --- | --- | --- | --- |
| Training mean | training_mean_01 | strategy=mean, training-observation mean | none |
| Linear | linear_01 | OLS/intercept, original five inputs | none |
| Tree | tree_07 | max_depth=None, min_samples_leaf=1 | 17 |
| SVR (primary) | svr_14 | StandardScaler; RBF; C=100, epsilon=0.1 dB, gamma=0.2 | none |
| MLP | mlp_06 | StandardScaler; (20,10); alpha=0.01; max_iter=2000 | 17 |

All remaining constructor settings come verbatim from saved records through
model_from_configuration, not build_model defaults, search grids or a new seed
choice. MLP keeps early_stopping=False, warm_start=False, Adam/ReLU and the saved
optimisation controls. It is one seed-17 model, not an ensemble.

Canonical configuration hashes (sorted compact JSON, UTF-8, allow_nan=False):
- mean: 8207df3c3fc204d71b91e7039541845039422cddde5d2af65f42d70673b4b8b6
- linear: 528ec3949ab9dcc0941bc4a5c16f3f603b921b0c94ba771fc16ba9d5265caa04
- tree: bfe382800ab16898a39e6ca119a92c212c3ce70b815dedc52ca0eb7be11c27c2
- SVR: 3e35b4c034c9bd7ee7ea4d9888e68bf6e2fd3a120aa35493b777c3debb0fc6f0
- MLP: c340318227047df9d71ce8ef07c39f23c936b7a534ad8cc3d853b95a2fb76f6a

These approved hashes are checked by read_frozen_choices before the real CLI
can load source observations. Synthetic fixtures supply their own distinct
hashes through the library API; the real CLI exposes no hash override.

## Implementation scope

New:
- src/airfoil_noise/final_evaluation.py: frozen reconstruction, separated fit/predict,
  protected record writing and metric calculation from saved predictions.
- tests/test_final_evaluation.py: synthetic final-workflow verification.
- docs/milestone-5-protocol.md: the approved procedure and recorded commands.

Extend existing files:
- src/airfoil_noise/__main__.py: one explicit final-evaluate command; no default run.
- src/airfoil_noise/plotting.py: final figures read saved records without fitting.
- tests/test_cli.py and tests/test_plotting.py: explicit-command and record-derived
  plot checks.
- README.md, AI_USE.md, docs/environment.txt, docs/test-report.txt: actual outcomes
  and reproduction instructions, preserving historical entries.
- docs/api/: regenerate API HTML/search/index, including final_evaluation.

Only new experiment output:
- reports/final/milestone-5-frozen-models/, with figures/ inside that directory.

No edits to data.py, split.py, models.py, metrics.py, validation.py,
development.py, benchmarking.py, configs/milestone4.json, raw data, frozen
assignments, development selections/recommendation or prior experiment outputs.
No new feature engineering, stopping procedure, search or benchmarking framework.

## Execution and approval boundary

1. Implement the proposed module/CLI/plots and synthetic tests. Do not touch
   final targets during implementation verification.
2. Run synthetic focused checks and the complete suite in the installed
   scikit-learn 1.7.2 environment. Data/split integration checks remain read-only.
3. Show the unchanged frozen file/configuration hashes, primary SVR recommendation,
   passing synthetic checks and exact final command before any real holdout run.
   Approval to implement does not by itself authorise a holdout run unless the
   user's approval explicitly includes it.
4. On approval to evaluate, refuse the output directory if it exists; there is
   no --force option. Preflight source SHA256, frozen CSV/metadata checksums,
   input/development fingerprints, feature order, row/condition counts and
   disjointness, selection/recommendation/configuration hashes, installed version
   and thread settings. Stop before fitting if any control fails.
5. Partition using the existing frozen manifest. Give the fit/predict function
   only the 1,194 development observations/targets and holdout inputs/IDs.
   Do not pass held-out targets to that function. Use the original five features
   in their saved order; IDs and targets are never features.
6. Reconstruct five independent, unfitted complete estimators/pipelines. Fit each
   once on all 1,194 development rows; preprocessing learns only from development.
   Call predict once per estimator on all 309 held-out input rows (22 conditions).
   Planned budget: exactly five fresh fits, five batch predict calls and
   5 x 309 = 1,545 prediction records. There is no CV, tuning, alternate seed,
   ensemble, replacement fit, benchmark repetition or extra training prediction.
7. Attach truth by validated row_id/condition_id alignment, save predictions once,
   reload with round-trip numeric parsing, validate completeness and compute all
   reported metrics from those saved records using existing metrics functions.
8. Generate predeclared tables/figures from saved records, validate their hashes
   and archive a complete review bundle without overwriting existing runs.


Commands for synthetic verification (installed scikit-learn 1.7.2 environment):

    python -m pytest tests/test_final_evaluation.py tests/test_cli.py tests/test_plotting.py -q -ra
    python -m pytest -q -ra

The following real command is implemented but MUST NOT be run until separately
approved after review of this implementation:

    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:NUMEXPR_NUM_THREADS = '1'
    python -m airfoil_noise final-evaluate --data data/raw/airfoil_self_noise.dat --directory data/splits --selection reports/development/milestone-4-search/selected_configurations.json --recommendation reports/development/milestone-4-benchmarks/recommendation.json --output-dir reports/final/milestone-5-frozen-models

The real loader retains the official default raw SHA256 check; it cannot
silently accept edited raw data. Plotting runs inside this explicit command
after successful saved-record validation and does not make additional fits
or predict calls. No routine verification command triggers this final run.

Thread settings remain process-local OMP_NUM_THREADS=MKL_NUM_THREADS=
OPENBLAS_NUM_THREADS=NUMEXPR_NUM_THREADS=1, with threadpool_limits(1).
Any single fit/predict duration is a diagnostic only, not another benchmark.
Source and provenance collection, input preparation, plotting and I/O are outside
those optional diagnostic timers.

## Predictions, metrics and predefined reports

Root predictions.csv: model, configuration_id, row_id, condition_id, truth_db,
prediction_db; one unique (model,row_id) pair per observation. Mark the partition
as final_test in run metadata; do not invent a CV fold label.

report/summary.csv: five rows in the predefined mean, linear, tree, SVR, MLP order,
including primary indicator, configuration/seed, observation/condition counts,
equal-condition MAE, pooled MAE, pooled RMSE, pooled R2 and undefined reason,
warning-qualified status and fit/predict diagnostics.

report/condition_metrics.csv: all 110 model/condition rows, n_observations, MAE/RMSE.
report/development_vs_final.csv: stage-labelled OOF development versus final results.
For MLP, compare final seed17 with its representative seed17 OOF result; show the
three-seed selection score separately if included, never as a seed17 prediction.

Primary metric for 22 conditions:
E = (1/22) sum_g [(1/n_g) sum_i |truth_gi - prediction_gi|].
Do not average condition RMSEs into pooled RMSE or average fold scores.
Existing R2 handling is unchanged: undefined for n<2 or exactly constant truth,
including perfect constant predictions; NaN internally, JSON null/CSV blank with
an explicit reason. Negative defined R2 remains valid.

Predefined figures:
- Predicted versus measured dB: all309 observations, five model panels,
  identity lines and consistent axes.
- Residual (truth minus prediction) versus frequency: every observation, five
  panels; a logarithmic frequency axis is only a plot choice, not a model transform.
- Measured/predicted frequency sweeps for ALL22 withheld conditions, sorted by
  condition_id and then frequency; paginate if needed, show all five frozen models.
- All-condition MAE heatmap/table: five models x22 conditions, fixed ID order.
No favourable examples are selected using errors. No new speed measurements.
Keep M4's measured cost evidence labelled separately from final accuracy.
No acceptable engineering dB threshold is invented.

## Diagnostics, failure policy and records

Capture every model's fit/predict stage, warnings/category/message, exceptions,
completed iterations and loss history, fitted scaler statistics, tree size/depth,
support-vector count and MLP parameter count as applicable. Reuse existing
diagnostic helpers; do not refit for diagnostic collection.

A ConvergenceWarning with finite complete predictions and valid required
diagnostics remains eligible and explicitly warning-qualified, matching M4.
It does not establish optimiser convergence. Record other warnings too.
A failed fit, exception, malformed/nonfinite prediction or incomplete alignment
fails that model. Continue other model families once each if possible, preserving
their records. Never silently retry or alter settings. I/O/global errors can stop
the run with actual attempted counts and partial records.

If any model fails or fewer than1,545 valid records are available, retain
diagnostics and partial predictions, exit nonzero, mark the run failed/incomplete,
and DO NOT produce a successful completion summary or final comparison figures.
No replacement runs are authorised by this protocol. Undefined R2 alone is not
failure. Completion is only written after all five fits and all saved-row/metric
checks succeed; warning-qualified completion is clearly labelled.

The protected directory records source SHA256 and raw path, frozen assignment/
metadata hashes, complete selection/recommendation snapshots/hashes, canonical
configuration hashes, full class/parameter pipeline records, original feature
order, seeds, row/condition identifiers, environment/package versions, Python,
hardware/thread details, command arguments, Git HEAD/worktree/source hashes,
warnings/failures/diagnostics, saved predictions, metric tables and figure
source hashes. Preserve earlier records byte-for-byte.

## Synthetic tests before final-target evaluation

- Counters prove one fresh complete-estimator fit and one test predict per method;
  no state sharing, hidden CV, repeated fits or benchmark calls.
- Spies verify training contains only development rows/targets; the inference
  function never receives test truth. Changing test truth cannot change fitted
  settings/state or predictions, only later metrics.
- Extreme holdout inputs leave fitted scaler means/variances and training state
  unchanged; statistics match development exactly. IDs/target are absent from
  the five-column feature matrix.
- Complete frozen configuration JSON round-trips for all five methods, exact
  hashes/seed17/early_stopping=False checked, corrupt/stale records rejected.
  Tiny synthetic MLP settings are labelled verification-only and never replace
  real saved settings.
- Shuffled rows/non-default indexes join by IDs correctly. Duplicate/missing IDs,
  conflicting condition labels, source/hash mismatches and condition overlap
  fail clearly. Every valid observation is predicted once per model.
- Recompute metrics from saved predictions, including hand-calculated cases,
  unequal condition sizes and existing undefined-R2 cases.
- Deliberate fit exceptions, prediction errors/nonfinite values, warnings and
  partial outputs verify status/counters/no-success-summary/no replacement fits.
- Output refusal occurs before fitting; routine audit/check-split/test/default
  commands cannot accidentally invoke final evaluation; plots perform no fits.

## Acceptance criteria and post-evaluation rule

Implementation acceptance: all synthetic/full tests pass, exact approved hashes
and five settings preserved, protected outputs untouched, explicit CLI invocation
and no-overwrite/failure controls verified before real target access.

Successful final-run acceptance: exactly five fits, five holdout predict calls,
1,545 complete unique prediction records across all22 conditions/model, correct
saved-record-derived metrics (110 condition rows), complete diagnostics/provenance,
four predeclared figure types and protected outputs. Test performance has no
predefined pass threshold. Failure is reported without a success summary.

After holdout exposure, no tuning, seed/configuration changes, new transforms,
ensembles or primary-recommendation switching based on test results. Unexpected
accuracy, warning or failure is reported honestly, with development evidence
distinguished from final evidence. The domain remains this dataset and supplied
five measurements, including known displacement thickness.

Implementation and synthetic verification are authorised. The real holdout run,
committing, pushing and GitHub settings changes require separate approval.
No real final-output directory is created during implementation verification.

## Implemented record boundary and regeneration

run_final_evaluation prepares development data and strips holdout truth before
calling fit_predict_frozen. The latter rejects any target column in its holdout
input argument. Only five complete estimators are fitted and each predicts one
batch; truth is attached afterward by row_id. Synthetic mode refuses the
official source checksum; the real CLI always uses final_holdout mode and the
approved original dataset/counts/settings/version.

The new root contains original-byte selection.json/recommendation.json snapshots,
run.json, holdout_inputs.csv, predictions.csv and diagnostics.json. evaluation.json
records exact fit/predict/model-failure counts and every required record checksum.
Only five complete models and validated saved records can produce report/.
completion.json is written last, after successful table/figure generation.
A ConvergenceWarning produces success_warning_qualified, not a convergence claim.
Actual fit exceptions and model failures (which also include prediction/diagnostic
failures) have separate counts. I/O failures propagate and leave no completion
marker; already saved diagnostics remain available where possible.

The four figure types are in report/figures/, with all conditions in ID order,
eleven sweep panels per page (two pages for the real22 conditions). Manifests
bind saved input records, figures and rendering source. No fitted model artifact
or original data file is required for report regeneration.

To regenerate tables and figures into a NEW directory without constructing,
fitting or predicting with estimators:

    python -m airfoil_noise report-final --run-dir reports/final/milestone-5-frozen-models --output-dir reports/final/milestone-5-report-reproduction

Or call regenerate_final_report(run_directory, new_report_directory) or
plot_final(run_directory, new_figure_directory). Saved-record integrity and
coverage checks run first. Existing output directories are refused. Failed or
incomplete evaluation records cannot generate a successful report. All reported
error metrics are calculated from reloaded predictions.csv; development
comparisons use the saved representative-seed OOF score, with configuration-
selection scores clearly labelled separately.

No threshold on final accuracy is a software acceptance criterion. Negative
R2 is retained. Peak memory/FLOPs and new speed measurements are outside scope.


## Execution record: separate approval received, 24 October 2026

The earlier approval-boundary/future-tense sections preserve the pre-run review
procedure. The student subsequently authorized **one** real holdout execution,
conditional on passed checkout/full-suite checks. Both checks were confirmed and
preflight passed. No implementation changes were required.

Completion:2026-10-24T00:40:44.684977+00:00; status:**success_warning_qualified**.
Budget used:**five fits, five batch prediction calls,1,545 records**. Training
used1,194 development observations; evaluation used309 withheld observations
across22 conditions. No failed models, replacement fits or new settings/seeds.

| Model | Equal-condition MAE (dB) | Pooled MAE (dB) | Pooled RMSE (dB) | Pooled R2 |
| --- | ---: | ---: | ---: | ---: |
| Training mean | 5.397600 | 5.426240 | 6.774592 | -0.007731 |
| Linear regression | 3.739213 | 3.743735 | 4.837754 | 0.486115 |
| Decision tree | 2.253768 | 2.281100 | 3.108217 | 0.787871 |
| SVR (preselected primary) | 2.503000 | 2.514759 | 3.957523 | 0.656105 |
| MLP (seed17; warning-qualified) | 3.065901 | 3.031129 | 4.104886 | 0.630018 |

The tree has the lowest observed final error. **SVR remains the preselected
primary recommendation**, unchanged after holdout exposure. MLPseed17 reached
2,000 iterations with the retained ConvergenceWarning; no convergence claim or
retry was made. SVR's worst condition MAE was11.876dB; all conditions are shown.

The protected output is reports/final/milestone-5-frozen-models/. Its three metric
CSVs were recomputed from saved predictions and matched byte-for-byte with all
model calls blocked. Source alignment, development-only scaler statistics,
record/manifests and all five PNG/SVG pairs passed checks. All24 run files and
all516 prior reports retained their hashes.

Exact command used once:

```powershell
C:\Users\lachi\AppData\Local\Temp\mma3001-m4-dependency-check-d0sfzxwi\venv\Scripts\python.exe -B -m airfoil_noise final-evaluate --data data/raw/airfoil_self_noise.dat --directory data/splits --selection reports/development/milestone-4-search/selected_configurations.json --recommendation reports/development/milestone-4-benchmarks/recommendation.json --output-dir reports/final/milestone-5-frozen-models
```

The command ran with all four documented thread variables=1, removed
PYTHONPATH/PYTHONHOME, no sys.path workaround and scikit-learn1.7.2. Preflight
versions matched the installed340-test checkout environment. No further fitting,
tuning, seed/configuration selection or primary switching is permitted based on
these final results. Regenerate reports only from saved records into new paths.
No project commit, push or GitHub settings change is authorized by this run.
