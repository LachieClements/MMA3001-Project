# Milestone 5 final holdout results

One separately approved execution completed on 24 October 2026. The review
contains actual final-test evidence, distinct from development tuning evidence.
Status: **success_warning_qualified**. There were five fresh fits on1,194
observations, five batch prediction calls on309 observations/22 complete
conditions, and1,545 unique model/row prediction records. No model failed;
no fit was retried. Settings, seeds, original five inputs and frozen splits
were unchanged. SVR remains the preselected primary engineering recommendation.

| Model | Equal-condition MAE (dB) | Pooled MAE (dB) | Pooled RMSE (dB) | Pooled R2 |
| --- | ---: | ---: | ---: | ---: |
| Training mean | 5.397600 | 5.426240 | 6.774592 | -0.007731 |
| Linear regression | 3.739213 | 3.743735 | 4.837754 | 0.486115 |
| Decision tree | 2.253768 | 2.281100 | 3.108217 | 0.787871 |
| SVR (preselected primary) | 2.503000 | 2.514759 | 3.957523 | 0.656105 |
| MLP (seed17; warning-qualified) | 3.065901 | 3.031129 | 4.104886 | 0.630018 |

The primary metric gives each of the22 operating conditions equal weight.
Pooled metrics give each observation equal weight. Every R2 here is defined;
the mean predictor's negative R2 is retained. All metrics were recalculated
from the saved round-trip prediction CSV, and the three resulting tables
matched the generated report tables byte-for-byte.

The tree had the lowest observed final equal-condition MAE (2.254dB), compared
with2.503dB for SVR. Development OOF evidence had favoured SVR (2.338dB versus
2.850dB for the tree). The changed ranking is reported without changing the
primary recommendation or using the final test for another selection stage.
SVR's condition MAE was lower on11 conditions and the tree's on11; aggregate
errors alone do not show this variation. SVR improved equal-condition MAE by
53.63% over the training-mean reference.

The largest SVR condition MAE was11.876dB at chord0.0254m, velocity71.3m/s,
angle22.2degrees. The tree's MAE there was3.931dB. The fixed-order sweep for
condition3 and the all-condition heatmap expose this case. All22 conditions
and all110 condition-error rows are included; none were removed after seeing
errors. A low average is not a uniform error bound. No physical cause or
acceptable dB threshold is inferred from these measurements alone.

MLP seed17 completed2,000 iterations and emitted one ConvergenceWarning:
"Stochastic Optimizer: Maximum iterations (2000) reached and the optimization
hasn't converged yet." Its predictions/loss diagnostics were finite and complete,
so the existing policy retains it as warning-qualified. No other warnings or
failures were recorded. Optimiser convergence is not claimed, and no extra epochs,
settings or seeds were attempted. The complete2,000-point loss history is saved.

Cost evidence remains the separately measured M4 repeated benchmarks. Their
median development fit times were about0.0041s for the tree,0.1468s for SVR,
and 3.7525 s for MLP. For the same prepared **256-observation batch**, tree/SVR
median prediction times were about 0.689/19.589 ms. The **32 prepared observations**
were the cycling pool for individual calls, with predict invoked separately for
each observation; those individual-call latencies were about 0.648/0.957 ms.
The [saved benchmark protocol](../reports/development/milestone-4-benchmarks/protocol.json)
records batch_rows=256, single_rows=32 and the explicit nested single-call loop.
These are historical benchmark results, not
new timings on the309 final observations. Single fit/predict durations in this
final record are diagnostics only. The fitted tree has2,383 nodes/depth22,
SVR has1,129 support vectors, and MLP has341 parameters; these are distinct
complexity descriptors, not comparable cost units or peak-memory estimates.

The predefined predicted-versus-measured, residual-versus-frequency and
condition-error figures, plus two sweep pages covering all22 conditions, were
derived from saved records. Their manifests/checksums passed, and all five PNG
figures were visually inspected without changing the rendering or predictions.
Read-only verification blocked constructors, fit, predict and evaluation; every
blocked-call counter remained zero. Source truth/row/condition alignment and
development-only scaler statistics were verified. All24 protected final-output
files and all516 prior report files retained their hashes after verification.

The result concerns withheld operating-condition combinations within this
dataset and its measurement arrangement, using all five supplied inputs.
Displacement thickness must already be known. This does not establish prediction
quality for other airfoils, new measurement arrangements or unsupported flow
regimes. Passing software tests and these checks does not prove physical validity.

The reviewed current code and checkout integrity had passed340 tests before
execution. Preflight rechecked565 checkout hashes, all573 current project-file
hashes, frozen configuration hashes/counts/disjointness, one-thread settings
and the installed scikit-learn1.7.2 environment, with zero fit/predict calls.
No implementation or test source was changed for this execution. The full suite
was not rerun after exposing holdout results. The one authorized real command was:

```powershell
C:\Users\lachi\AppData\Local\Temp\mma3001-m4-dependency-check-d0sfzxwi\venv\Scripts\python.exe -B -m airfoil_noise final-evaluate --data data/raw/airfoil_self_noise.dat --directory data/splits --selection reports/development/milestone-4-search/selected_configurations.json --recommendation reports/development/milestone-4-benchmarks/recommendation.json --output-dir reports/final/milestone-5-frozen-models
```

Working directory: C:\Users\lachi\Documents\2026 - University\MMA3001\Project\MMA3001-Project
Process-local environment overrides:
```json
{
  "OMP_NUM_THREADS": "1",
  "MKL_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "NUMEXPR_NUM_THREADS": "1",
  "PYTHONDONTWRITEBYTECODE": "1",
  "MPLCONFIGDIR": "C:\\Users\\lachi\\AppData\\Local\\Temp\\mma3001-final-once-xl76tx0q\\matplotlib",
  "PYTHONIOENCODING": "utf-8"
}
```
PYTHONPATH/PYTHONHOME were removed; no sys.path workaround was used.
Command exit0; total workflow elapsed16.130s, including
provenance, fitting, prediction, record I/O and figure generation. That total is
not a model benchmark. Completion timestamp:2026-10-24T00:40:44.684977+00:00.
Prediction SHA256:c3425298441d0de7a4e7b0ca1a4352e13ca9b108947e9cdaeb610a1bfda06e6e.

Do not repeat the real final-evaluate command, tune settings, change seeds,
introduce an ensemble or switch the primary recommendation after this result.
Regenerate reporting into a NEW directory using only saved records:

```powershell
python -m airfoil_noise report-final --run-dir reports/final/milestone-5-frozen-models --output-dir reports/final/milestone-5-report-reproduction
```

No project commit, push, repository rename or GitHub setting change was made.
These execution/checking/interpretation steps were AI-assisted; they do not claim
that the student has independently checked or understood them.
