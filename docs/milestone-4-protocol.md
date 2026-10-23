# Approved Milestone 4 development protocol

Use only the original five inputs and frozen development rows/folds (84
conditions, 1,194 rows, five folds). Final-test targets and predictions are
excluded. No target scaling, new features or split regeneration is allowed.
The machine-readable grid and **every fixed constructor setting** are in
[configs/milestone4.json](../configs/milestone4.json). These settings are checked
against complete pipeline records before fitting. MLP max_iter is **2000**;
synthetic test settings never replace this full-study setting.

| Family | Cartesian search | Seeds | CV fits |
| --- | --- | --- | --- |
| Training mean | strategy=mean | none | 5 |
| Linear | original-five-input OLS | none | 5 |
| Tree | depth 3, 6, None; minimum leaf 1, 5, 10 | 17 | 45 |
| RBF SVR | C 1, 10, 100; epsilon 0.1, 1.0 dB; gamma 0.02, 0.2, 2.0 | none | 90 |
| MLP | layers (10,), (20,), (20,10); alpha 0.0001, 0.01 | 17, 29, 43 | 90 |

There are 35 hyperparameter configurations, 47 configuration/seed variants
and 235 CV fits. Each variant is evaluated and saved separately through the
existing evaluator. Each fit uses a fresh estimator; pipeline scalers learn
only training inputs. Training predictions are separately labelled fit_fold
and used only to describe complexity. Search timing includes separate fit,
validation prediction, training prediction and evaluator/writing wall time.
These single measurements are diagnostic search costs, not final benchmarks.

Fixed settings (including inactive constructor defaults, for sklearn 1.7.2):

- Mean: strategy="mean", constant=None, quantile=None.
- Linear: fit_intercept=True, copy_X=True, n_jobs=1, positive=False,
  tol=1e-6 (inactive for this dense OLS solver).
- Tree: criterion="squared_error", splitter="best", min_samples_split=2,
  min_weight_fraction_leaf=0, max_features=None, max_leaf_nodes=None,
  min_impurity_decrease=0, ccp_alpha=0, monotonic_cst=None, random_state=17.
- SVR: kernel="rbf", degree=3, coef0=0, tol=0.001, shrinking=True,
  cache_size=200 MB, max_iter=-1, verbose=False. Numeric gamma is applied to
  scaled inputs; epsilon remains in dB.
- MLP: loss="squared_error", activation="relu", solver="adam",
  batch_size="auto", learning_rate="constant", learning_rate_init=0.001,
  power_t=0.5, max_iter=2000, shuffle=True, tol=0.0001,
  n_iter_no_change=20, verbose=False, warm_start=False, early_stopping=False,
  validation_fraction=0.1 (inactive), momentum=0.9,
  nesterovs_momentum=True (inactive for Adam), beta_1=0.9, beta_2=0.999,
  epsilon=1e-8, max_fun=15000 (inactive for Adam). The three recorded seeds
  supply random_state.
- StandardScaler (SVR/MLP only): copy=True, with_mean=True, with_std=True.
  Pipeline: memory=None, transform_input=None, verbose=False; steps named
  scaler and regressor. Other families use no preprocessing. No target scaling.

## Selection and failures

For each variant calculate equal-condition MAE over **all** out-of-fold rows:

E = (1/G) sum_g [(1/n_g) sum_i |y_gi - prediction_gi|].

Never average folds equally. For MLP, average the three seeds' E scores to
select hyperparameters; report sample standard deviation (ddof=1) and range.
This is neither an ensemble nor selection of the best seed. Seed 17 is the
predefined representative for benchmarks and any later final evaluation.
Pooled MAE/RMSE/R2 and per-condition errors are supplementary. R2 undefined
for constant truth or fewer than two observations does not invalidate a run.

A variant needs all five successful fits, exactly one finite prediction per
development row, complete fold coverage and finite MLP loss diagnostics.
An MLP configuration needs all three complete seeds. Record every warning,
failure and incomplete run; continue other configurations. There are no retry
fits, substitute seeds or post-result search expansions. Global data/fold
preflight failures stop before fits. A local failure never deletes success
records. ConvergenceWarnings with finite complete predictions remain eligible,
labelled warning-qualified. The converged-only MLP sensitivity requires **all
15 fits** for a hyperparameter configuration to complete without a
ConvergenceWarning. If none qualifies, report that explicitly. Absence of a
warning is an optimiser stopping diagnostic, not proof of a global optimum.

Within 1e-10 dB of the minimum, use the lower fraction of convergence-warned
fits, then within a family lower mean structural size (tree nodes, SVR support
vectors, MLP parameters), then declared grid order. Across families use warning
fraction then mean, linear, tree, SVR, MLP order. Structural counts from different
families are never compared as equivalent cost units. This numerical tie rule
does not define acceptable engineering error.

Freeze one selected configuration per eligible family and the accuracy leader
**before benchmarking**. Timing never changes those choices or the search.
Make and freeze the engineering recommendation **after** considering measured
cost, accuracy, warnings and complexity. Final evaluation requires a separate
student approval of frozen configurations and the exact final procedure.

## Separate repeated benchmarks

Use the same machine/environment and single-thread numerical libraries for
search and benchmarks: OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS,
NUMEXPR_NUM_THREADS=1 before Python imports, plus threadpool_limits(1). Record
loaded pools and hardware. No global Git or machine settings change.

For each selected family fit all development observations once for warmup,
then five fresh measured fits with the identical representative seed. Five
measured rounds cyclically rotate family order. At most 30 benchmark fits,
**265 total real-data fits** including CV. Keep the last measured successful
estimator for prediction/size diagnostics; no extra refit. Failures are recorded
without replacements, and incomplete benchmarks are explicitly labelled.
These repeated fits measure timing variation, not stochastic seed sensitivity.

Prepare inputs outside timers. Use 256 evenly spaced rows in sorted row_id
order for each batch call. Separately prepare the same 32 one-row DataFrames
for every family. Single-observation timing calls predict separately for each
row, cycling through the same order in every round: no 32-row batch and no
division of batch time by row count. Record row IDs, loop structure and actual
call counts, including warmup and calibration. Warm up for 20 rounds, calibrate
rounds by doubling until a block lasts at least 0.1 s (cap 1,048,576 rounds),
then measure 20 blocks per method, rotating method order. Report median and
IQR of block mean seconds per call, plus raw blocks. These are individual-call
latency estimates averaged within blocks, not an individual-call p95.

Fit includes preprocessing; end-to-end prediction includes pipeline scaling.
Exclude loading, preparation, plotting, serialization and output writing from
isolated model timers. Report search wall cost separately from selected-model
fit time, fixed-batch time and single-call latency. Pickle protocol-5 byte length
describes serialized fitted model size, not peak RAM. No FLOP or peak-memory
claim is made; profiling is outside this study's fit budget.

## Records, plots and interpretation

Use new protected search, benchmark and figure directories; refuse overwrites.
Record protocol/checksums, full class/parameter configuration, seeds, source,
environment, row/condition/fold IDs, truth/predictions, errors and warnings.
Save detailed predictions separately from summaries. Produce figures from
saved records: accuracy versus training and prediction time; tree training/OOF
error and complexity; SVR setting/support sensitivity; MLP seed variation,
iterations/loss histories; and all-condition errors. Accuracy-time MLP points
use seed-17 OOF accuracy to match the timed model; show seed-average selection
and variation separately. No favourable condition examples are selected.

Development CV is used for tuning and is not independent final evidence.
Interpretation remains within this experimental domain, with displacement
thickness supplied as an input. Do not invent an acceptable dB threshold or
claim prediction of other airfoils or unsupported flow regimes.
