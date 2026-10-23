# Milestone 4 development interpretation

The frozen engineering recommendation is the scaled RBF **SVR** with C=100,
epsilon=0.1 dB and gamma=0.2. It was the accuracy leader before timing and remains
the recommended main model after considering the measured costs and diagnostics.
This recommendation is frozen in ../milestone-4-benchmarks/recommendation.json.
The family choices were never changed using timings.

## Comparison

| Model | Selection condition MAE (dB) | Representative condition MAE (dB) | Representative pooled RMSE (dB) | Fit (ms) | 256-row batch (ms) | One-row call (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| training_mean | 5.680 | 5.680 | 6.944 | 0.315 | 0.021 | 0.020 |
| linear | 3.771 | 3.771 | 4.855 | 1.389 | 0.626 | 0.620 |
| tree | 2.850 | 2.850 | 3.930 | 4.087 | 0.689 | 0.648 |
| svr | 2.338 | 2.338 | 3.374 | 146.751 | 19.589 | 0.957 |
| mlp | 3.364 | 3.265 | 4.530 | 3752.502 | 0.916 | 0.869 |

Selection uses all out-of-fold observations and gives each of 84 conditions equal
weight. MLP selection averages seeds 17,29,43; the representative is predefined
seed 17. Its selection mean is 3.364101 dB, sample SD
0.090710 dB, range 3.265332 to
3.443681 dB. It is not an ensemble or a best-seed result.
The accuracy-time plots use representative accuracy to match the timed model.
All five measured fit repeats are fresh estimators with the same seed; they
measure timing variation, rather than seed sensitivity.

The frozen tree is max_depth=None, min_samples_leaf=1, seed17; the frozen MLP is
layers (20,10), alpha=0.01, seed17 for representative benchmarking. All other
settings, including MLP max_iter=2000 and early_stopping=False, are unchanged from
the approved protocol. Mean and linear use their fixed original-five-input
settings. Full pipeline classes/parameters and seeds are saved separately.

SVR reduces condition MAE by 38.0%
relative to linear and 18.0%
relative to the tree on these development predictions. It improves
74/84 condition errors versus linear and 50/84
versus the tree. Average improvement does not cover every condition: the SVR's
worst condition, c=0.0254 m, v=39.6 m/s, a=22.2 degrees, has MAE 10.502 dB.
The tree's worst condition MAE is 7.819 dB and linear's is 7.884 dB.
All condition errors are shown; no favourable sweeps were selected.

## Cost and practical choice

SVR's median fit is 0.147 s versus
4.09 ms for the tree and
3.75 s for MLP. The tree gives a useful
throughput trade-off: its batch prediction is about
28.4
times faster than SVR here, while SVR has lower primary error. Single-observation
latency differs much less; it includes dispatch and pipeline preprocessing and
was measured with separate predict calls, never inferred from batch throughput.
Linear has simple coefficients and low training cost, with higher error.

The development-refit tree has 2,383 nodes, depth22 and 1,192 leaves; the SVR has
1,129 support vectors; MLP has 341 fitted weight/bias parameters. These are
different structural descriptors, not equivalent cost units. Serialized sizes
(protocol5 bytes) are in the benchmark summary and are not peak RAM.
The unpruned tree has zero repeated-fold-training MAE but 2.850 dB OOF condition
MAE: fitting training observations exactly does not establish generalisation.

Search effort is separate from fitting one chosen configuration:

| Family | CV fits | Summed fit seconds | Evaluation plus saving seconds | Convergence-warned fits |
| --- | --- | --- | --- | --- |
| training_mean | 5 | 0.002 | 0.189 | 0 |
| linear | 5 | 0.007 | 0.202 | 0 |
| tree | 45 | 0.114 | 2.103 | 0 |
| svr | 90 | 5.013 | 14.847 | 0 |
| mlp | 90 | 222.243 | 226.778 | 78 |

The sequential search wall time was 244.160 s (246.472 s including CLI startup,
loading and preflight). Search wall time includes metrics, diagnostics and
per-variant saving; isolated fit timers do not. Searches have unequal family
budgets and their selected scores can benefit from that opportunity.
Benchmark wall time is 59.781 s, including warmup/calibration/output overhead;
use isolated medians/IQRs in summary.csv rather than that wall time as model cost.
Raw repeats and exact counters are retained.

## Warnings, sensitivity and verification

All 235 CV fits and 30 benchmark fits completed: **265 total**, no failed fits,
no replacement runs, no search expansion. 78/90 MLP CV fits emitted
ConvergenceWarning; the other families emitted none. The selected MLP has 9/15
warned CV fits and all five seed-17 validation fits warn. All six full-development
MLP benchmark fits (one warmup, five measured) warn. There are no warning-free
MLP configurations under the required all-15-fits rule; converged-only selection
is explicitly null. Finite warning-qualified predictions remain eligible;
they are not labelled fully converged. Loss histories and iterations are saved.
Some other seeds stop using training-loss tolerance; no grouped internal stopping
procedure or row-based validation subset is used.

The prediction study made 394,735 actual calls including warmup, calibration and
measured blocks. Every single-observation round cycles through the same 32
prepared one-row DataFrames; each makes its own predict call. Each batch call
uses the same 256 prepared rows. Twenty measured blocks per model/mode report
median and IQR of block means; no individual-call p95 or fragile speed guarantee
is asserted. Thread variables and loaded numerical pools were one throughout
search/benchmark runs on the same machine/environment.

All 306 full-suite tests passed before real experiments. A later one-test
synthetic figure check passed after a label-only plotting revision. Record checks
independently recalculated all variant metrics, seed means and selected minima;
verified 56,118 OOF rows, 224,472 training rows, exact folds/IDs, 265-fit accounting,
timing counts/statistics and configuration reconstruction without fitting.
The original dataset, frozen bundle and prior reviewed outputs retain their
bytes. No final-test predictions or error metrics were calculated.

Original figure outputs are retained. Label overlap found during visual review
was corrected in plotting.py; reviewed/ contains the regenerated figures with
no repeated real-data fits or timings. Consequently plotting.py alone differs
from the implementation hash in fit records; the fitting/evaluation source
still matches exactly. Figure manifests bind the records and revised plot source.

## Presentation explanation and limits

"I held complete operating conditions together, so neighbouring frequencies
from one condition cannot appear on both sides of a development fold. I compared
the same folds for every model and selected by average error per condition.
SVR gave the best development accuracy. The tree predicts batches much faster,
and linear is easier to interpret. The neural network cost more to train and
often reached its iteration limit. I froze these choices before any final test."

These tuned grouped CV results are development evidence, not an independent
final estimate, physical validation or a claim of acceptable dB error. No final
holdout has been opened for performance evaluation. The model needs all five
measurements, including supplied displacement thickness; it does not calculate
boundary-layer thickness. Claims stay within this experimental dataset domain,
not new airfoils, arrangements or unsupported flow regimes. Peak RAM/profiling,
larger searches, alternative transforms, target scaling and new stopping
procedures were not part of this approved study. Runtime values depend on this
machine and software; package ranges are not a full environment lock.
