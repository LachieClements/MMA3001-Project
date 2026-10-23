# Milestone 2: fixed baseline development results

This study uses the approved seed-3001 holdout and saved five-fold grouped
development assignments, unchanged from Milestone 1. Each model produces one
out-of-fold prediction for each of 1,194 development observations across 84
conditions. All frequencies of a condition stay together. There are two fixed
configurations and ten fits, with no tuning or seed search. Final-test
predictions have not been calculated.

| Method | Equal-condition MAE (dB), primary | Pooled MAE (dB) | Pooled RMSE (dB) | Pooled R² |
| --- | ---: | ---: | ---: | ---: |
| Training mean | 5.680 | 5.685 | 6.944 | -0.0045 |
| Multiple linear regression | 3.771 | 3.768 | 4.855 | 0.5091 |

Equal-condition MAE first averages absolute errors within each condition, then
averages those 84 values equally. The overall result is calculated from all
out-of-fold predictions, not the unweighted average of five fold scores.
Pooled errors instead give each frequency observation equal weight. Errors
and predictions remain in dB; no target scaling is used.

The training-mean predictor ignores the inputs and predicts the average target
of its training fold. Linear regression uses an intercept and a weighted sum
of the original five measurements, without scaling, log-frequency or additional
features. Its coefficients minimise squared training error, while our declared
comparison metric is equal-condition absolute error.

Linear regression lowers the primary error by 33.6% relative to the mean
benchmark, and has lower per-condition MAE for 76 of 84 conditions. This is a
reduction in a prediction-error measure, not a reduction in physical noise.
The gain is not universal; condition_metrics.csv includes every condition.
Linear fold-level equal-condition MAEs range from 3.489 to 4.289 dB, compared
with 5.391 to 5.848 dB for the mean predictor. Fold variation is descriptive,
not a confidence interval: these fits have overlapping training observations.

The mean predictor's slightly negative pooled R² is permissible. Each fold
uses its own training mean; it need not beat the hypothetical constant mean
of all the pooled validation truths used in R²'s denominator. Undefined R²
(constant truth or fewer than two observations) is explicitly distinguished
from a negative defined value by the metric implementation.

All ten fits completed without warnings or failures. The five linear fits
each report rank 5 for their five-feature training matrix. This checks numerical
rank, not physical independence or causal interpretation of coefficients.
Full settings, fitted means/coefficients/intercepts, singular values and
separate fit/prediction timings are retained in diagnostics.json and run.json.
Timing uses one measurement per fold, without warmup or repeated identical-data
fits. Fold sizes vary and startup/cache effects can matter; these records do
not support a final training-speed or prediction-speed ranking. No model-size,
peak-memory or FLOP estimate is reported in this milestone.

For a presentation: "Holding whole operating conditions together, linear
regression reduced average condition-level absolute error from about 5.68 dB
to 3.77 dB on our development folds. The inputs contain useful predictive
information beyond a constant training mean, but this does not yet demonstrate
an acceptable engineering error or independent final-test performance."

Linear regression is the stronger of these two development baselines. Selection
of the main method remains for the later, approved development comparison.
A single linear relationship may miss curvature and interactions; subsequent
methods can test that possibility using the same saved folds. No settings
were changed in response to these results.

This evidence concerns withheld combinations within the represented experimental
domain. It does not establish performance on other airfoils, measurement setups,
unsupported flow regimes or physical extrapolation. Displacement thickness is
a supplied measurement, not calculated by the model. No acceptable dB-error
threshold has been specified. Grouped CV used during development is not an
independent final estimate; the 22 withheld conditions remain reserved for the
approved final-evaluation procedure.

To inspect or reproduce the evidence, use predictions.csv, summary.csv,
condition_metrics.csv, fold_metrics.csv, diagnostics.json and run.json in this
directory. Source/split metadata and implementation-file hashes are in run.json.
Implementation hashes record the exact working-file bytes used for this run;
Git text-ending conversion can change those hashes without changing Python
semantics. Dataset and assignment checksums retain their protected byte rules.
The verification code, actual commands and executed tests are recorded in
../../../docs/test-report.txt. The README gives a reproduction command writing
to a new directory; existing outputs cannot be replaced by the CLI.
