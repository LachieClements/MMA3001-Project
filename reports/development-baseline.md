# Initial development baseline

These are **development-only** results from `airfoil-noise baseline-cv` on
2026-09-28. Five-fold group cross-validation keeps all frequencies from one
chord / speed / angle condition within the same fold. The 309 final-test rows
were not used to fit models or choose features.

| Model | Mean MAE (dB) | Mean RMSE (dB) | Mean fit + prediction time per fold (s) |
| --- | ---: | ---: | ---: |
| Training mean | 5.6830 | 6.9415 | 0.0004 |
| Linear regression with log10 frequency | 3.8698 | 4.9564 | 0.0022 |

The linear model improves the average development-fold MAE by approximately
1.81 dB compared with a constant prediction. The per-fold MAEs range from
3.72 to 4.18 dB for the linear model. Timing is only a local measurement on
this small dataset; repeat timing before drawing performance conclusions.

These figures measure interpolation across withheld experimental condition
combinations within the dataset. They do not establish accuracy on different
airfoils, observer positions or flight conditions. Next, examine errors by
frequency and condition, compare a nonlinear method using development folds,
and document how model settings are selected before opening final-test labels.

