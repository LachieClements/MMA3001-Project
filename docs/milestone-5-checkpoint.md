# Milestone 5 checkpoint proposed for approval

Prepared24 October2026. **No staging, commit or push has been performed.**
This is the complete proposed45-file scope relative to current HEAD.

## Commit and destination

Proposed commit message:

    Add verified final holdout evaluation and protected result records

Current branch: `main`.
Current HEAD and live remote main: `35a4bb47ec7f0e44e571d48b403cb0bb6030a7d4`.
Local origin push URL: `https://github.com/LachieClements/MMA3001-Project.git`.
Repository: [LachieClements/MMA3001-Project](https://github.com/LachieClements/MMA3001-Project).
Read-only GitHub metadata confirmed **private** visibility and the unchanged
repository name on24 October2026. The index is empty.

Approval would authorize one commit containing exactly the45 files below and a
normal push of that commit from local `main` to `origin/main` (`main:main`).
Before committing, check the actual staged path list against this list; do not
stage whole directories indiscriminately. Recheck the remote main reference
before pushing; any unexpected remote changes require a new assessment. This
proposal does not authorize a force push, history deletion, rename, visibility
change or any additional experiment. Existing commits/history are preserved.

## Change summary and evidence

The checkpoint includes the reviewed M5 module, explicit CLI, saved-record
figures and synthetic tests; the scikit-learn>=1.7 package guidance with1.7.2
recorded for experiments; report-byte Git attributes; generated HTML; actual
verification/AI-use records; and the single approved final evaluation.

The two latest text corrections distinguish the256-observation benchmark batch
from the32-observation cycling pool and put approval/completion status at the top
of the protocol while retaining its original procedure. The corrected results
narrative is now versioned in docs/milestone-5-results.md.

Completed evidence (not repeated for these documentation edits):
- Installed full suite:340 passed in48.67s, with no skips/xfails/deselections.
- Disposable checkout with clone-local core.autocrlf=true before checkout:
  all565 checked source/split/report/synthetic fixture hashes matched.
- Real final evaluation: five fits on1194 development observations, five batch
  predict calls on309 withheld observations/22 conditions,1545 saved prediction
  rows and110 per-condition metric rows. No failures or retries.
- MLPseed17 reached2000 iterations with a retained ConvergenceWarning; completion
  is success_warning_qualified. Tree's holdout equal-condition MAE is2.254dB,
  SVR's2.503dB. SVR remains the preselected primary recommendation.
- Saved-record metrics, source/row/condition alignment, development-only scaler
  statistics, report manifests and five PNG/SVG figure pairs passed checks with
  model calls blocked. The current documentation correction preserves every
  existing report file:540 in total, including the24 final artifacts.
- git diff --check is part of this documentation checkpoint review. No tests or
  experiments are rerun for these text corrections.

Remaining limitations: MLP is not claimed to have converged; SVR's worst holdout
condition MAE is11.876dB. Results concern this dataset/measurement domain with
known displacement thickness. No acceptable dB threshold or new physical cause
is inferred. The individual report, investigation notebook, presentation and
submission packaging remain separate work.

## Complete proposed file list

### Implementation and package controls (5)

```text
.gitattributes
pyproject.toml
src/airfoil_noise/__main__.py
src/airfoil_noise/final_evaluation.py
src/airfoil_noise/plotting.py
```

### Tests (3)

```text
tests/test_cli.py
tests/test_final_evaluation.py
tests/test_plotting.py
```

### Documentation and executed records (8)

```text
AI_USE.md
README.md
docs/environment.txt
docs/milestone-5-checkpoint.md
docs/milestone-5-protocol.md
docs/milestone-5-results.md
docs/reproducibility-report.txt
docs/test-report.txt
```

### Generated HTML API documentation (5)

```text
docs/api/airfoil_noise.html
docs/api/airfoil_noise/__main__.html
docs/api/airfoil_noise/final_evaluation.html
docs/api/airfoil_noise/plotting.html
docs/api/search.js
```

### Protected final-evaluation output (24)

```text
reports/final/milestone-5-frozen-models/completion.json
reports/final/milestone-5-frozen-models/diagnostics.json
reports/final/milestone-5-frozen-models/evaluation.json
reports/final/milestone-5-frozen-models/holdout_inputs.csv
reports/final/milestone-5-frozen-models/predictions.csv
reports/final/milestone-5-frozen-models/recommendation.json
reports/final/milestone-5-frozen-models/report/condition_metrics.csv
reports/final/milestone-5-frozen-models/report/development_vs_final.csv
reports/final/milestone-5-frozen-models/report/figures/condition_key.csv
reports/final/milestone-5-frozen-models/report/figures/condition_mae.png
reports/final/milestone-5-frozen-models/report/figures/condition_mae.svg
reports/final/milestone-5-frozen-models/report/figures/frequency_sweeps_01.png
reports/final/milestone-5-frozen-models/report/figures/frequency_sweeps_01.svg
reports/final/milestone-5-frozen-models/report/figures/frequency_sweeps_02.png
reports/final/milestone-5-frozen-models/report/figures/frequency_sweeps_02.svg
reports/final/milestone-5-frozen-models/report/figures/manifest.json
reports/final/milestone-5-frozen-models/report/figures/predicted_vs_measured.png
reports/final/milestone-5-frozen-models/report/figures/predicted_vs_measured.svg
reports/final/milestone-5-frozen-models/report/figures/residuals_by_frequency.png
reports/final/milestone-5-frozen-models/report/figures/residuals_by_frequency.svg
reports/final/milestone-5-frozen-models/report/manifest.json
reports/final/milestone-5-frozen-models/report/summary.csv
reports/final/milestone-5-frozen-models/run.json
reports/final/milestone-5-frozen-models/selection.json
```

## Exclusions and unchanged material

No unrelated changed files were found. There are no deletions in the proposed
scope. The original raw dataset, frozen split bundle, configs/milestone4.json,
selected configurations, recommendation and all prior experiment outputs are
unchanged and therefore have no new content in this commit. The24 final outputs
are newly committed records from the approved run, with their original bytes.

Review ZIPs/checksum sidecars and execution/helper temporaries remain outside
the repository. Environments, caches, credentials, .git internals and supplied
teaching/assessment materials are excluded. Generated HTML is deliberately
included here because the assessment requires current API documentation.

Approval is pending for both the commit and push. No additional fitting,
configuration/seed changes or switching the primary recommendation is proposed.
