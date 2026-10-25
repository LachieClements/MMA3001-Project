# NACA 0012 Aerofoil Self-Noise

MMA3001 Numerical Methods and Machine Learning, Semester 2, 2026  
Lachie Clements, 33134529

A reproducible Python investigation of how accurately five regression methods
predict scaled sound pressure level at NACA 0012 operating conditions withheld
from training. The comparison considers prediction error, training cost,
prediction speed and convergence.

## Contents

- [Quick start and verification](#quick-start-and-verification)
- [Principal results](#principal-results)
- [Data and evaluation design](#data-and-evaluation-design)
- [Workflow and reproducibility](#workflow-and-reproducibility)
- [HTML API documentation](#html-api-documentation)
- [Repository structure](#repository-structure)
- [Licence and assessment](#licence-and-assessment)

## Quick start and verification

Install Git and Python 3.10 or later. The recorded experiments used Python
3.10.5 and scikit-learn 1.7.2. From a terminal, clone the public repository and
create a fresh environment. Windows PowerShell:

~~~powershell
git clone https://github.com/LachieClements/MMA3001-Project.git
cd MMA3001-Project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]" "scikit-learn==1.7.2"
~~~

On macOS/Linux, use the same clone and directory commands, replace environment
creation/activation with the following, then run the installation command above:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

Run the normal verification commands from the repository root:

~~~powershell
python -m pip check
airfoil-noise --help
airfoil-noise audit
airfoil-noise check-split
python -m pytest -q -ra
~~~

These commands check installation, original data, frozen assignments and software
behaviour. Tests use temporary outputs and synthetic model fits; they do not
repeat the real-data search, benchmarks or final holdout evaluation. The module
entry point, python -m airfoil_noise, accepts the same commands as airfoil-noise.

The latest recorded full suite passed **340 test cases**. See the executed
[test report](docs/test-report.txt) and [checkout/install verification](docs/reproducibility-report.txt).
Passing software tests does not establish predictive or physical validity.

## Principal results

The approved final evaluation is complete: five frozen models were fitted once
on 1,194 development observations and evaluated on 309 withheld observations
from 22 complete operating conditions.

| Method | Development selection MAE (dB) | Final equal-condition MAE (dB) |
| --- | ---: | ---: |
| Training mean | 5.680 | 5.398 |
| Multiple linear regression | 3.771 | 3.739 |
| Decision tree | 2.850 | 2.254 |
| Scaled RBF SVR | 2.338 | 2.503 |
| Scaled MLP | 3.364 | 3.066 |

Development scores use the saved grouped folds. MLP selection averages seeds
17, 29 and 43; its final evaluation uses the predefined representative seed 17.

SVR was selected before final testing. The tree achieved the lowest observed
final-test error; the original selection remains recorded. The final MLP reached
2,000 iterations with a ConvergenceWarning and remains explicitly
warning-qualified. SVR's worst final condition MAE was 11.876 dB, so its average
error is not a uniform guarantee.

- [Final summary](reports/final/milestone-5-frozen-models/report/summary.csv),
  [all condition errors](reports/final/milestone-5-frozen-models/report/condition_metrics.csv)
  and [final figures](reports/final/milestone-5-frozen-models/report/figures/).
- [Repeated timing benchmarks](reports/development/milestone-4-benchmarks/summary.csv)
  and [accuracy versus prediction time](reports/development/milestone-4-figures/reviewed/accuracy_vs_prediction_time.png).
  Benchmarks distinguish a 256-row batch from separate one-row calls cycling
  through 32 prepared observations.
- [Results interpretation and limitations](docs/milestone-5-results.md).

## Data and evaluation design

The attributed [UCI Airfoil Self-Noise dataset](data/README.md) contains 1,503
observations across 106 chord-length, velocity and angle-of-attack conditions.
The five inputs are frequency (Hz), angle (degrees), chord (m), velocity (m/s)
and suction-side displacement thickness (m). The target is scaled sound pressure
level (dB). Displacement thickness must be supplied; this package does not
calculate it independently.

[Seed 3001 and the frozen assignments](data/splits/README.md) reserve 84 conditions
for development and 22 for final testing. All frequencies from one condition
remain together. Five saved development folds are shared by every method.
The primary MAE gives each condition equal weight; pooled MAE, RMSE and R-squared
are supplementary.

The original raw bytes and checksum-bound assignments/results are preserved.
Use check-split to validate the existing bundle. No cleaning, imputation,
seed search or silent replacement is part of the workflow. The results concern
this experimental dataset and do not establish performance for other aerofoils
or measurement arrangements, or an application-specific acceptable error.

## Workflow and reproducibility

The [detailed workflow and milestone history](docs/workflow.md) retains the
installation notes, baseline commands, configurable-model API, search settings,
timing procedure and completed final-evaluation procedure.

- [Milestone 4 search and benchmark protocol](docs/milestone-4-protocol.md).
- [Milestone 5 final-evaluation protocol](docs/milestone-5-protocol.md).
- [Recorded environments and exact package inventories](docs/environment.txt).
- [Actual AI assistance, checks, errors and student decisions](AI_USE.md).

For tables and figures, regenerate from saved predictions into a **new**
directory:

~~~powershell
python -m airfoil_noise report-final --run-dir reports/final/milestone-5-frozen-models --output-dir reports/final/milestone-5-report-reproduction
~~~

This validates saved records and makes zero estimator construction, fit or
predict calls. Existing output directories are refused. The completed holdout
evaluation must not be repeated merely to recreate reporting or tune settings.

The declared dependency ranges are in [pyproject.toml](pyproject.toml).
Pinning scikit-learn alone is not a complete environment lock; use the recorded
Python/package versions and thread settings when intentionally reproducing
development experiments. Numerical and timing results may vary with environment
and hardware. Full experiment commands and their approval boundaries are in the
[detailed guide](docs/workflow.md).

## HTML API documentation

[Open the HTML API index](docs/api/index.html). GitHub displays HTML file source;
to read the rendered documentation, open docs/api/index.html from your cloned
folder in a web browser. On Windows PowerShell:

~~~powershell
Start-Process .\docs\api\index.html
~~~

For a local HTTP view, including browser search support, run:

~~~powershell
python -m http.server 8000 --bind 127.0.0.1 --directory docs/api
~~~

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
Press Ctrl+C in the terminal to stop the server.

The API pages document the package's data checks, frozen splits, estimators,
metrics, evaluation, benchmarking and plotting. After changing Python
docstrings, rebuild from the installed checkout with:

~~~powershell
pdoc -o docs/api airfoil_noise airfoil_noise.__main__
~~~

## Repository structure

| Path | Responsibility |
| --- | --- |
| src/airfoil_noise/ | Reusable implementation and command-line entry point |
| tests/ | Unit, integration and synthetic workflow verification |
| configs/ | Explicit approved experiment settings |
| data/raw/ | Original attributed dataset |
| data/splits/ | Frozen assignments and provenance metadata |
| reports/ | Saved audit, coverage, predictions, diagnostics, timings and figures |
| docs/ | Detailed workflow, protocols, environment and verification records |
| docs/api/ | Generated HTML API documentation |
| notebooks/ | Optional exploration using the package's reusable functions |
| AI_USE.md | AI assistance and verification log |

## Licence and assessment

Project code and documentation use the MIT [licence](LICENSE). The dataset
retains its CC BY 4.0 attribution, documented in [data/README.md](data/README.md).

Submission requires an accessible repository link and a zipped clone, alongside
the individual report and presentation specified by the brief. See the
[assessment notes](docs/workflow.md#repository-and-assessment) and unit overview
for the complete requirements and dates.
