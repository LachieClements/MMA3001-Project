# Airfoil self-noise prediction — MMA3001

Individual project by Lachie Clements, Monash University.

## Engineering question
How accurately can regression predict NACA 0012 airfoil self-noise across
frequency at operating conditions withheld from model development?

Airfoil noise matters to engineering design and acoustic assessment. This
project compares computational methods using existing wind-tunnel measurements.
The target is scaled sound pressure level (dB). Inputs are frequency (Hz),
angle of attack (degrees), chord (m), flow speed (m/s), and suction-side
displacement thickness (m). Thickness is a measured input, not a prediction.
See [data definitions and attribution](data/README.md).

## Current status
The repository has been reset to a development starting point. It retains
the loader, audit, optional grouped-split utilities and introductory baseline
code, with tests and HTML source documentation. No split assignment or model
results are committed in the current version. Earlier work remains in Git
history. No trained model or final predictive performance is provided.

## Setup
Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux alternative:
# source .venv/bin/activate
python -m pip install -e ".[dev]"
airfoil-noise audit
python -m pytest -q
pdoc -o docs/api airfoil_noise airfoil_noise.__main__
```

Dependencies are declared in pyproject.toml: NumPy, pandas and scikit-learn;
development tools are pytest and pdoc. A validated environment snapshot is
in [docs/environment.txt](docs/environment.txt). Version ranges in
pyproject.toml are not a complete dependency lock.

The audit reports row count, duplicates, operating-condition counts and ranges.
It rejects malformed, missing, nonnumeric, nonfinite or nonpositive basic
physical inputs. Valid format does not establish physical accuracy or
suitability for extrapolation.

## Decide the evaluation design before modelling
Rows at different frequencies can share the same chord/speed/angle condition.
Consider keeping these complete sweeps together to avoid condition leakage.
Document the choice before creating a new split. The optional utilities have
defaults of a 20% condition holdout and seed 3001; these are starter settings,
not a newly approved or committed experimental design.

Once the design is justified, create and commit the manifest:
```bash
airfoil-noise make-split
airfoil-noise check-split
airfoil-noise baseline-cv
```

The final command compares a training-mean predictor with linear regression
using log10 frequency on development-only grouped folds. It requires the
manifest. Choose at least one method taught in MMA3001, compare credible
alternatives, and select settings using development data only. Reserve final
test data for the final evaluation. Report accuracy, runtime, relevant
sensitivity/robustness and engineering limitations.

## Repository map
| Path | Purpose |
| --- | --- |
| data/raw/ | Original attributed dataset |
| data/splits/ | Future documented condition assignments |
| data/processed/ | Future derived data |
| src/airfoil_noise/ | Loader, audit and optional split/baseline utilities |
| notebooks/ | Exploratory work |
| tests/ | Data checks and tests of split/baseline utilities |
| docs/api/ | Generated HTML source documentation |
| docs/test-report.txt | Latest actual test run |
| reports/report-outline.md | Assessment-aligned report scaffold |
| AI_USE.md | AI assistance and critical reflection record |

## Assessment deliverables
The supplied 2026 brief requires a functional main branch, README, source
comments/docstrings, HTML documentation, licensing, test functions and a test
report. Maintain meaningful development commits and keep documentation current.

Submit a 5–10 page individual report via Moodle (references and appendices
excluded from the 10-page limit), an accessible final repository link and a
zipped clone. Prepare a 5-minute demonstration/presentation and 3 minutes of
questions in Week 14. Obtain dates from the unit overview and appointment
schedule; this repository does not invent deadlines.

## Limits and ownership
This single experimental dataset cannot establish performance for other
airfoils, observer positions or flow regimes. Document input domain,
measurement uncertainty, failure modes and unsupported uses. The report
must distinguish software verification from predictive validation.

Review and understand all AI-assisted material; record actual checks and
decisions in [AI_USE.md](AI_USE.md). Code/documentation use the MIT
[licence](LICENSE); source data retains CC BY 4.0 attribution.
