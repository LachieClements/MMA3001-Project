# AI-use log

The project brief requires a specific, critical reflection. Keep this log as
work proceeds and verify the entries against what actually happened.

| Date | Tool and task | AI contribution | Independent check | Decision or error |
| --- | --- | --- | --- | --- |
| 2026-09-28 | ChatGPT helped draft the repository scaffold, loader, grouped split, baseline and documentation. | Initial code and wording. | Review source, run tests, inspect group counts and explain the evaluation design independently. | The student must decide whether the split and models answer the engineering question. |

Before submission, describe approximate contribution, why AI was used, what
it got wrong or could not establish, and how it changed your understanding.
Do not claim that automated tests establish physical accuracy.


## 2026-10-23: brief-aligned repository reset

ChatGPT reviewed the supplied project brief, revised repository guidance and
report outline, removed old split assignments and result summaries from the
current version, and adapted utility tests to construct temporary splits.
AI supplied the wording and code edits for this reset. Earlier AI-assisted
code and commits remain in history. Automated checks are recorded in
`docs/test-report.txt`; these do not establish predictive or physical validity.
Student review, method selection, independent explanation and critical
reflection remain outstanding. Do not represent planned checks as completed
student verification. Before submission, estimate actual AI contribution and
record errors, reasons for use and impact on learning.

## 2026-10-23: inspection and Milestone 1

Codex inspected the local checkout, read the supplied brief and relevant unit
notebook sections, verified GitHub name/visibility through a read-only connector,
checked the dataset bytes and grouping, and drafted the development plan. The
student approved Milestone 1, its listed files and seed 3001 by replying
"proceed". GitHub settings, commits and pushes were not authorised or performed.

Codex drafted essentially all new implementation, tests and documentation in
this milestone. It added source/row identity, input-only condition/fold creation,
saved provenance checks, development-only loading and protected output commands.
The student supplied the engineering question, methods, constraints and review
requirements. Overall AI contribution to the eventual submission has not yet
been estimated; this entry is not a claim of independent student verification.

Automated checks exercised synthetic invalid inputs, source integrity, stable
IDs, complete/disjoint conditions and folds, CSV round trips, stale/edited
records, held-out target independence and the CLI. The retained starter baseline
test was rerun for software regression only. No new model comparison, tuning or
final-test evaluation was performed. Actual executed results and environment
are recorded in docs/test-report.txt and docs/environment.txt.

Inspection found that the starter split function lost custom indices and that
its manifest checks could not detect changed frequency inputs. The starter
baseline also applied log-frequency and regenerated folds. The index and split
integrity issues were addressed; the legacy baseline CLI is paused pending the
approved shared evaluator in Milestone 2, and its Python source is retained.

One initial target-independence test failed because injecting missing values
converted synthetic integer targets to floating point. The fixture was changed
to floating-point targets, matching the real target type; equality assertions
were retained. Signed-zero condition keys and nullable missing inputs were also
checked during review. These are software checks, not evidence of physical
validity or acceptable prediction error.

Student review remains outstanding: explain the identifier/checksum distinction,
why whole conditions are withheld, why CV is development evidence, how the fixed
seed was chosen, and why thickness must be supplied. Record actual student
checks, disagreements, revisions and effects on learning when they occur.

## 2026-10-23: Git checkout and installation reproducibility follow-up

The student reported that an external reviewer independently ran all 125 tests
and validated the original frozen bundle. This is attributed to the student's
report; it does not establish independent student understanding or physical
validity. The student then requested and approved a seven-file follow-up scope.

Codex drafted the Git attributes, CLI subprocess test correction, README changes
and verification records. Essentially all new text/code in this follow-up was
AI drafted; the student identified the review concerns and approved the policy.
There had been no .gitattributes: the original data uses CRLF, while the two
checksum-bound assignment CSVs use LF. The new rules preserve this distinction.
The previous CLI subprocess test also injected PYTHONPATH, so it was changed
to remove that variable and test the installed package outside the checkout.

Verification used a disposable fixture commit and a separate clone, with
clone-local core.autocrlf=true set before checkout. The original raw checksum
and both assignment checksums survived commit/checkout; assignment CSVs were LF.
A fresh virtual environment then installed using the README command, without
dependency bypasses or source-path overrides. Installed console/module entry
points, audit and check-split passed; two focused integrity tests and all 125
tests passed, with no skips, expected failures or deselections. Actual commands,
environment, package versions and outputs are in docs/reproducibility-report.txt.

The first installation attempt was blocked by sandbox network permissions;
the same command succeeded after download permission was granted. A temporary
logging helper initially failed to print pip's Unicode error symbols to its
cp1252 console; stdout was changed to UTF-8. The complete pip error had already
been retained in the log. Neither issue required a package-code or pyproject
change. These operational failures are retained in the verification record.

The approved dataset, metadata, seed and assignments were not edited. Project
HEAD, index and Git configuration were checked against their starting hashes.
Only disposable repositories received fixture commits/configuration; no project
commit, push or global Git configuration change occurred. The retained baseline
regression test ran within pytest; no model comparison or final evaluation ran.
Student interpretation and presentation preparation remain outstanding.

## 2026-10-23: Milestone 2 shared development baselines

The student requested and approved the file scope for Milestone 2 only: mean
and multiple linear regression, shared metrics/evaluation, CLI restoration,
synthetic verification and saved development results. Codex drafted essentially
all new code, tests, documentation and interpretation for this milestone
(approximately 100% of this milestone's new implementation text). The student
specified the methodological controls and approved the scope. This is not an
estimate of AI's contribution to the eventual whole assignment, or a claim
that the student has independently checked or understood the implementation.

The new estimator builder uses the original five inputs with no transformations.
The legacy log-frequency and regenerated-fold baseline path was removed. The
shared evaluator receives only development observations and their checked
saved row-fold mapping. It creates fresh estimators, retains complete
out-of-fold predictions, calculates equal-condition MAE over all conditions,
and records full settings, diagnostics, warnings/failures and separate
preliminary fit/prediction times. Constant-target and single-row R-squared are
explicitly undefined rather than forced to zero or one.

A fresh temporary virtual environment installed the README's editable dev
requirement. The first install attempt could not access the network inside the
sandbox; retrying with download permission succeeded. There was no dependency
bypass, PYTHONPATH override or sys.path injection. All 77 focused synthetic/CLI
tests passed, followed by all 191 tests in the full suite, with no skipped,
expected-failing or deselected tests. Tests injected failed fits, warnings,
invalid predictions and reused-state traps through temporary test doubles;
those deliberate faults are not active in the implementation. Hand-calculated
metrics, unequal condition/fold sizes, synthetic linear recovery, saved-output
round trips and withheld-target independence were checked. No failed pytest
run occurred during this milestone.

The frozen development study then completed ten fits, with no warnings or
failures, producing 2,388 prediction records. Equal-condition MAE was 5.679567 dB
for the training mean and 3.771056 dB for linear regression. These results were
recomputed from saved predictions; 76 of 84 condition MAEs favoured linear
regression. The interpretation describes CV as development evidence and does
not invent an acceptable engineering error threshold or treat the diagnostic
timings as a benchmark. No method settings were changed after this comparison.

Generated pdoc HTML initially contained trailing whitespace detected by Git's
diff check; generated-file whitespace was normalised. A Windows CIM hardware
query was denied; the CPU name was obtained through a read-only registry query
instead. Neither issue required model, data or split changes. Actual commands,
verification code/outcomes and environment are in docs/test-report.txt and
docs/environment.txt. Original dataset, assignment CSVs, metadata and Git index
hashes remained unchanged. No final-test predictions, additional model families,
project commit, push or repository-setting change occurred.

Student review remains necessary: explain the arithmetic training mean versus
equal-condition scoring, why aggregate CV differs from an unweighted mean of
fold scores, why folds must use fresh models, how target-isolation tests work,
the definition of undefined R-squared, and what the observed error reduction
does and does not show. Review coefficients as associations, not causal effects.
Record the student's actual checks, decisions, disagreements and learning
reflection when they occur; automated verification does not substitute for them.
