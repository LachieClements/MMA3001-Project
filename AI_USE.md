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
