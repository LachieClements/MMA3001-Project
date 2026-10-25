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

## 2026-10-24: Milestone 3 constructors, configurations and diagnostics

The student reported independent external review of Milestone 2: 191 tests,
all 2,388 prediction records, ten reconstructed fits and the reported summary
comparisons matched. This is attributed to the student's report, not claimed
as the student's own verification or evidence of independent understanding.
The student explicitly authorised the reviewed Milestone 2 commit/push, which
created commit 5069e19. The student then approved the proposed Milestone 3 file
scope before any Milestone 3 edits. That approval did not authorise a new commit,
a tuning study or final evaluation.

Codex drafted essentially all new implementation, tests and documentation in
this milestone (approximately 100% of the newly added text/code, while reusing
the existing package). The student supplied the requirements and approved the
scope. There is no claim that the student has independently read, checked or
understood these changes, or that this fraction describes the eventual whole
assignment. Student decisions about later tuning and method selection remain
outstanding.

The existing builder now supports a raw-input decision tree, StandardScaler
plus RBF SVR, and StandardScaler plus a small Adam MLP. The original five inputs
and unscaled dB target are retained. Tree/MLP default model seed 17 is explicitly
recorded, separate from holdout seed 3001; it was declared before new method
comparisons. Tree depth 5/leaf size 5, SVR C=10/epsilon=0.1/gamma='scale', and
MLP (20,)/max_iter=2000 are starting construction settings, not tuned results.
The small-MLP policy permits one or two layers with at most 64 neurons each.
Automatic early stopping remains disabled; training-loss stopping can still
occur. No row-based internal validation set or target scaling was introduced.

The same evaluator now accepts explicit named configurations. Unfitted
prototypes are checked before any fold is fitted, then sklearn clone creates
fresh nested scalers and regressors for each fold. Recursive JSON records retain
estimator classes, every constructor parameter, named pipeline steps and tuple
types, with reconstruction limited to the seven supported sklearn classes.
Original Milestone 2 baseline configuration records remain readable. Fitted
state is separate: scaler statistics, tree size/depth, SVR support/iteration
counts, and MLP iteration count, parameter count, loss history and warnings.
Unsupported objects are rejected before writing output rather than converted
to opaque strings. New run metadata uses schema version 2 and records actual
preprocessing and per-configuration seeds.

All 121 focused tests passed, followed by all 270 full-suite tests, with zero
skips, expected failures or deselections. No pytest failure occurred during
this milestone. Tests check hand-defined policies, finite prediction shape,
seed repeatability, training-only scaling, fresh nested state, complete saved
OOF records and configuration reconstruction. The small five-method synthetic
workflow reconstructs each saved configuration and reproduces its fold
predictions within numerical tolerances. Changing withheld targets leaves
development results unchanged for all five methods. Existing M2 linear recovery
and metric checks still pass; no model ranking or fragile timing threshold is
asserted.

Temporary test doubles deliberately produce warnings, failed fits and invalid
predictions. A real one-epoch synthetic MLP test also reaches its iteration
limit, confirming that convergence warnings and loss diagnostics are retained
without claiming convergence. These are verification cases, not empirical
evidence that the new methods predict airfoil noise well. There are no active
deliberate faults in the implementation.

Verification reuses the installed editable Milestone 2 virtual environment;
it is not a new installation-reproducibility experiment. No dependencies,
PYTHONPATH overrides or source-path injection were added. README configuration
examples, pip check, installed console/module audit and check-split commands,
generated HTML and protected-file hashes were checked. Actual commands,
outcomes and environment are appended to docs/test-report.txt and
docs/environment.txt. Model tests fit synthetic observations only; real-source
checks audit the data and frozen bundle without evaluating models.

The dataset, frozen assignments, split code, metrics, original baseline wrapper,
CLI and all reviewed baseline outputs remain unchanged. No real-data comparison
of the new methods, tuning study, final-test evaluation or new commit/push ran
in Milestone 3. No performance claim or acceptable engineering error threshold
was inferred from the synthetic tests.

Student review remains necessary: follow a scaler's training-only statistics,
explain fresh cloning versus sharing fitted state, distinguish constructor
JSON from a saved fitted model, reconstruct a pipeline, interpret iteration
limits/loss history, and explain why disabled validation-based early stopping
still permits training-loss stopping. Record actual student checks, disagreements
and reflections when they occur. Planned searches and seed sensitivity must
be agreed before the later real-data development study.

## Milestone 4: approved development search and separate benchmarks

Date: 2026-10-24T01:28:32.391255+11:00
The student supplied/approved grids, seeds and fixed settings, and clarified
pre-benchmark family/accuracy selection versus later engineering recommendation,
all-15-fit converged-only eligibility and the 32 separate-call latency loop.
AI drafted essentially all new M4 source, tests, protocol and interpretation
in this session (approximately 100% of this session's new code/document drafting,
not a measured percentage of the whole assignment). Existing reviewed utilities
and evaluator were reused. Student authorship and engineering decisions remain
the student's responsibility.

AI implemented the bounded runner, optional training records, seed-mean selection,
complete structured pipeline JSON, warning/failure isolation, repeated timings
and record-derived figures. It ran 306 full-suite tests successfully, then the
authorised 235 CV + 30 benchmark fits with no replacements or search expansion.
A later one-test synthetic figure check passed after a plotting-label correction.
Temporary deliberate faults check failure handling/call counts and are restored
by fixtures; there are no active injected faults.

SVR C=100, epsilon=0.1, gamma=0.2 selected at 2.338213 dB equal-condition OOF MAE.
MLP (20,10), alpha=0.01 selected by three-seed mean 3.364101 dB, with predefined
seed17 representing it for benchmarks. 78/90 MLP CV fits and all six MLP benchmark
fits warn; no all-15-fits warning-free MLP configuration qualifies. No real fit
failed. SVR's recommendation was frozen after considering measured cost,
accuracy, warnings and complexity, including slower batch prediction and its
10.502 dB worst-condition error. Future final-test evidence cannot retune choices.

Record checks independently recomputed metrics/minima without refitting, traced
56118 OOF and 224472 training rows, verified configurations/folds, total265 fits,
thread settings, exact394735 prediction calls and timing summaries. Original
data, assignments and prior reviewed results keep their bytes. Matplotlib was
installed into the existing temporary editable environment after a sandbox
network denial was resolved by the approved retry; this is not a fresh-venv
reproduction. Visual review found overlapping labels, fixed only in plotting.py;
new reviewed figures retain original outputs and repeat no real fits/timings.
Fit-record hashes differ only for this documented plotting revision.

A documentation helper had a nested-quote SyntaxError before writing any project
files, then a missing-helper error; it was corrected without changing numerical
outputs. The evidence and failure logs are retained. AI generated documentation
and the review bundle. No M4 commit/push or final-test evaluation occurred.

These are AI-executed checks, not independent student review. Prior external M3
review is student-reported. Nothing here claims student understanding or checking.
Student review should trace a condition through a fold, explain seed means
versus ensembles/best seed, distinguish total search cost from selected-model fit
cost, inspect the 32-row separate-call loop and loss/warning records, and defend
accuracy/throughput/worst-condition trade-offs. Notebook, individual report,
presentation and personal critical reflection remain student work. Record actual
student checks/disagreements when they occur rather than asserting them early.

## Milestone4 cleanup and proposed final-evaluation protocol

Date: 2026-10-24T07:30:33.813048+11:00
The student explicitly authorised a dependency-floor correction and fresh
installation/software verification, while withholding permission to commit,
push, implement Milestone5 or evaluate the holdout. The student reported that
Milestone4's development study had been reviewed. AI observed the clean main
checkout at35a4bb4; connected GitHub read-only metadata confirmed private
visibility and the same main SHA. This records existing state, not an AI commit.

AI made the student-specified >=1.7,<2 correction, clarified the README's
supported range versus exact recorded sklearn1.7.2 environment, and appended
this session's actual records. Approximately100% of the new documentation
drafting/verification scripting in this cleanup was AI-assisted; the dependency
decision and requested scope came from the student. Historical records and all
science outputs remain untouched.

A new isolated venv installed the editable project with sklearn1.7.2. Initial
sandboxed dependency resolution failed before runtime installation; the approved
network-enabled retry succeeded. A patch context mismatch was corrected before
project edits were applied. All306 installed-package tests passed in41.65s,
with no skips/xfails/deselections and no unexpected warnings. pip check, installed
console/module audit and frozen-bundle commands, outside-checkout imports/help,
metadata-floor assertions and five frozen configuration round-trips passed
without PYTHONPATH or sys.path injection. Model fits in tests are synthetic;
no real-data experiment, timing repetition or final-target evaluation ran.

AI drafted a separate approval-only Milestone5 proposal in temporary review
evidence: retain primarySVR, reconstruct the five saved configurations, use
MLPseed17, perform five development fits and1545 holdout prediction records,
and retain warnings/failures without retuning. No final-evaluation source,
tests, CLI or experiment output was implemented or executed. The checkpoint
remains unstaged/uncommitted pending student approval. These are AI-executed
checks, not claims that the student independently checked or understood them.
Student review of the proposed final procedure/diagnostics remains necessary.

## Milestone5 implementation and synthetic verification only

Date: 2026-10-24T08:01:26.466970+11:00
The student approved the bounded final module/CLI/record-derived plots/tests/docs
scope and explicitly withheld permission to evaluate the real holdout, commit,
push or change GitHub settings. AI preserved the existing uncommitted dependency
cleanup and all frozen science outputs. No new model choices or settings were
selected. SVR remains primary, and the future representative MLP stays seed17.

AI drafted essentially all new M5 source, tests, protocol and documentation in
this session (approximately100% of this session's implementation drafting;
not a measured percentage of the entire assignment). It reused the reviewed
model reconstruction, data/split, metric, diagnostic and record-writing helpers.
The new inference boundary rejects holdout truth, uses five original features,
fits fresh complete estimators once each, and attaches truth only after prediction.
Saved-record reporting/plotting constructs no estimators and makes no fit/predict
calls; complete-record hashes/coverage are checked before reporting.

AI executed the focused tests (47,49,then50 passed) and complete suites
(339,then340 passed after a visual-label/pagination addition) in the documented
installed sklearn1.7.2 environment. No pytest failure occurred; skips/xfails/
deselections were zero. Deliberate synthetic fit/predict/I/O failures and MLP
ConvergenceWarnings verify retention, no retries, partial records and no false
completion. Tests verify scaler statistics, held-out truth isolation, row/condition
alignment, five fresh fits/predict calls, unchanged configuration hashes/seed17,
hand calculations and report regeneration with model calls disabled.

Visual inspection found long synthetic IDs crowded sweep titles; AI used numbered
operating-condition titles with units and retained full IDs in a saved key.
The final22-condition synthetic case verifies all110 condition errors and both
sweep pages. API HTML was regenerated and locally linked pages checked. The
542 protected files and historical record byte prefixes were verified unchanged.
No real-data fit/predict, search, benchmark rerun or final-target evaluation ran.
All numerical smoke records are synthetic, including those in the review ZIP.

The real command is implemented but remains unrun. Its future budget is five
fits and1545 predictions; settings/recommendation cannot be changed based on
test performance. Finite convergence-warning runs stay warning-qualified;
failed/incomplete evaluations never receive a successful comparison report or
completion marker. Software tests do not prove physical validity, acceptable
error, final convergence or final accuracy.

These were AI-executed checks, not independent student verification. Student
review should trace a row through inference/truth join and saved scoring, explain
the five-fit budget and version/hash controls, inspect a failed synthetic run and
the no-model-call regeneration path, and verify the exact command before giving
separate real-holdout approval. No claim of student understanding is made.


## Report checkout correction and disposable-clone verification

Date: 2026-10-24T11:25:15.171141+11:00
The student reported independent340-test review success and authorized only the
/reports/** -text rule and appended verification records. AI added that rule,
retaining raw-dataset and LF assignment-CSV rules. The user specified the policy;
AI drafted approximately100% of this session's small edit and appended record
text (not a measured fraction of the project). Earlier implementation and all
scientific outputs/settings/seeds/recommendation were preserved.

AI copied the current tree and two already saved synthetic runs into a temporary
fixture repository, committed only there, cloned without checkout, set clone-local
core.autocrlf=true, then checked out. A control file converted to CRLF; all516
project reports and46 saved synthetic files retained their hashes. Approved
selection/recommendation anchors, raw source and both CSV hashes matched.
A fresh installed sklearn1.7.2 environment imported clone/src from outside the
checkout without PYTHONPATH or sys.path workarounds.

The temporary helper initially overconstrained metadata JSON byte endings, and
the saved-record guard used the wrong run-kind field; both were corrected without
changing source, integrity checks or records. Sandboxed pip build-dependency
resolution failed; a network-enabled retry of the same README command succeeded.
A temporary Unicode display issue and documentation-script quoting error were
corrected; an optional denied process query was not retried. These unsuccessful
attempts are recorded explicitly in the reproducibility appendix.

AI ran installed audit/check-split, two source/frozen tests, all340 tests and
pip check successfully. No skipped/expected failures/deselections or pytest
warnings summary. Saved-record regeneration produced three tables and four/five
figure pairs from the two saved runs with constructors, fit, predict and evaluation
blocked; observed call counts were0. Old records/manifests/figure bytes and
historical documentation prefixes remained unchanged. No experiments were
repeated to obtain these fixtures. Project HEAD/index/configuration and all569
other starting file hashes were verified unchanged.

These are AI-executed follow-up checks, not claims of independent student checking
or understanding. The correction addresses checkout integrity, not predictive
validity. No real-data study, benchmark repetition, real-holdout evaluation,
project commit/push or GitHub setting change occurred. Further such actions
require separate approval.


## Approved single final holdout execution and reporting

Date:2026-10-24T11:45:00.791807+11:00
The student explicitly authorized one execution after the passed checkout/full
test checks. AI inspected the unchanged reviewed worktree and verified frozen
source/split/configuration hashes,1194/309 row counts,84/22 conditions, separation,
feature order, primarySVR, MLPseed17 and sklearn1.7.2/thread settings before fitting.
The environment versions matched the successful340-test clone suite. No preflight
failure occurred. AI ran the reviewed final-evaluate CLI exactly once: five fresh
fits, five309-row batch prediction calls and1545 saved prediction records.

All five models completed. MLPseed17 reached2000 iterations with one preserved
ConvergenceWarning; finite complete results remain warning-qualified under the
existing policy. No model failed, no fit was retried, and no setting/seed or primary
recommendation changed. Tree had the lowest holdout equal-condition MAE (2.254dB)
versus SVR2.503dB. AI reported the reversed development ranking and SVR's worst
condition MAE11.876dB without inventing a physical cause, acceptable threshold or
new selection stage. SVR remains primary as chosen before final evaluation.

AI verified saved-record hashes, complete source/row/condition alignment and
development-only scaler statistics with all model construction/fit/predict and
evaluation functions blocked. All three metric tables recomputed byte-identically;
all24 run files stayed unchanged. AI inspected the five generated PNG figures,
including both sweep pages showing every condition, without changing them. No
new tests or implementation were written; the prior340-test result was rechecked,
not rerun after exposing final results. No additional fits, tuning, CV or benchmarks
followed the approved run.

AI drafted the execution helper scripts, actual command/environment/check logs,
README status/results and review narrative/package (approximately100% of this
session's drafting, not a measured project-wide fraction). These are AI-executed
checks and interpretations, not independent student verification or understanding.
Student review should explain equal-condition versus pooled errors, the held-out
condition failure, preserved MLP warning and why a final result cannot be used to
retune or switch the predeclared primary recommendation. No project commit/push
or GitHub setting change occurred. Prior records/bytes were retained.


## Milestone 5 documentation corrections and approval-only checkpoint

Date: 2026-10-24T12:01:15.583962+11:00
The student identified two documentation issues. AI's earlier review narrative
incorrectly called the saved benchmark a32-observation batch; the immutable
benchmark protocol records batch_rows=256 and single_rows=32. AI corrected the
versioned narrative in docs/milestone-5-results.md and linked it from README: the
256 observations form the batch, while32 prepared observations cycle through
separate individual predict calls. Original benchmark values/records were not
changed. The earlier review ZIP/narrative remain historical copies; the current
version explicitly corrects that interpretation.

AI replaced the stale protocol heading and added a dated approval/completion
note. Its original procedure body remains intact. The documented actual run
completed24 October2026 at00:40:44UTC /11:40:44AEDT, with five fits/five batch
calls/1545 records and the retained MLP warning; no new experiment was needed.

AI prepared an exact file list for a proposed single M5 commit and ordinary
main-to-main push. Read-only GitHub queries confirmed the unchanged repository
name, private visibility, and remote main matching local35a4bb4. No files were
staged and no commit/push/configuration mutation occurred. All540 existing report
files, including the24 final artifacts, retain their bytes. Source/tests/data,
settings/seeds and primarySVR are unchanged. No tests, fits, predictions or
benchmarks were rerun for these text corrections. Latest340-test evidence remains
historical rather than a new execution claim.

The student supplied the correction; AI drafted the revised text/checkpoint and
verification scripts (approximately100% of this session's drafting, not a measured
project contribution). These are AI-executed checks, not a claim of independent
student checking or understanding. Commit and push await explicit approval of
the concrete scope.


## 2026-10-25: README orientation and detailed workflow guide

The student explicitly requested a clone-and-cd quick start, contents list,
direct HTML documentation link and local viewing instructions, prominent results
and routine verification commands, and relocation of detailed milestone material
into docs. Connected GitHub metadata confirmed public visibility before editing;
no visibility change was necessary.

ChatGPT/Codex drafted the new README and guide framing, moved the existing
technical sections into docs/workflow.md, adjusted their relative links and
replaced the obsolete remaining-milestones note with current submission guidance.
Approximately all new wording and verification scripting in this task was
AI-drafted. This is not a measured fraction of the whole assignment or a claim
of independent student verification.

Documentation checks validated 60 local links/heading references and balanced
code fences, confirmed the clone URL/directory and HTML viewing commands, and
verified that all 11 original fenced command blocks were retained unchanged in
the guide. README length decreased from approximately 3,550 to 1,050 words.
The proposed commit is limited to README.md, docs/workflow.md and this appended
log entry. Existing AI-use entries are preserved.

No Python source, tests, dependencies, raw data, assignments, experiment settings,
predictions, timing records or generated API pages are edited. No software tests,
model fits, predictions, development studies, benchmarks or final evaluation
were rerun. The recorded 340-test result remains historical evidence. These
documentation checks were AI-executed; student interpretation and explanation
remain the student's responsibility.
