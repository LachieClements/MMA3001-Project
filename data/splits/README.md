# Frozen condition and development-fold assignments

Milestone 1 adopts holdout seed **3001**, test fraction **0.2**, and five grouped
development folds. The student approved this scope on 2026-10-23 before new
model comparisons. The seed was retained from the starter design; no alternative
holdout seeds were searched for performance.

| Partition | Conditions | Rows |
| --- | ---: | ---: |
| Development | 84 | 1,194 |
| Final test | 22 | 309 |

| Validation fold | Conditions | Rows |
| --- | ---: | ---: |
| 1 | 17 | 241 |
| 2 | 17 | 241 |
| 3 | 17 | 240 |
| 4 | 17 | 240 |
| 5 | 16 | 232 |

## Files and meaning

- conditions.csv contains condition_id, chord_m, velocity_m_s, angle_deg and
  split. Every observed condition appears exactly once.
- development_folds.csv contains condition_id and the one-based validation
  fold. Test conditions are excluded.
- metadata.json binds these assignments to source bytes, row/input identity,
  algorithms, grouping, seed, identifier schemes, counts and generation
  environment. It includes checksums for both CSVs and the metadata fields.

GroupShuffleSplit selects complete conditions, using numerically sorted
chord–velocity–angle group labels and seed 3001. The fraction applies to condition
count, giving 22 test conditions rather than exactly 20% of observations.

GroupKFold uses development inputs only, with no shuffling and no fold seed.
It approximately balances row counts. Each development observation belongs
to exactly one validation fold; all other folds supply training rows.
These saved assignments must be identical for every method/configuration.

Condition IDs encode the three numeric key values. They remain stable on
subsets; unlike recalculated group numbers, they do not depend on which other
conditions are present. Signed zero is normalised. Row IDs are one-based
source record numbers, such as row_000001, attached before filtering.
Their dataset identity comes from the source SHA-256. Neither identifier is a
model feature.

CSV numeric keys use full floating-point precision, so a decimal can appear
longer than in the raw file. This preserves the parsed number; it does not edit
or clean the original data. Readers use round-trip float parsing.

## Validation and use

From the repository root, after setup:

~~~powershell
python -m airfoil_noise check-split
~~~

This reads saved assignments and checks schemas, source/input fingerprints,
checksums, IDs, key correspondence, complete/unique coverage, fold labels and
counts. It does not fit models, regenerate assignments or access test targets
for analysis.

Development callers should use load_development(data, directory), returning
development observations and a matching row_id / condition_id / fold table.
Split checks and coverage functions also work without a target column.
Source-byte integrity includes the original target bytes; in-memory target
values are deliberately absent from the input fingerprint and split decisions.

The low-level apply_manifest function remains available for explicit partition
inspection and compatibility; it returns both partitions. It now preserves
original DataFrame indices as well as explicit IDs.

## Preservation and interpretation

There is no force-overwrite option. make-split refuses to replace any of the
three bundle files; an incomplete bundle also fails validation. Audit and
coverage output commands similarly refuse replacement. Checksums detect
accidental changes; they are not signatures against an actor rewriting every
file and checksum.

[Coverage evidence](../../reports/split-coverage.csv) contains only inputs.
Quartiles are observation-weighted within each split. Marginal ranges overlap,
but this neither proves coverage of all input combinations nor establishes
transfer to new airfoils or experimental arrangements.

Model evaluation and selection remain outstanding. Final-test targets must
stay outside development evaluation. Before final results are opened, the
student must approve frozen configurations, selection rationale and the exact
evaluation procedure.
