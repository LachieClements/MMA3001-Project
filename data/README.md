# Dataset and attribution

raw/airfoil_self_noise.dat is the **Airfoil Self-Noise** dataset hosted by the
UCI Machine Learning Repository. It is headerless and tab-separated. Each row
records one frequency within an experimental operating condition.

> Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise*
> [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5VW2C

Source: https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
Download: https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip
Recorded retrieval date: 2026-09-28
SHA-256 of the original .dat bytes:
74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3

UCI lists the dataset under **CC BY 4.0**. The repository's MIT licence applies
to project code/documentation; the dataset retains its source licence and
attribution. The raw bytes have been preserved.

| Order | Variable | Units | Role |
| --- | --- | --- | --- |
| 1 | Frequency | Hz | Input |
| 2 | Angle of attack | degrees | Input |
| 3 | Chord length | m | Input |
| 4 | Free-stream velocity | m/s | Input |
| 5 | Suction-side displacement thickness | m | Input |
| 6 | Scaled sound pressure level | dB | Target |

## Audit and identifiers

The [executed audit](../reports/data-audit.json) confirms 1,503 rows, 106 distinct
chord–velocity–angle combinations, no exact duplicate observations and no
repeated frequency within a condition. Sweeps contain 8–19 frequencies.
Displacement thickness is constant within every condition.

The loader checks schema, numeric/finite values and positive frequency, chord,
velocity and thickness. Negative angles are permitted by the general loader.
It does not silently drop blank records, impute, remove outliers or overwrite
the source. Invalid format does not establish a physical input limit.

load_airfoil appends row_id and condition_id to the six measurement columns.
Row IDs identify original source records; condition IDs encode the declared
numeric key. Retain these IDs after subsetting/reordering. feature_matrix
selects only the five inputs, in source order.

The loader can accept an expected checksum; normal CLI commands require the
recorded UCI checksum by default. Deliberate synthetic/custom CLI data require
their own explicit --expected-sha256 value. A valid schema alone is not a source
integrity check.

Thickness is assumed supplied at prediction time. The models do not calculate
it independently. Grouping checks report thickness conflicts and split creation
rejects them rather than redefining conditions.

## Approved evaluation design and limits

[Saved assignments](splits/README.md) reserve 22 complete conditions (309 rows)
for final testing. The remaining 84 conditions (1,194 rows) have frozen
five-fold grouped development assignments.

[Input coverage](../reports/split-coverage.csv) reports ranges, quartiles,
distinct values and holdout values outside development ranges, without target
statistics by partition. Complete sweeps stay together. Split selection uses
inputs and a fixed seed, never prediction errors.

The dataset represents NACA 0012 wind-tunnel measurements with a fixed span and
observer position according to UCI. Withheld combinations assess generalisation
within this experimental dataset. They do not establish performance for other
airfoils, measurement arrangements or unsupported flow regimes.

