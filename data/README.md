# Dataset and attribution

`raw/airfoil_self_noise.dat` is the **Airfoil Self-Noise** dataset hosted by the
UCI Machine Learning Repository. It is a tab-separated file without a header.
Each row records one frequency within an experimental operating condition.

> Brooks, T., Pope, D., & Marcolini, M. (1989). *Airfoil Self-Noise*
> [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5VW2C

Source: https://archive.ics.uci.edu/dataset/291/airfoil+self+noise  
Download: https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip  
Retrieved: 2026-09-28  
SHA-256 of the extracted `.dat` file:
`74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3`

UCI lists the dataset under **CC BY 4.0**. The software licence in the
repository root applies to project code and documentation; the dataset keeps
its source licence and attribution.

| Column order | Name | Unit | Role |
| --- | --- | --- | --- |
| 1 | Frequency | Hz | Input |
| 2 | Angle of attack | degrees | Input |
| 3 | Chord length | m | Input |
| 4 | Free-stream velocity | m/s | Input |
| 5 | Suction-side displacement thickness | m | Input |
| 6 | Scaled sound pressure level | dB | Target |

The dataset has 1,503 rows and 106 distinct combinations of chord, velocity
and angle. Each combination contains a frequency sweep. No split assignments are committed
in the current starter version. Justify the validation design before generating
and committing a new manifest with `airfoil-noise make-split`.

