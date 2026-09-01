# Final winner: 20-run replication

## Protocol

- Final `train.py` artifact:
  `train_candidate_full_trigram2048_2048_l1bigram_compact_fp32scratch.py`
- Train SHA256:
  `620a9d14eb504b0538029054716816c609bc8881db4ccfa276686ec6c7f5694c`
- Fixed `prepare.py` SHA256:
  `4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150`
- Hardware: one NVIDIA B200, fixed default `SEED=42`.
- Command: bare `/root/.local/bin/uv run train.py`.
- Cohort: 20 independent sequential processes, comprising the two existing
  exact-SHA verification runs plus 18 additional runs.
- Validity: training time 299.5--300.5 seconds, total time below 600 seconds,
  one complete final evaluation, exit code zero, 94.4M reported parameters,
  depth 8, and matching train/prepare hashes.
- No result was excluded based on score or step count. No leaderboard
  submission was made.

## Results

| Run | val_bpb | Steps | Tokens (M) | Train (s) | Total (s) | Peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.889596 | 3438 | 507.0 | 300.0 | 414.6 | 176623.5 |
| 02 | 0.889745 | 3439 | 507.1 | 300.0 | 363.2 | 176623.5 |
| 03 | 0.889875 | 3437 | 506.8 | 300.0 | 360.2 | 176623.5 |
| 04 | 0.889818 | 3435 | 506.5 | 300.1 | 363.6 | 176623.5 |
| 05 | 0.889654 | 3434 | 506.4 | 300.1 | 364.9 | 176623.5 |
| 06 | 0.889454 | 3438 | 507.0 | 300.1 | 363.8 | 176623.5 |
| 07 | 0.889529 | 3438 | 507.0 | 300.1 | 365.6 | 176623.5 |
| 08 | 0.889882 | 3434 | 506.4 | 300.1 | 362.9 | 176623.5 |
| 09 | 0.889698 | 3437 | 506.8 | 300.1 | 364.3 | 176623.5 |
| 10 | 0.889626 | 3436 | 506.7 | 300.1 | 361.1 | 176623.5 |
| 11 | 0.889689 | 3438 | 507.0 | 300.0 | 367.6 | 176623.5 |
| 12 | 0.889622 | 3435 | 506.5 | 300.0 | 361.5 | 176623.5 |
| 13 | 0.889624 | 3436 | 506.7 | 300.0 | 363.4 | 176623.5 |
| 14 | 0.889815 | 3436 | 506.7 | 300.0 | 363.8 | 176623.5 |
| 15 | 0.889630 | 3437 | 506.8 | 300.1 | 361.7 | 176623.5 |
| 16 | 0.889454 | 3440 | 507.2 | 300.0 | 366.4 | 176623.5 |
| 17 | 0.889648 | 3439 | 507.1 | 300.0 | 366.6 | 176623.5 |
| **18** | **0.889336** | **3443** | **507.7** | **300.1** | **369.6** | **176623.5** |
| 19 | 0.890015 | 3424 | 504.9 | 300.1 | 366.9 | 176623.5 |
| 20 | 0.889416 | 3440 | 507.2 | 300.0 | 360.4 | 176623.5 |

## Statistics

Statistics below use the six-decimal `val_bpb` values reported by the
benchmark. Variances were computed with a numerically stable Welford update.

| Statistic | Value |
|---|---:|
| Best / minimum | **0.889336 (Run 18)** |
| Mean | 0.8896563000 |
| Median | 0.889639 |
| Maximum | 0.890015 (Run 19) |
| Range | 0.000679 |
| Population variance | **2.776881e-8** |
| Population standard deviation | 0.0001666397612 |
| Sample variance | **2.9230326316e-8** |
| Sample standard deviation | 0.0001709687875 |
| Mean 95% Student-t CI (df=19) | [0.8895762841, 0.8897363159] |
| Runs below 0.890 | 19/20 (95%) |

Step counts had mean 3436.7, median 3437, range 3424--3443,
population variance 13.11, and population standard deviation 3.6207734.
The Pearson correlation between steps and `val_bpb` was -0.75660693. The
descriptive OLS fit was
`val_bpb = 1.0093273844 - 0.00003482151 * steps` (`R^2=0.57245405`).
This is an association within fixed-time runs, not a causal estimate.

## Integrity notes

- All 20 logs passed independent metric and hash checks; all had exit code 0.
- Peak VRAM was 176623.5 MiB in every run. Training time was reported as
  either 300.0 or 300.1 seconds, reflecting one-decimal reporting and the last
  completed step around the fixed budget.
- Run 04 was initially stopped by an overly strict post-run validator that
  required the printed value to equal exactly 300.0. Its complete log reported
  300.1 seconds and was retained after the validator was corrected to the
  preregistered 299.5--300.5-second acceptance interval. The training itself
  was not restarted or altered.
- Run 19 was the only score above 0.890. It was valid and was retained in all
  statistics; it also had the cohort's lowest step count (3424).
- Raw runtime logs are intentionally not included in this source-only
  repository; the table above is the audited extraction from those logs.
