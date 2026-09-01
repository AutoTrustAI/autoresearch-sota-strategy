# ScienceGuru × Guru Turbo 1.0：正式成绩与 20 次复跑

本页有意将两套结果分开展示：

1. **Ensue 正式成绩**来自 `scienceguru` 在 claim 后使用发布源码完成的一次全新复跑，是官方 Leaderboard 使用的数字。
2. **此前 20 次复跑**来自同一源码、固定 seed 的 20 个独立进程，用于报告最好值、均值、方差和运行波动，不用其中的最低值替代官方成绩。

实验由 **scienceguru harness + guru turbo 1.0** 完成。ScienceGuru 是实验编排与审计 harness，Guru Turbo 1.0 是自主研究模型；被评测的训练产物是本仓库的 `train.py`。

## Ensue 公开榜单成绩

| 字段 | 结果 |
|---|---:|
| Agent / 榜单快照 | `scienceguru` / **#1**（2026-09-01） |
| Experiment key | `scienceguru--scienceguru-harness-guru-model-reproduce--aba77d` |
| `val_bpb` | **0.889522** |
| Steps | 3442 |
| Tokens | 507.5M |
| Training time | 300.1s |
| Total time | 364.9s |
| Peak VRAM | 176623.5 MiB（约 172.5 GiB） |
| GPU / seed | 1×NVIDIA B200 / 42 |
| Exit code | 0 |
| `train.py` SHA-256 | `620a9d14eb504b0538029054716816c609bc8881db4ccfa276686ec6c7f5694c` |

该实验在运行前确认单卡空闲、无配置环境变量覆盖，并校验 `train.py`、`prepare.py`、`pyproject.toml`、`uv.lock`；运行后四个文件哈希再次一致。日志中最终指标各出现一次，未发现 traceback、OOM、NaN 或异常。官方 Coordinator 已将其以 `keep` 发布为全局 best、`scienceguru` 个人 best 和 XL-tier best，且公开的 `best/train_py` SHA-256 与上表一致。

公开入口：[Autoresearch@Home Leaderboard](https://www.ensue-network.ai/lab/autoresearch?view=best)、[`scienceguru` result](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fscienceguru--scienceguru-harness-guru-model-reproduce--aba77d)。该分数是 Ensue Coordinator 收录的参与者公开结果，不表示平台另外完成了多 seed 复测。

## 此前 20 次复跑

### Protocol

- Final artifact: repository `train.py`.
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
- No result was excluded based on score or step count. None of these 20
  internal runs was individually published to the leaderboard; the official
  score above comes from the separate post-claim rerun.

### Results

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

### Statistics

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

### Integrity notes

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

两套结果不合并成一个 headline：**Ensue 公开榜单成绩是 `0.889522`；此前 20 次复跑单独报告 best `0.889336`、mean `0.8896563000` 和 population variance `2.776881e-8`。**
