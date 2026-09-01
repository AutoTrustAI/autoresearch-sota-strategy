**English** | [简体中文](README.zh-CN.md)

# ScienceGuru × Guru Turbo 1.0: Autoresearch@Home Strategy and Reproduction Report

> **Official leaderboard snapshot as of 2026-09-01: `scienceguru` ranks #1 with `val_bpb=0.889522`.**

This project was produced by **ScienceGuru harness + Guru Turbo 1.0**. The ScienceGuru harness orchestrates experiments, runs them remotely on a B200, audits integrity, and handles the official claim/publish workflow. Guru Turbo 1.0 is the autonomous research model that proposes, implements, and selects training changes. The `train.py` in this repository is the resulting NanoChat training strategy; it is not Guru Turbo 1.0 itself.

The objective is to minimize validation bits per byte (`val_bpb`; lower is better) under a fixed single-GPU, five-minute training budget. This repository includes the final reproducible `train.py`, the unmodified official evaluator `prepare.py`, an environment fixed by `uv.lock`, statistics from the earlier 20 independent internal reruns, and a separate leaderboard rerun completed and published after the formal claim.

> [!IMPORTANT]
> **Fair-comparison note: this strategy uses the same core evaluation protocol and backbone configuration as the specified `vora/lbx154` B200 reference.** Both use one NVIDIA B200, 300 seconds of timed training, the same data and 8,192-token BPE tokenizer, the unmodified official `prepare.py/evaluate_bpb`, `SEED=42`, batch 72 × sequence length 2,048, a depth-8 × width-768 backbone, and the same `val_bpb` metric. The gain comes from the strategy changes below, not from extending the training budget, replacing the evaluator, or using additional GPUs.
>
> “Same configuration” refers only to the evaluation protocol and backbone settings listed above. It does not mean that dependencies, implementation details, or algorithmic hyperparameters are identical; n-gram capacity, trigram learning rate, late attention-source reuse, memory implementation, and the supporting runtime dependencies are intentional changes in this strategy.

## Ensue Public Leaderboard Result

The formal experiment was first claimed by `scienceguru`, then rerun on one B200 with the exact published source and pre/post-run hash audits, and finally published by the official Coordinator with status `keep`:

| Metric | Formal result |
|---|---:|
| Ensue agent / rank | `scienceguru` / **#1** (2026-09-01 snapshot) |
| `val_bpb` | **0.889522** |
| Steps / tokens | 3,442 / 507.5M |
| Training / total time | 300.1s / 364.9s |
| Peak VRAM | 176,623.5 MiB (about 172.5 GiB) |
| GPU / seed | 1 × NVIDIA B200 / 42 |
| Exit status | 0; each evaluation metric appeared once; no OOM, NaN, or exception |
| `train.py` SHA-256 | `620a9d14eb504b0538029054716816c609bc8881db4ccfa276686ec6c7f5694c` |

This formal result is `0.010363` lower than the previous public best of `0.899885`, an improvement of about 1.1516%. The public result, personal best, global best, XL-tier best, and `best/train_py` were all cross-checked. See the [Autoresearch@Home Leaderboard](https://www.ensue-network.ai/lab/autoresearch?view=best) and the [`scienceguru` result](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fscienceguru--scienceguru-harness-guru-model-reproduce--aba77d). This Ensue number is a public leaderboard entry recorded by the Coordinator; it does not imply that the platform independently performed a multi-seed reevaluation.

## 20-Run Replication Statistics

The same source and fixed `SEED=42` were executed sequentially in 20 independent processes:

| Metric | Result |
|---|---:|
| Best `val_bpb` | **0.889336** |
| 20-run mean | **0.8896563000** |
| Median | 0.889639 |
| Worst result | 0.890015 |
| Population standard deviation | 0.0001666398 |
| `< 0.890` | 19 / 20 |
| Steps / tokens for the best run | 3,443 / 507.7M |
| Step range | 3,424–3,443 |
| Peak VRAM | 176,623.5 MiB (about 172.5 GiB) |
| Training / total-time limits | 300s / less than 600s |

Against the previous `0.899885` reference under the same evaluation conditions, the 20-run best is lower by `0.010549`, and the 20-run mean is lower by `0.0102287`. See [RESULTS.md](RESULTS.md) for all per-run results and statistics. Because every rerun uses the same seed, this cohort mainly measures fixed-budget throughput, GPU atomic-operation effects, and numerical nondeterminism; it does not measure generalization across seeds. None of these 20 internal runs was individually published to the leaderboard.

The two result sets are reported separately and are not combined into a 21-run cohort: the Ensue leaderboard score is `0.889522`, while `0.889336` is only the best result in the prior 20-run internal cohort.

## Public Results Comparison

This is a snapshot of publicly reported results as of 2026-09-01. A single run, a best observation, a mean, and a median are not directly interchangeable, and B200 results should not be mechanically ranked against H100/H200 results. See [COMPARISON.md](COMPARISON.md) for the full hardware details, statistical protocols, and pinned sources.

| Team / system | Reported `val_bpb` | Primary protocol | Primary sources |
|---|---:|---|---|
| **ScienceGuru harness + Guru Turbo 1.0** | **0.889522** | Formal Ensue single run; 1×B200; separate 20-run mean of 0.8896563 with the same `train.py` SHA-256 | [Ensue result](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fscienceguru--scienceguru-harness-guru-model-reproduce--aba77d), [20-run statistics](RESULTS.md) |
| Ensue `vora` | 0.899885 | Formal community single run; 1×B200 | [Ensue result](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fvora--shared-trigram-ve-single-table--4f0fa4ed) |
| Tencent Hunyuan Hyra | 0.901543 | Single run with a complete log; 1×B200 | [Official README](https://github.com/Tencent-Hunyuan/Hyra-results/blob/26ebfbe7d491e6521d8bb5fc21fe88bb31460825/AI4AI/nanochat_autoresearch/README.md), [full log](https://github.com/Tencent-Hunyuan/Hyra-results/blob/26ebfbe7d491e6521d8bb5fc21fe88bb31460825/AI4AI/nanochat_autoresearch/FULL_TRAINING_LOG_0.901543.log) |
| HiLoop | 0.9016 | Median of 25 interleaved confirmation runs; best single run 0.8999; 1×B200 per run | [Official blog](https://hiloop.ai/blog/search-is-enough/), [pinned commit](https://github.com/hiloopai/search-is-enough/blob/28d677941d1bbd7fe263e5e95894e16b5c6a28e0/README.md) |
| Recursive (Yuandong Tian's team) | 0.9108745 | Mean across 10 random seeds; 1×B200 per run | [Official article](https://www.recursive.com/articles/first-steps-toward-automated-ai-research), [pinned README](https://github.com/recursive-org/first-steps-toward-automated-ai-research/blob/a962ec43e2e3d7c018e59a2ece623fe6e232fdfb/nanochat_autoresearch/README.md), [raw CSV](https://github.com/recursive-org/first-steps-toward-automated-ai-research/blob/a962ec43e2e3d7c018e59a2ece623fe6e232fdfb/nanochat_autoresearch/results/val_bpb.csv) |
| Imbue Catalyst | 0.9361 | Single-point evaluation after 340 search experiments; 1×H100 | [Official research blog](https://imbue.com/blog/2026-07-20-imbue-catalyst-nanochat), [pinned Catalyst commit](https://github.com/imbue-ai/catalyst/commit/a53221e82c4209fb3c91f7c5d0d87e4dc2cca4ff) |
| SkyPilot | 0.974 | Best observation from about 700 valid experiments; H100/H200 search pool | [Official blog](https://skypilot.ai/blog/scaling-autoresearch), [reproduction example](https://github.com/skypilot-org/skypilot/tree/8b1c320078f5e5c148bd1826ab4d0d9aa6ed4c25/examples/autoresearch) |

## Strategy

Starting from the depth-8, width-768 backbone of the specified `vora/lbx154` B200 reference, this version focuses on n-gram capacity, information reuse, and fixed-budget throughput without changing the official evaluator or the 300-second training budget:

- At layers 1/3/5/7, use two half-width bigram value-embedding tables per layer (eight tables total), expand them to `512x`, and preserve the original `64x` prefix, initialization, and random-number trajectory through strict-CRN expansion.
- Expand the two trigram tables shared by layers 1, 5, and 7 to `2048x / 2048x`; the value embeddings are shared, while each layer retains an independent gate.
- Set the trigram learning-rate scale to `1.0` to match the collision rate and update density after expansion.
- Reuse the post-layer-4 activation as the Q/K/V and attention-gate source for layers 5/6/7, while residual and MLP streams continue to use each layer's current state. This adds no parameters or random draws.
- Reset n-gram history at BOS document boundaries so bigrams and trigrams never span adjacent documents.
- Use occurrence-compact FP32 DirectScratch with dense `int32` owner maps for the two trigram tables and the layer-1 bigram tables, allocating accumulation space only for rows actually used by the `B×T` batch.
- Use a pinned FA4 revision, native BF16, `torch.compile(max-autotune)`, fused RMSProp/Muon updates, CPU tokenizer/packer prefetch, double-buffered asynchronous H2D transfers, and automatic GPU-NUMA affinity to maximize tokens processed within 300 seconds.

## Fixed Configuration

| Item | Configuration |
|---|---|
| GPU | 1 × NVIDIA B200 (Blackwell / SM100) |
| Training budget | `TIME_BUDGET=300s` |
| Command | `uv run train.py` |
| Seed | `42` |
| Sequence length | `2048` |
| Device batch size | `72` |
| Backbone | depth `8`, width `768`, 6 heads |
| Attention pattern | `TTTL` |
| Precision | native BF16; selected n-gram scratch buffers use FP32 |
| Compile mode | `max-autotune` |
| FA4 revision | `7f952e7e7ec1787ad1f7d209d0bdefdb34747af2` |
| Reported scaling parameters | `94.4M` (excludes bigram/trigram tables and must not be interpreted as the total parameter count) |

## Reproduction

Requirements: Linux, one NVIDIA B200 with approximately 180 GB of VRAM, a working CUDA environment, and sufficient local cache and disk space. The code has no attention fallback for non-Blackwell GPUs.

```bash
git clone https://github.com/aajing/autoresearch-sota-strategy.git
cd autoresearch-sota-strategy
uv sync --frozen
uv run prepare.py
uv run train.py > run.log 2>&1
grep '^val_bpb:\|^training_seconds:\|^total_seconds:\|^peak_vram_mb:\|^total_tokens_M:\|^num_steps:' run.log
```

Run from a clean shell and do not set environment variables that override defaults embedded in `train.py`. The first run may download the pinned FA4 kernel revision from Hugging Face. Data preparation, compilation, and final evaluation are excluded from the 300-second training window, but total runtime must remain below the benchmark's 600-second limit.

## Integrity and Provenance

- `train.py` SHA-256: `620a9d14eb504b0538029054716816c609bc8881db4ccfa276686ec6c7f5694c`
- `prepare.py` SHA-256: `4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150`
- `pyproject.toml` SHA-256: `ccc61ee465aa6c648b0b54bd99ce5303a178ff1e9e310da9b11ee6a8615b7183`
- `uv.lock` SHA-256: `1531c911d5a3b7842f483ba6db66a6410ab5492f9ed75d479dde0de4a95df9db`
- Upstream working-tree baseline: [`lbx154/autoresearch-at-home`](https://github.com/lbx154/autoresearch-at-home), commit `7db3ffd1b6047a7865265e57e6e2e18fb04ec20b`
- Benchmark page: [Autoresearch@Home](https://www.ensue-network.ai/lab/autoresearch?view=best)
- Public-result protocols and sources: [COMPARISON.md](COMPARISON.md)
- Original project: [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)

`prepare.py`, data order, tokenizer, and the final `evaluate_bpb` are unchanged. The original copyright and SPDX notices remain intact in the source; see [ATTRIBUTION.md](ATTRIBUTION.md) and [LICENSE](LICENSE).
