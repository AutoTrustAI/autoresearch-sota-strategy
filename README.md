# Autoresearch@Home：本地最优策略

这是一个尚未提交排行榜的 **Autoresearch@Home SOTA candidate**。目标是在固定的单 GPU、5 分钟训练预算内，最小化验证集的 `val_bpb`（越低越好）。仓库保留了可直接复现的最终 `train.py`、未修改的官方评估器 `prepare.py`、锁定依赖和 20 次复现实验结果。

> [!IMPORTANT]
> **公平对比说明：当前策略与此前 SOTA 使用相同的评测配置和骨干训练配置。** 两者均使用单张 NVIDIA B200、300 秒计时训练、相同数据与 8192 BPE tokenizer、未修改的官方 `prepare.py/evaluate_bpb`、`SEED=42`、batch 72 × sequence length 2048、depth 8 × width 768，以及同一个 `val_bpb` 指标。成绩提升来自下述策略改动，而不是延长训练时间、更换评估器或使用更多 GPU。
>
> 这里的“相同配置”特指公平比较所需的运行环境、评测协议和骨干设置，并不表示完整算法超参数逐项相同；n-gram 容量、trigram 学习率、late attention-source reuse 和显存实现正是本策略有意优化的部分。

## 成绩

同一份源码、固定 `SEED=42`，以 20 个相互独立的进程顺序运行：

| 指标 | 结果 |
|---|---:|
| 最好 `val_bpb` | **0.889336** |
| 20 次均值 | **0.8896563000** |
| 中位数 | 0.889639 |
| 最差结果 | 0.890015 |
| 总体标准差 | 0.0001666398 |
| `< 0.890` | 19 / 20 |
| 最好一次 steps / tokens | 3443 / 507.7M |
| steps 范围 | 3424–3443 |
| 峰值显存 | 176623.5 MiB（约 172.5 GiB） |
| 训练 / 总时限 | 300 秒 / 小于 600 秒 |

相对同评测环境中的此前参考分 `0.899885`，最好成绩下降 `0.010549`，20 次均值下降 `0.0102287`。完整逐次结果和统计见 [RESULTS.md](RESULTS.md)。20 次均为相同种子，因此主要刻画固定时限下的吞吐、GPU 原子操作及数值非确定性，不代表跨种子泛化；“最好一次”也应与均值同时阅读。

当前结果**尚未提交** Autoresearch@Home leaderboard，不能视为官方榜单名次。

## 策略简介

该版本以此前 SOTA 的 depth-8、width-768 骨干为起点，在不改变官方评估器和 300 秒预算的前提下，集中优化 n-gram 容量、信息复用和固定时限吞吐：

- 将所有八层的 bigram value-embedding 表扩展到 `512x`，并用 strict-CRN 扩容保留原始 `64x` 前缀、初始化和随机数轨迹。
- 将层 1、5、7 共用的两张 trigram 表扩展到 `2048x / 2048x`；共享 value embedding，但各层保留独立 gate。
- 将 trigram 学习率尺度设为 `1.0`，与扩容后的碰撞率和更新密度匹配。
- 层 5/6/7 的 Q/K/V 与 attention gate 复用 post-layer-4 activation；残差和 MLP 流仍使用当前层状态。该改动不增加参数或随机数抽样。
- 在 BOS 文档边界重置 n-gram 历史，避免 bigram/trigram 跨文档串联。
- 为两张 trigram 表及 layer-1 bigram 表使用 occurrence-compact FP32 DirectScratch 与 dense `int32` owner map，只为实际出现的 `B×T` 行分配累加空间。
- 使用固定 revision 的 FA4、原生 BF16、`torch.compile(max-autotune)`、融合 RMSProp/Muon 更新、CPU tokenizer/packer 预取、双缓冲异步 H2D 和自动 GPU-NUMA 亲和性，提高 300 秒内完成的 token 数。

## 固定配置

| 项目 | 配置 |
|---|---|
| GPU | 1 × NVIDIA B200（Blackwell / SM100） |
| 训练预算 | `TIME_BUDGET=300s` |
| 命令 | `uv run train.py` |
| Seed | `42` |
| Sequence length | `2048` |
| Device batch size | `72` |
| Backbone | depth `8`, width `768`, 6 heads |
| Attention pattern | `TTTL` |
| Precision | native BF16；指定 n-gram scratch 为 FP32 |
| Compile | `max-autotune` |
| FA4 revision | `7f952e7e7ec1787ad1f7d209d0bdefdb34747af2` |
| 报告的 scaling params | `94.4M`（不包含 bigram/trigram 表，不应理解为总参数量） |

## 复现

环境要求：Linux、单张约 180 GB 显存的 NVIDIA B200、可用 CUDA，以及足够的本地数据缓存和磁盘空间。代码在非 Blackwell GPU 上没有 attention fallback。

```bash
git clone <THIS_PRIVATE_REPOSITORY_URL>
cd autoresearch-sota-strategy
uv sync --frozen
uv run prepare.py
uv run train.py > run.log 2>&1
grep '^val_bpb:\|^training_seconds:\|^total_seconds:\|^peak_vram_mb:\|^total_tokens_M:\|^num_steps:' run.log
```

请在干净 shell 中运行，不要设置会覆盖 `train.py` 内置默认值的环境变量。首次运行可能会从 Hugging Face 获取已固定 revision 的 FA4 kernel。数据准备、编译和最终评估不计入 300 秒训练窗口，但总运行时间必须低于 benchmark 的 600 秒上限。

## 完整性与来源

- `train.py` SHA-256：`620a9d14eb504b0538029054716816c609bc8881db4ccfa276686ec6c7f5694c`
- `prepare.py` SHA-256：`4f2ba9cbb8ba8c4a3d35be405a913e2f3be3af9aea103ed52ef7b2a662058150`
- `pyproject.toml` SHA-256：`ccc61ee465aa6c648b0b54bd99ce5303a178ff1e9e310da9b11ee6a8615b7183`
- `uv.lock` SHA-256：`1531c911d5a3b7842f483ba6db66a6410ab5492f9ed75d479dde0de4a95df9db`
- 上游工作树基线：[`lbx154/autoresearch-at-home`](https://github.com/lbx154/autoresearch-at-home)，commit `7db3ffd1b6047a7865265e57e6e2e18fb04ec20b`
- Benchmark 页面：[Autoresearch@Home](https://www.ensue-network.ai/lab/autoresearch?view=best)
- 原始项目：[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)

`prepare.py`、数据顺序、tokenizer 和最终 `evaluate_bpb` 均未修改。源码中的原作者版权与 SPDX 声明已原样保留；详见 [ATTRIBUTION.md](ATTRIBUTION.md) 和 [LICENSE](LICENSE)。
