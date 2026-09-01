# ScienceGuru × Guru Turbo 1.0：Autoresearch@Home 策略与复现报告

> **2026-09-01 官方榜单快照：`scienceguru` 以 `val_bpb=0.889522` 排名第 1。**

本项目由 **scienceguru harness + guru turbo 1.0** 完成：ScienceGuru harness 负责实验编排、远程 B200 执行、完整性审计以及官方 claim/publish；Guru Turbo 1.0 是提出、实现和筛选训练改动的自主研究模型。仓库中的 `train.py` 是二者产出的 NanoChat 训练策略，并不是 Guru Turbo 1.0 本身。

目标是在固定单 GPU、5 分钟训练预算内最小化验证集 `val_bpb`（越低越好）。仓库保留了可直接复现的最终 `train.py`、未修改的官方评估器 `prepare.py`、由 `uv.lock` 固化的环境、此前 20 次独立内部复跑统计，以及一次在正式 claim 后完成并发布的榜单复跑结果。

> [!IMPORTANT]
> **公平对比说明：当前策略与指定的 `vora/lbx154` B200 reference 使用相同的核心评测协议与骨干配置。** 两者均使用单张 NVIDIA B200、300 秒计时训练、相同数据与 8192 BPE tokenizer、未修改的官方 `prepare.py/evaluate_bpb`、`SEED=42`、batch 72 × sequence length 2048、depth 8 × width 768，以及同一个 `val_bpb` 指标。成绩提升来自下述策略改动，而不是延长训练时间、更换评估器或使用更多 GPU。
>
> 这里的“相同配置”只指上述评测协议与骨干设置，并不表示依赖、实现和算法超参数逐项相同；n-gram 容量、trigram 学习率、late attention-source reuse、显存实现和相应运行时依赖正是本策略有意优化的部分。

## Ensue 公开榜单成绩

正式实验先由 `scienceguru` claim，随后在单张 B200 上以完全相同的发布源码重新运行并通过运行前后哈希审计，最后由官方 Coordinator 以 `keep` 发布：

| 指标 | 正式结果 |
|---|---:|
| Ensue agent / 排名 | `scienceguru` / **#1**（2026-09-01 快照） |
| `val_bpb` | **0.889522** |
| Steps / tokens | 3442 / 507.5M |
| 训练 / 总时间 | 300.1s / 364.9s |
| 峰值显存 | 176623.5 MiB（约 172.5 GiB） |
| GPU / seed | 1 × NVIDIA B200 / 42 |
| 退出状态 | 0；评估指标各出现一次；无 OOM/NaN/异常 |
| `train.py` SHA-256 | `620a9d14eb504b0538029054716816c609bc8881db4ccfa276686ec6c7f5694c` |

该次正式成绩比提交前的公开 best `0.899885` 低 `0.010363`（约 1.1516%）。公开记录、个人 best、全局 best、XL-tier best 与 `best/train_py` 均已核验；可查看 [Autoresearch@Home Leaderboard](https://www.ensue-network.ai/lab/autoresearch?view=best) 和 [`scienceguru` result](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fscienceguru--scienceguru-harness-guru-model-reproduce--aba77d)。这里的 Ensue 数字是 Coordinator 收录的公开榜单成绩，不代表平台另行完成了多 seed 复测。

## 复现实验统计

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

相对同评测环境中的此前参考分 `0.899885`，20 次最好成绩低 `0.010549`，20 次均值低 `0.0102287`。完整逐次结果和统计见 [RESULTS.md](RESULTS.md)。这些复跑均使用相同 seed，主要刻画固定时限下的吞吐、GPU 原子操作及数值非确定性，不代表跨 seed 泛化；内部最好值 `0.889336` 与 Ensue 榜单值 `0.889522` 始终分开展示。

## 公开结果对比

以下仅是截至 2026-09-01 的公开结果快照。单次最好、均值、中位数，以及 B200 与 H100/H200 结果不能机械排名；完整硬件、统计口径和固定来源见 [COMPARISON.md](COMPARISON.md)。

| 团队 / 系统 | 报告的 `val_bpb` | 主要口径 |
|---|---:|---|
| **scienceguru harness + guru turbo 1.0** | **0.889522** | Ensue 正式单次；1×B200；另有同 SHA 20 次均值 0.8896563 |
| Ensue `vora` | 0.899885 | 社区正式单次；1×B200 |
| 腾讯混元 Hyra | 0.901543 | 单次完整日志；1×B200 |
| HiLoop | 0.9016 | 25 次交错确认中位数；最佳单次 0.8999；1×B200 |
| Recursive（田渊栋所在团队） | 0.9108745 | 10 个随机 seed 均值；1×B200 |
| Imbue Catalyst | 0.9361 | 340 次搜索后的单点评估；1×H100 |
| SkyPilot | 0.974 | 约 700 次有效实验中的最好观察值；H100/H200 搜索池 |

## 策略简介

该版本以指定 `vora/lbx154` B200 reference 的 depth-8、width-768 骨干为起点，在不改变官方评估器和 300 秒预算的前提下，集中优化 n-gram 容量、信息复用和固定时限吞吐：

- 在第 1/3/5/7 层使用两张 half-width bigram value-embedding 表（共 8 张），将其扩展到 `512x`，并用 strict-CRN 扩容保留原始 `64x` 前缀、初始化和随机数轨迹。
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
git clone https://github.com/aajing/autoresearch-sota-strategy.git
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
- 公开结果与统计口径：[COMPARISON.md](COMPARISON.md)
- 原始项目：[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)

`prepare.py`、数据顺序、tokenizer 和最终 `evaluate_bpb` 均未修改。源码中的原作者版权与 SPDX 声明已原样保留；详见 [ATTRIBUTION.md](ATTRIBUTION.md) 和 [LICENSE](LICENSE)。
