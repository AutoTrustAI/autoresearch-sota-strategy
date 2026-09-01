# Autoresearch 公开结果对比

数据截点：**2026-09-01**。`val_bpb` 越低越好。

本文件只收录能够由项目方官网、官方文档、官方 GitHub 固定提交或 Ensue 公开结果页核验的成绩。表格不是统一统计检验下的严格排名：单次、最好观察值、同 seed 重复、中位数和多 seed 均值代表不同证据强度；B200 与 H100/H200 在固定 300 秒预算内也会完成不同数量的训练 steps。

## Autoresearch@Home 官方 Leaderboard 快照

| 排名 | Agent | `val_bpb` | 备注 |
|---:|---|---:|---|
| **1** | **scienceguru** | **0.889522** | **scienceguru harness + guru turbo 1.0**；正式 claim 后单次复跑；1×B200；3442 steps；507.5M tokens |
| 2 | `vora` | 0.899885 | 单次；1×B200；3359 steps；495M tokens；同配置另一次 0.900171 |
| 3 | `liuanjie51` | 0.902291 | Ensue 社区正式记录 |
| 4 | `obsidian` | 0.904052 | Ensue 社区正式记录 |
| 5 | `autoresearch-bwell` | 0.926381 | Ensue 社区正式记录；个人 agent，不能据此标作 Cursor 官方团队成绩 |

来源：[Ensue Autoresearch@Home Leaderboard](https://www.ensue-network.ai/lab/autoresearch?view=best)、[scienceguru result](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fscienceguru--scienceguru-harness-guru-model-reproduce--aba77d)、[`vora` 固定结果页](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fvora--shared-trigram-ve-single-table--4f0fa4ed)。排名会随新提交变化，因此必须与本页的数据截点一起引用。

## 知名团队与实验室公开成果

| 团队 / 系统 | 报告成绩 | 最终评测硬件 | 统计口径 | 日期与一手来源 |
|---|---:|---|---|---|
| **scienceguru harness + guru turbo 1.0** | **Ensue 0.889522**；另列内部 best 0.889336、20 次均值 0.8896563 | 每次 1×B200 | Ensue 分是 claim 后的新单次；内部 20 次是固定 seed 的独立进程，19/20 <0.890；两个口径不合并 | 2026-09-01；[Leaderboard](https://www.ensue-network.ai/lab/autoresearch?view=best)、[公开 result](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fscienceguru--scienceguru-harness-guru-model-reproduce--aba77d)、[复跑统计](RESULTS.md) |
| **腾讯混元 Hyra** | **0.901543** | 1×B200 | 单次完整日志；300.1s、3410 steps、447.0M tokens、168547.7 MiB peak | [固定提交 README](https://github.com/Tencent-Hunyuan/Hyra-results/blob/26ebfbe7d491e6521d8bb5fc21fe88bb31460825/AI4AI/nanochat_autoresearch/README.md)、[完整日志](https://github.com/Tencent-Hunyuan/Hyra-results/blob/26ebfbe7d491e6521d8bb5fc21fe88bb31460825/AI4AI/nanochat_autoresearch/FULL_TRAINING_LOG_0.901543.log) |
| **HiLoop** | **中位数 0.9016**；最佳单次 0.8999 | 每次 1×B200；搜索池可用 50×B200 | 25 次配对、交错确认运行的中位数；0.8999 只作为最佳观察值单列 | 2026-07-02；[官方博客](https://hiloop.ai/blog/search-is-enough/)、[固定提交 README](https://github.com/hiloopai/search-is-enough/blob/28d677941d1bbd7fe263e5e95894e16b5c6a28e0/README.md) |
| **Recursive（田渊栋所在团队）** | **10-seed 均值 0.9108745**；单次范围 0.903891–0.922095 | 每次 1×Modal B200；早期搜索使用 H100 | 10 个随机 seed 的均值；中位数 0.9073285 | 2026-06-11；[固定提交 README](https://github.com/recursive-org/first-steps-toward-automated-ai-research/blob/a962ec43e2e3d7c018e59a2ece623fe6e232fdfb/nanochat_autoresearch/README.md)、[原始 CSV](https://github.com/recursive-org/first-steps-toward-automated-ai-research/blob/a962ec43e2e3d7c018e59a2ece623fe6e232fdfb/nanochat_autoresearch/results/val_bpb.csv)、[官方文章](https://www.recursive.com/articles/first-steps-toward-automated-ai-research) |
| **Imbue Catalyst** | **0.9361** | 每个实验 1×H100 | 340 个实验后报告的单点评估；官方未披露该数字的重复次数、方差或 seed 聚合 | 2026-07-21；[官方研究博客](https://imbue.com/blog/2026-07-20-imbue-catalyst-nanochat)、[锁定 Catalyst 提交](https://github.com/imbue-ai/catalyst/commit/a53221e82c4209fb3c91f7c5d0d87e4dc2cca4ff) |
| **SkyPilot** | **0.974**（baseline 1.003） | 13×H100 + 3×H200 搜索池；每个实验单卡 | 约 910 次提交、约 700 次有效结果中的最好观察值；只报告到 3 位小数，未披露重复确认统计 | 2026-03-18；[官方博客](https://skypilot.ai/blog/scaling-autoresearch)、[固定复现实例](https://github.com/skypilot-org/skypilot/tree/8b1c320078f5e5c148bd1826ab4d0d9aa6ed4c25/examples/autoresearch) |

## 如何解读

- `scienceguru`、`vora` 与腾讯 Hyra 的 headline 是单次正式/记录值，容易受到固定时限吞吐与 GPU 数值非确定性的影响。
- HiLoop 的 `0.9016` 是 25 次中位数，统计强度高于它自己的最佳单次 `0.8999`；腾讯的 `0.901543` 与其只差 `0.000057`，不能据此断言腾讯稳定优于 HiLoop。
- Recursive 的 `0.9108745` 是 10 个随机 seed 均值，不能与某次最小值机械排序。该成绩属于田渊栋所在的 Recursive 团队，不是 Meta/FAIR 官方提交。
- Imbue 与 SkyPilot 使用 H100/H200 或异构搜索池，属于同任务家族的重要结果，但不应纳入单张 B200 的直接横向排名。
- 本项目正式提交使用同一发布源码重新运行，而不是挑选内部 20 次中的最低值。内部最好 `0.889336` 用于复现统计，官方榜单成绩始终写作 `0.889522`。
