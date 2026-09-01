# Attribution and research lineage

## 本项目的角色

- **ScienceGuru harness**：负责候选实验编排、远程单卡 B200 执行、日志与哈希审计，以及 Ensue claim/publish 流程。
- **Guru Turbo 1.0**：提出、实现、筛选并复核训练策略改动的自主研究模型。
- 本仓库的 `train.py`：上述流程产生并接受 Autoresearch@Home 评测的 NanoChat 训练产物；它不是 Guru Turbo 1.0 模型本身。

## 代码基线

`train.py` 和 `prepare.py` 源于 Autoresearch/NanoChat 研究代码。源码保留了原有版权与许可证声明：

- Copyright 2026 Recursive
- Copyright 2025 Andrej Karpathy
- SPDX-License-Identifier: Apache-2.0

本项目直接使用的工作树基线是 [`lbx154/autoresearch-at-home@7db3ffd1`](https://github.com/lbx154/autoresearch-at-home/tree/7db3ffd1b6047a7865265e57e6e2e18fb04ec20b)，其上游研究项目包括 [`mutable-state-inc/autoresearch-at-home`](https://github.com/mutable-state-inc/autoresearch-at-home)、[`recursive-org/first-steps-toward-automated-ai-research`](https://github.com/recursive-org/first-steps-toward-automated-ai-research) 和 [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch)。

共享 trigram value-embedding 路线沿用了基线中标注的 `vora` 公开方案；对应公开结果为 [`vora--shared-trigram-ve-single-table`](https://www.ensue-network.ai/lab/autoresearch?run=results%2Fvora--shared-trigram-ve-single-table--4f0fa4ed)。Late attention-source reuse 的研究脉络参考了 [HiLoop `search-is-enough`](https://github.com/hiloopai/search-is-enough/tree/28d677941d1bbd7fe263e5e95894e16b5c6a28e0) 的公开方向；本仓库的具体实现、组合和消融由 ScienceGuru harness + Guru Turbo 1.0 完成。这里描述的是研究来源与灵感，不主张对上游方案的独占原创。

## 第三方运行时

第三方包与下载 kernel 仍分别受其自身许可证约束。运行时通过 Hugging Face Kernels 获取 [`kernels-community/flash-attn4`](https://huggingface.co/kernels-community/flash-attn4) 的固定 revision `7f952e7e7ec1787ad1f7d209d0bdefdb34747af2`；该 kernel 未 vendored 到本仓库。为运行固定的 FA4/CUTLASS 路径，`pyproject.toml` 和 `uv.lock` 明确包含 `apache-tvm-ffi`、`einops`、`nvidia-cutlass-dsl` 等运行时依赖，因此“与 reference 相同配置”只指评测协议和骨干设置，不表示依赖文件逐字相同。
