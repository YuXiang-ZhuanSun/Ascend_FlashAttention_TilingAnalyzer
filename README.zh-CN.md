# FlashAttention Tiling Analyzer

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-6%2F6%20passed-16A34A)
![Cases](https://img.shields.io/badge/cases-23%2F23%20passed-0F766E)
![Replay](https://img.shields.io/badge/replay-source--aligned-1D4ED8)

[中文 README](README.zh-CN.md) | [English README](README.md) | [姊妹项目：MatMul Tiling Analyzer](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer)

![FlashAttention Tiling Analyzer 横幅](assets/brand/hero-banner.svg)

FlashAttention. Decoded.

像读源码一样读 FlashAttention tiling，像性能工程师一样看分核负载。

FlashAttention 调优里最贵的部分，往往不是算术本身，而是不确定性：这条 testcase 到底命中了哪条 host 侧 tiling 路径，分核是否均衡，每个物理 core 究竟拿到了什么工作？

`FlashAttention Tiling Analyzer` 把这些“猜测”变成“证据”，从仓库内附带的 Prompt Flash Attention host 实现出发，重建 tiling 逻辑并展开成按 core 的工作视图。

## 为什么做这个项目

- 在进入昂贵 profiling 循环之前，先暴露 host 侧 split 选错的问题。
- 判断某个 shape 的逻辑分组和物理 core 展开是否均衡。
- 用具体的按 core 结果解释尾块、覆盖率和利用率风险。
- 生成适合调试、评审和回归跟踪的共享产物。

## 为什么这个结果可信

这个仓库的核心不是手写 tiling 表，而是源码对齐。

- Python replay 直接从仓库附带的 Prompt Flash Attention host fixture 重建而来。
- 当前 public API 与 testcase 路径是 `PFA V3`，但真正被复刻的 host 实现是 `prompt_flash_attention_tiling_v2.cpp`。
- 源码分析、testcase CSV、replay 输出、traceability 文档和 SVG 可视化全部一起交付。
- 当前样板已经验证 `23 / 23` 条 replay 用例和 `6 / 6` 条自动化测试。

## 你能拿到什么

对每条 case：

- 有源码依据的 split 因子和 setter 到运行时字段的映射
- `logical_core_assignments` 与展开后的 `core_assignments`
- `task_units`、`task_segments` 和每个 core 的摘要
- 按 `(batch, head)` 分组的 `Q x KV block` SVG

对批量运行：

- `docs/fpa_source_analysis.json`
- `examples/fa_tiling_output.json`
- `examples/fa_tiling_summary.csv`
- `examples/visualizations/*.svg`

## 快速开始

```bash
python -m pip install -e .
python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json
python tiling_tool.py replay-cases --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
python -m unittest discover -s tests -v
```

如果你想显式指定输入，也可以这样运行：

```bash
python tiling_tool.py replay-cases --source-root fixtures/prompt_flash_attention --cases testcases/fa_testcases.csv --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
```

## 示例产物

![FlashAttention 结果墙](assets/gallery/results-wall.svg)

当前样板会产出：

- 基于 Prompt Flash Attention fixture 的源码分析报告
- 带逻辑 core / 物理 core 细节的 replay JSON
- testcase 汇总 CSV
- 每个物理 core 的 `Q x KV block` SVG 可视化

## 当前范围

- 当前实现重点：`PromptFlashAttentionTilingV2`
- 当前 public API / testcase 路径：`aclnnPromptFlashAttentionV3`
- 当前重点验证 split 路径：`SPLIT_NBS_CUBE`
- 项目定位：分析与诊断工具，不是运行时替代品

## 文档

- [English README](README.md)
- [架构说明](docs/architecture.md)
- [FPA Traceability](docs/fpa_traceability.md)
- [测试报告](docs/test_report.md)
- [源码分析 JSON](docs/fpa_source_analysis.json)
- [示例输出](examples/visualizations)
- [贡献说明](CONTRIBUTING.md)

## 姊妹项目

如果你在找同系列的 MatMul 项目，可以看 [ascend_matmul_tiling_analyzer](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer)。
