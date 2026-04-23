# FlashAttention Tiling Analyzer

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-12%2F12%20passed-16A34A)
![Cases](https://img.shields.io/badge/cases-23%2F23%20passed-0F766E)
![Replay](https://img.shields.io/badge/replay-host%2Bkernel--aligned-1D4ED8)

[中文 README](README.zh-CN.md) | [English README](README.md) | [姊妹项目：MatMul Tiling Analyzer](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer)

![FlashAttention Tiling Analyzer 横幅](assets/brand/banner.png)

FlashAttention. Decoded.

这个仓库从 Prompt Flash Attention 源码树直接重建 tiling 分析，不再只停在 host 侧分核，而是把 kernel 入口、dispatch 分支、tiling key 候选和每个物理 core 的执行语境一起串出来。

## 现在覆盖什么

- `op_host`：结构体、常量、setter 映射、split 逻辑
- `op_kernel`：入口函数、dispatch 分支、tiling key 模板、vector/cube lane 合同
- replay：逻辑 core 分组、物理 core 展开、每核工作摘要
- 可视化：每个物理 core 的 `Q x KV block` SVG

## 为什么这个版本更可信

- replay 仍然来自真实源码，不是手写表。
- `fixtures/prompt_flash_attention` 已经补成完整算子快照，不再是只保留 `op_host` 的残缺副本。
- 当前 fixture 还会输出 `manifest_sha256`，并和工作区源码做文件级同步校验。
- `analyze-source` 现在同时分析 `op_host` 和 `op_kernel`。
- `replay-cases` 除了 workload，还会输出 `kernel_execution_model` 和每个 core 的 `kernel_execution`。
- 当前样板通过 `23 / 23` 条 replay 行验证和 `12 / 12` 条自动化测试。

## 你能拿到什么

源码分析产物：

- `docs/fpa_source_analysis.json`
- fixture 完整性检查与来源信息
- fixture manifest 哈希与工作区同步状态
- host 结构体 / setter / 函数 traceability
- kernel 入口、dispatch 分支、tiling key 模板 traceability

case replay 产物：

- `examples/fa_tiling_output.json`
- `examples/fa_tiling_summary.csv`
- `examples/visualizations/*.svg`
- `kernel_execution_model`
- 每个物理 core 的 `kernel_execution`

## 快速开始

```bash
python -m pip install -e .
python cli.py --input=testcases/fa_testcases.csv --output-dir=results/quickstart
python -m unittest discover -s tests -v
```

这条命令会生成：

- `results/quickstart/replay.json`
- `results/quickstart/summary.csv`
- `results/quickstart/visualizations/*.svg`

如果你想做源码分析，或者想显式控制产物路径，仍然可以用进阶命令：

```bash
python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json
python tiling_tool.py replay-cases --source-root fixtures/prompt_flash_attention --input testcases/fa_testcases.csv --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
```

## 项目结构

- `cli.py`：面向常用 replay 场景的简洁入口，主打 `--input` + `--output-dir`
- `fixtures/prompt_flash_attention/`：可独立交付时使用的完整算子快照
- `testcases/`：仓库内置 testcase 副本
- `tiling_tool.py`：保留的兼容入口，适合进阶子命令
- `src/flashattention_cli.py`：CLI 入口
- `src/flashattention_models.py`：共享数据模型
- `src/flashattention_utils.py`：工具函数
- `src/flashattention_analyzers/`：host / kernel 解析器与 replay 逻辑
- `tests/`：自动化验证

## 当前范围

- 当前适配器：`PromptFlashAttentionTilingV2`
- 当前 testcase / API 路径：`aclnnPromptFlashAttentionV3`
- 当前重点验证 split 路径：`SPLIT_NBS_CUBE`
- 项目定位：分析与诊断工具，不是运行时替代品

## 文档

- [架构说明](docs/architecture.md)
- [FPA Traceability](docs/fpa_traceability.md)
- [测试报告](docs/test_report.md)
- [Skill 构建报告](docs/skill_build_report.md)
- [Fixture 来源说明](fixtures/prompt_flash_attention/FIXTURE_SOURCE.md)
