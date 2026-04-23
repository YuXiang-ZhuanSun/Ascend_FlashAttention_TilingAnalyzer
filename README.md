# FlashAttention Tiling Analyzer

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-6%2F6%20passed-16A34A)
![Cases](https://img.shields.io/badge/cases-23%2F23%20passed-0F766E)
![Replay](https://img.shields.io/badge/replay-source--aligned-1D4ED8)

[中文 README](README.zh-CN.md) | [English README](README.md) | [Sister Project: MatMul Tiling Analyzer](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer)

![FlashAttention Tiling Analyzer Banner](assets/brand/hero-banner.svg)

FlashAttention. Decoded.

像读源码一样读 FlashAttention tiling，像性能工程师一样看分核负载。

Most FlashAttention tuning time is not spent on arithmetic. It is spent on uncertainty: did the testcase hit the expected host-side tiling path, is core splitting balanced, and what exactly did each physical core receive?

`FlashAttention Tiling Analyzer` turns that uncertainty into evidence by replaying the shipped Prompt Flash Attention host implementation and expanding it into per-core workload views.

## Why This Project

- Surface host-side split mistakes before expensive profiling loops.
- Check whether logical-core groups and physical-core expansion are balanced for a shape.
- Explain tail work, coverage, and utilization risk with concrete per-core evidence.
- Produce shareable artifacts for debugging, review, and regression tracking.

## What Makes It Credible

This repository is built on source alignment, not handwritten tiling tables.

- The Python replay is reconstructed from the shipped Prompt Flash Attention host fixture.
- The public API and testcase path are `PFA V3`, while the host implementation actually replayed is `prompt_flash_attention_tiling_v2.cpp`.
- Source analysis, testcase CSVs, replay outputs, traceability docs, and SVG visualizations ship together in one repository.
- The current sample validates `23 / 23` replay cases and `6 / 6` automated tests.

## What You Get

For each case:

- source-backed split factors and setter-to-runtime field mapping
- `logical_core_assignments` and expanded `core_assignments`
- `task_units`, `task_segments`, and per-core summaries
- per-core `Q x KV block` SVGs grouped by `(batch, head)`

For batch runs:

- `docs/fpa_source_analysis.json`
- `examples/fa_tiling_output.json`
- `examples/fa_tiling_summary.csv`
- `examples/visualizations/*.svg`

## Quick Start

```bash
python -m pip install -e .
python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json
python tiling_tool.py replay-cases --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
python -m unittest discover -s tests -v
```

Explicit inputs are also supported:

```bash
python tiling_tool.py replay-cases --source-root fixtures/prompt_flash_attention --cases testcases/fa_testcases.csv --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
```

## Example Outputs

![FlashAttention Result Wall](assets/gallery/results-wall.svg)

The shipped sample produces:

- a source analysis report from the Prompt Flash Attention fixture
- replay JSON with logical-core and physical-core detail
- a testcase summary CSV
- per-core `Q x KV block` SVG visualizations

## Scope

- Current implementation focus: `PromptFlashAttentionTilingV2`
- Current public API / testcase path: `aclnnPromptFlashAttentionV3`
- Current validated split focus: `SPLIT_NBS_CUBE`
- Positioning: analysis and diagnosis tool, not runtime replacement

## Documentation

- [Chinese README](README.zh-CN.md)
- [Architecture](docs/architecture.md)
- [FPA Traceability](docs/fpa_traceability.md)
- [Test Report](docs/test_report.md)
- [Source Analysis JSON](docs/fpa_source_analysis.json)
- [Example Outputs](examples/visualizations)
- [Contributing](CONTRIBUTING.md)

## Sister Project

If you are looking for the MatMul companion in the same family, see [ascend_matmul_tiling_analyzer](https://github.com/YuXiang-ZhuanSun/ascend_matmul_tiling_analyzer).
