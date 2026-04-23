# Test Report

## Environment

- Repository root: `flashattention_tiling_analyzer`
- Source fixture root: `fixtures/prompt_flash_attention`
- Testcase file: `testcases/fa_testcases.csv`
- Python: `3.14.3`
- Replay profile: `aiv_num=32`, `aic_num=32`, `fa_run_flag=true`

## Commands

```bash
python -m unittest discover -s tests -v
python cli.py --input testcases/fa_testcases.csv --output-dir results/quickstart
python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json
python tiling_tool.py replay-cases --input testcases/fa_testcases.csv --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
```

## Results

- Unit tests: `12 / 12` passed
- CSV replay rows: `23 / 23` passed
- Coverage validation: `23 / 23` with `coverage_ok=True`
- Weight validation: `23 / 23` with `weighted_coverage_ok=True`
- Visualization generation: `23` SVG emitted for `23` replayed cases
- Tiling trace: every replayed case emits host branch decisions and a selected kernel tiling key
- Fixture completeness check: `full_operator_snapshot=True`
- Fixture workspace sync: `content_aligned=True`, `missing=0`, `extra=0`, `mismatch=0`
- Kernel dispatch candidate extraction: verified in automated tests

## Notes

- The shipped testcase path is `PFA V3`, while the replayed host-side tiling implementation is `PromptFlashAttentionTilingV2`.
- All shipped testcase rows currently land on the `SPLIT_NBS_CUBE` main path.
- The current main path replays with `singleProcessSOuterSize=64`, `singleProcessSInnerSize=128`, and `effectiveSplitSOuterSize=128`.
- Replay output now includes workload detail, tiling branch traces, selected tiling keys, and kernel execution context.
- The shipped CSV contains `23` rows but `22` unique `testcase_name` values; one duplicated `PFAV3_case7` row is replayed and visualized twice, with `__2` suffixing used to keep artifact names unique.
