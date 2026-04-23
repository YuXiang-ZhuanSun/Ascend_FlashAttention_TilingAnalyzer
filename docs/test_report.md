# Test Report

## Environment

- Workspace root: `source_driven_tiling_tool`
- Source fixture root: `fixtures/prompt_flash_attention`
- Testcase file: `testcases/fa_testcases.csv`
- Python: `3.14.3`
- Replay profile: `aiv_num=32`, `aic_num=32`, `fa_run_flag=true`

## Commands

```bash
python -m unittest discover -s tests -v
python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json
python tiling_tool.py replay-cases --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
```

## Results

- Unit tests: 6 / 6 passed
- CSV replay cases: 23 / 23 passed
- Coverage validation: 23 / 23 `coverage_ok=True`
- Weight validation: 23 / 23 `weighted_coverage_ok=True`
- Visualization generation: 23 SVG emitted for 23 replayed cases
- Duplicate `case_id` handling verified: `PFAV3_case7.svg` and `PFAV3_case7__2.svg` both retained

## Notes

- The shipped testcase path is `PFA V3`, while the replayed host-side tiling implementation is `PromptFlashAttentionTilingV2`.
- All shipped testcase rows currently land on the `SPLIT_NBS_CUBE` main path.
- All shipped testcase rows currently replay with `singleProcessSOuterSize=64`, `singleProcessSInnerSize=128`, and `effectiveSplitSOuterSize=128`.
- The SVG output shows both unit-index coverage and per-core `Q x KV block` coverage.
