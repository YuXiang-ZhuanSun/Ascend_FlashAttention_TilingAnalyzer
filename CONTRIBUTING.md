# Contributing

Contributions are welcome, especially when they improve one of the following:

- source extraction robustness
- operator adapter coverage
- testcase coverage
- replay correctness checks
- visualization clarity

## Development Loop

1. Make the code change.
2. Run:

```bash
python -m unittest discover -s tests -v
python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json
python tiling_tool.py replay-cases --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
```

3. Update docs if behavior, scope, or output structure changed.
4. Keep `examples/` and `docs/` in sync with the current code.

## Adding a New Operator

1. Add a new analyzer under `src/op_tiling_analyzer/analyzers`.
2. Add a local source fixture snapshot.
3. Add testcase CSVs.
4. Add at least one golden-case assertion.
5. Add replay outputs and validation docs.

## Quality Bar

- Do not hardcode per-case answers.
- Prefer source-driven extraction over manual tables.
- Keep logical core groups and physical core expansion distinct if the source does.
- Do not remove traceability.
