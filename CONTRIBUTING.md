# Contributing

Contributions are welcome when they improve one of these areas:

- host-side source extraction robustness
- kernel-side dispatch traceability
- replay correctness
- testcase coverage
- visualization clarity

## Development Loop

```bash
python -m unittest discover -s tests -v
python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json
python tiling_tool.py replay-cases --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
```

Update docs when output structure, scope, or traceability changes.

## Adding a New Operator

1. Add or extend analyzers under `src/flashattention_analyzers/`.
2. If you ship `fixtures/`, keep a complete operator snapshot instead of a trimmed partial copy.
3. Add testcase CSVs and at least one focused assertion on a golden case.
4. Keep host tiling and kernel dispatch traceability together.
5. Regenerate `docs/` and `examples/`.

## Quality Bar

- Do not hardcode per-case answers.
- Prefer source-driven extraction over manual tables.
- Keep logical core groups and physical core expansion distinct if the source does.
- Do not stop at `op_host` if the question is really about what each physical core executes.
- Do not keep a partial fixture snapshot without documenting why.
