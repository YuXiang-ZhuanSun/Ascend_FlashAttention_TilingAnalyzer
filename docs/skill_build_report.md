# Skill Build Report

## Goal

Validate that the rewritten Chinese skill can start from source code and testcase inputs, create a new standalone tiling tool directory, and drive it all the way through replay, validation, and delivery.

## Skill Used

The standalone project was built with the Chinese workspace skill `skills/op-tiling-workflow-zh/SKILL.md` during construction. The skill file itself is not shipped inside this repository snapshot.

## What Was Built

The skill was used to construct this standalone project directory:

- [source_driven_tiling_tool](../)

This directory now contains:

- local testcase copy
- local source fixture snapshot
- Python package and CLI
- tests
- docs
- replay outputs

It no longer depends on a pre-existing tool entry in the parent workspace.

## Inputs

- Testcase copy: [testcases/fa_testcases.csv](../testcases/fa_testcases.csv)
- Source fixture root: [fixtures/prompt_flash_attention](../fixtures/prompt_flash_attention)

## Validation Commands

```bash
python -m unittest discover -s tests -v
python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json
python tiling_tool.py replay-cases --output examples/fa_tiling_output.json --summary-csv examples/fa_tiling_summary.csv --visualize-dir examples/visualizations
```

## Validation Results

- 6 / 6 unit tests passed
- 23 / 23 testcase rows replayed successfully
- 23 / 23 testcase rows with `coverage_ok=True`
- 23 / 23 testcase rows with `weighted_coverage_ok=True`
- 23 SVGs generated

## What The Skill Rewrite Fixed

Compared with the earlier skill draft, the rewrite fixed four structural problems:

1. It no longer assumes `tiling_tool.py` already exists.
2. It explicitly requires creating a new standalone tool directory first.
3. It turns “pass every testcase” into a hard delivery gate.
4. It requires project cleanup and a project-level `agent.md` / project map after implementation.

## Additional Improvement In This Iteration

This round went one step further and made the project itself self-contained:

- testcase rows are copied into the project
- a minimal local source snapshot is copied into the project
- the CLI now defaults to those local fixtures

That means the repository can now be run as a standalone sample project, not only as a workspace artifact.

## Residual Reality

- The current productionized adapter is still only `PromptFlashAttentionTilingV2`.
- The testcase/API path is `PFA V3`, while the replayed host-side tiling implementation remains `tiling V2`.
- The framework is reusable, but each new operator still needs a dedicated analyzer.
