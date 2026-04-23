# Fixture Source Note

This directory is intentionally kept as a complete Prompt Flash Attention operator snapshot.

## Origin

- Upstream reference: [gitcode.com/cann/ops-transformer/tree/master/attention/prompt_flash_attention](https://gitcode.com/cann/ops-transformer/tree/master/attention/prompt_flash_attention)

## Rule

- Either do not ship `fixtures/` at all, or ship a complete operator snapshot.
- Do not trim this snapshot down to only `op_host`, because the analyzer now relies on both `op_host` and `op_kernel`.

## Sync Expectation

When this snapshot is refreshed:

1. sync from the workspace-level `prompt_flash_attention/`
2. rerun `analyze-source`
3. rerun `replay-cases`
4. rerun `python -m unittest discover -s tests -v`
