# FPA Traceability

This document ties the Python replay implementation back to the shipped FPA source fixture.

## Naming Reality

The current shipped sample contains two different version labels:

- testcase / public API: `PFA V3`, for example `aclnnPromptFlashAttentionV3`
- host-side tiling implementation: `PromptFlashAttentionTilingV2`

This project therefore reconstructs the `tiling V2` implementation that backs the shipped `PFA V3` testcase path.

## Source Fixture Files

- Host-side tiling header:
  [fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling.h](../fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling.h)
- Host-side tiling implementation:
  [fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp](../fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp)
- Tiling constants:
  [fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_const.h](../fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_const.h)
- V3 API bridge:
  [fixtures/prompt_flash_attention/op_api/aclnn_prompt_flash_attention_v3.cpp](../fixtures/prompt_flash_attention/op_api/aclnn_prompt_flash_attention_v3.cpp)

## Struct Reuse

`PromptAttentionSeqParams` is defined in the host-side tiling header and then reused at runtime to carry per-core split ranges.

| Struct field | Header comment | Runtime meaning |
| --- | --- | --- |
| `CoreHeadNumTail` | `coreNStart` | `coreNidStart` |
| `actualS1` | `coreNEnd` | `coreNidEnd` |
| `actualCoreNums` | `coreSidStart` | `coreSidStart` |
| `singleCoreHeadNumSize` | `coreSidEnd` | `coreSidEnd` |
| `coreSeqPosStart` | `coreSeqPosStart` | `coreSposStart` |
| `coreSeqPosEnd` | `coreSeqPosEnd` | `coreSposEnd` |

Setter writes are recovered from `prompt_flash_attention_tiling_v2.cpp` and also emitted in [fpa_source_analysis.json](fpa_source_analysis.json).

## Python To C++ Mapping

| Python method | C++ function |
| --- | --- |
| `PromptFlashAttentionV2Replayer._select_split_factors` | `PromptFlashAttentionTilingV2::AdjustCVTilingCVDiff` |
| `PromptFlashAttentionV2Replayer._sparse_tokens` | `PromptFlashAttentionTilingV2::SetSparseModeData` |
| `PromptFlashAttentionV2Replayer._get_pre_next_tokens_left_up` | `PromptFlashAttentionTilingV2::GetPreNextTokensLeftUp` |
| `PromptFlashAttentionV2Replayer._get_actual_inner_block_nums` | `PromptFlashAttentionTilingV2::GetActualInnerBlockNums` |
| `PromptFlashAttentionV2Replayer._get_cut_block_nums` | `PromptFlashAttentionTilingV2::GetCutBlockNums` |
| `PromptFlashAttentionV2Replayer._fix_param_with_row_invalid` | `PromptFlashAttentionTilingV2::FixParamWithRowInvalid` |
| `PromptFlashAttentionV2Replayer._get_calc_block_nums_one_head` | `PromptFlashAttentionTilingV2::GetCalcBlockNumsOneHead` |
| `PromptFlashAttentionV2Replayer._build_units` | `PromptFlashAttentionTilingV2::ComputeSplitNBSeq` |
| `PromptFlashAttentionV2Replayer.replay_case` | `PromptFlashAttentionTilingV2::PromptFlashAttentionSplitNBSeq` |

Exact spans are emitted in [fpa_source_analysis.json](fpa_source_analysis.json).

## Output Semantics

- `logical_core_assignments`: source-level logical split groups
- `core_assignments`: physical cores after lane expansion
- `task_units`: finest replay unit with `(sid, nid, spos)`, token ranges, and block ranges
- `task_segments`: contiguous task compression for human inspection
- `task_summary`: compact one-line summary per physical core

## Visualization Semantics

For every physical core SVG:

- the left bar shows unit-index coverage
- the right panes show `Q x KV block` coverage grouped by `(batch, head)`
- identical `vector / cube` panes are expected under `SPLIT_NBS_CUBE`, because both lanes share the same logical split group
