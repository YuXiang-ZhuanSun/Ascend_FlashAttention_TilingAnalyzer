# FPA Traceability

This document ties the replay implementation back to both the host-side tiling logic and the kernel-side execution path.

## Naming Reality

- testcase / public API: `PFA V3`, for example `aclnnPromptFlashAttentionV3`
- host-side tiling implementation: `PromptFlashAttentionTilingV2`

So the current tool reconstructs the `tiling V2` implementation that backs the shipped `PFA V3` testcase path.

## Source Files Used

Host side:

- [`fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling.h`](../fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling.h)
- [`fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp`](../fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp)
- [`fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_const.h`](../fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_const.h)
- [`fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_register.cpp`](../fixtures/prompt_flash_attention/op_host/prompt_flash_attention_tiling_register.cpp)

Kernel side:

- [`fixtures/prompt_flash_attention/op_kernel/prompt_flash_attention.cpp`](../fixtures/prompt_flash_attention/op_kernel/prompt_flash_attention.cpp)
- [`fixtures/prompt_flash_attention/op_kernel/prompt_flash_attention_arch32.h`](../fixtures/prompt_flash_attention/op_kernel/prompt_flash_attention_arch32.h)
- [`fixtures/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_template_tiling_key.h`](../fixtures/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_template_tiling_key.h)

## Fixture Sync Evidence

`analyze-source` now emits fixture-level provenance in [`fpa_source_analysis.json`](fpa_source_analysis.json):

- `file_count`: total files shipped in the fixture snapshot
- `manifest_sha256`: aggregate manifest hash for the shipped snapshot
- `workspace_sync`: file-level comparison against the workspace root `prompt_flash_attention/`

In the current workspace run, the shipped fixture matches the workspace source tree exactly after excluding the explanatory `FIXTURE_SOURCE.md` file:

- missing files: `0`
- extra files: `0`
- content mismatches: `0`

## Struct Reuse

`PromptAttentionSeqParams` is defined in the host-side tiling header and reused at runtime to carry per-core split ranges.

| Struct field | Header comment | Runtime meaning |
| --- | --- | --- |
| `CoreHeadNumTail` | `coreNStart` | `coreNidStart` |
| `actualS1` | `coreNEnd` | `coreNidEnd` |
| `actualCoreNums` | `coreSidStart` | `coreSidStart` |
| `singleCoreHeadNumSize` | `coreSidEnd` | `coreSidEnd` |
| `coreSeqPosStart` | `coreSeqPosStart` | `coreSposStart` |
| `coreSeqPosEnd` | `coreSeqPosEnd` | `coreSposEnd` |

Setter writes are recovered directly from `prompt_flash_attention_tiling_v2.cpp`.

## Host To Python Mapping

| Python method | C++ function |
| --- | --- |
| `PromptFlashAttentionV2Replayer._select_split_factors` | `PromptFlashAttentionTilingV2::AdjustCVTilingCVDiff` |
| `PromptFlashAttentionV2Replayer._build_units` | `PromptFlashAttentionTilingV2::ComputeSplitNBSeq` |
| `PromptFlashAttentionV2Replayer.replay_case` | `PromptFlashAttentionTilingV2::PromptFlashAttentionSplitNBSeq` |

Exact spans are emitted in [`fpa_source_analysis.json`](fpa_source_analysis.json).

## Kernel Model

The current shipped path uses:

- entry function: `prompt_flash_attention_FIAS`
- architecture dispatch: `prompt_flash_attention_FIAS_arch32`
- mixed task contract: `KERNEL_TYPE_MIX_AIC_1_2`
- lane model under `SPLIT_NBS_CUBE`: `vector + cube`

That is why replay output now contains:

- case-level `kernel_execution_model`
- case-level `tiling_trace`
- per-core `kernel_execution`

These fields explain how one logical tiling group maps to two physical lanes that share the same host-side work window but participate in different vector/cube roles.

## Output Semantics

- `logical_core_assignments`: host-side logical split groups
- `core_assignments`: physical cores after lane expansion
- `task_units`: finest replay unit with `(sid, nid, spos)` and block coverage
- `task_segments`: contiguous compression for inspection
- `tiling_trace`: host-side split branch, branch condition results, DN override state, selected tiling key, numeric tiling key value, and candidate dispatch branches
- `kernel_execution_model`: kernel-side entry / dispatch / tiling-key context for the case
- `core_assignments[].kernel_execution`: kernel execution context for one physical core
