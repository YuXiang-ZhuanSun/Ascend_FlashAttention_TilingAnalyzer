from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SourceSpan:
    path: str
    start_line: int
    end_line: int
    label: str | None = None


@dataclass(slots=True)
class StructField:
    name: str
    cpp_type: str
    array_length: int | None
    comment: str | None
    source: SourceSpan


@dataclass(slots=True)
class StructDefinition:
    name: str
    fields: list[StructField]
    source: SourceSpan


@dataclass(slots=True)
class PlatformProfile:
    aiv_num: int = 32
    aic_num: int = 32
    fa_run_flag: bool = True


@dataclass(slots=True)
class PromptFlashAttentionCase:
    case_id: str
    api_name: str
    layout: str
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None
    input_dtype: str
    key_dtype: str
    value_dtype: str
    output_dtype: str
    attributes: dict[str, Any]
    batch_size: int
    head_num: int
    kv_head_num: int
    seq_q: int
    seq_kv: int
    query_d: int
    value_d: int
    actual_seq_lengths: list[int]
    actual_seq_lengths_kv: list[int]
    pre_tokens: int
    next_tokens: int
    sparse_mode: int
    has_attention_mask: bool
    has_pse_shift: bool


@dataclass(slots=True)
class BlockUnit:
    sid: int
    nid: int
    spos: int
    weight: int
    q_token_start: int
    q_token_end: int
    outer_block_count: int
    inner_block_count: int
    active_inner_block_ranges: tuple[tuple[int, int], ...]
    active_kv_token_ranges: tuple[tuple[int, int], ...]
