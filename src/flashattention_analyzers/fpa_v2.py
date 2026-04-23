from __future__ import annotations

import csv
import hashlib
import html
from pathlib import Path
from typing import Any

from flashattention_analyzers.cpp_tiling import CppTilingExtractor
from flashattention_analyzers.kernel_source import KernelSourceExtractor
from flashattention_models import BlockUnit, PlatformProfile, PromptFlashAttentionCase
from flashattention_utils import ceil_div, cpp_trunc_div, literal_parse


SPARSE_MODE_NO_MASK = 0
SPARSE_MODE_ALL_MASK = 1
SPARSE_MODE_LEFT_UP = 2
SPARSE_MODE_RIGHT_DOWN = 3
SPARSE_MODE_BAND = 4
SPARSE_MODE_INT_MAX = 2147483647

SOUTER_FACTOR_SUB = 32
SOUTER_FACTOR_DEFAULT = 64
SINNER_FACTOR_SUB = 64
SINNER_FACTOR_DEFAULT = 128
SINNER_FACTOR_DOUBLE = 256
CV_RATIO = 2


class PromptFlashAttentionV2Replayer:
    def __init__(self, repo_root: Path, source_root: Path | None = None) -> None:
        self.repo_root = repo_root.resolve()
        default_source_root = self.repo_root / "fixtures" / "prompt_flash_attention"
        if not default_source_root.exists():
            default_source_root = self.repo_root / "prompt_flash_attention"
        self.source_root = (source_root or default_source_root).resolve()
        self.op_host_root = self.source_root / "op_host"
        self.op_kernel_root = self.source_root / "op_kernel"
        self.header_path = self.op_host_root / "prompt_flash_attention_tiling.h"
        self.source_path = self.op_host_root / "prompt_flash_attention_tiling_v2.cpp"
        self.const_path = self.op_host_root / "prompt_flash_attention_tiling_const.h"
        self.tiling_register_path = self.op_host_root / "prompt_flash_attention_tiling_register.cpp"
        self.kernel_entry_path = self.op_kernel_root / "prompt_flash_attention.cpp"
        self.kernel_dispatch_path = self.op_kernel_root / "prompt_flash_attention_arch32.h"
        self.kernel_template_key_path = self.op_kernel_root / "arch35" / "prompt_flash_attention_template_tiling_key.h"
        self.extractor = CppTilingExtractor()
        self.kernel_extractor = KernelSourceExtractor()

    def load_cases(self, csv_path: Path) -> list[PromptFlashAttentionCase]:
        cases: list[PromptFlashAttentionCase] = []
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cases.append(self.parse_case_row(row))
        return cases

    def parse_case_row(self, row: dict[str, str]) -> PromptFlashAttentionCase:
        shapes = literal_parse(row["tensor_view_shapes"])
        dtypes = literal_parse(row["tensor_dtypes"])
        attributes = literal_parse(row["attributes"])

        layout = str(attributes.get("inputLayout", "BNSD"))
        query_shape = tuple(int(item) for item in shapes[0])
        key_shape = tuple(int(item) for item in shapes[1])
        value_shape = tuple(int(item) for item in shapes[2])
        output_shape = tuple(int(item) for item in shapes[-1]) if shapes[-1] is not None else None

        batch_size, _, seq_q, query_d = self._shape_info(query_shape, layout, int(attributes["numHeads"]))
        _, kv_heads_from_shape, seq_kv, value_d = self._shape_info(
            key_shape,
            layout,
            int(attributes.get("numKeyValueHeads") or attributes["numHeads"]),
        )

        head_num = int(attributes["numHeads"])
        kv_head_num = int(attributes.get("numKeyValueHeads") or kv_heads_from_shape or head_num)
        actual_seq_lengths = self._expand_lengths(attributes.get("actualSeqLengths"), batch_size, seq_q)
        actual_seq_lengths_kv = self._expand_lengths(attributes.get("actualSeqLengthsKv"), batch_size, seq_kv)

        return PromptFlashAttentionCase(
            case_id=row["testcase_name"],
            api_name=row["api_name"],
            layout=layout,
            query_shape=query_shape,
            key_shape=key_shape,
            value_shape=value_shape,
            output_shape=output_shape,
            input_dtype=str(dtypes[0]),
            key_dtype=str(dtypes[1]),
            value_dtype=str(dtypes[2]),
            output_dtype=str(dtypes[-1]),
            attributes=attributes,
            batch_size=batch_size,
            head_num=head_num,
            kv_head_num=kv_head_num,
            seq_q=seq_q,
            seq_kv=seq_kv,
            query_d=query_d,
            value_d=value_d,
            actual_seq_lengths=actual_seq_lengths,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            pre_tokens=int(attributes.get("preTokens", SPARSE_MODE_INT_MAX)),
            next_tokens=int(attributes.get("nextTokens", SPARSE_MODE_INT_MAX)),
            sparse_mode=int(attributes.get("sparseMode", SPARSE_MODE_NO_MASK)),
            has_attention_mask=len(shapes) > 4 and shapes[4] is not None,
            has_pse_shift=len(shapes) > 3 and shapes[3] is not None,
        )

    def analyze_source(self) -> dict[str, Any]:
        structs = self.extractor.extract_tiling_structs(self.header_path)
        struct_lookup = {struct.name: struct for struct in structs}
        seq_struct = struct_lookup["PromptAttentionSeqParams"]
        alias_by_field = {
            field.name: (field.comment or field.name) for field in seq_struct.fields
        }
        mappings = self.extractor.extract_assignment_mapping(
            self.source_path,
            receiver="seqParams",
            alias_by_field=alias_by_field,
        )
        constants = self.extractor.extract_constants(
            [self.source_path, self.const_path],
            {
                "SOUTER_FACTOR_SUB",
                "SOUTER_FACTOR_DEFAULT",
                "SINNER_FACTOR_SUB",
                "SINNER_FACTOR_DEFAULT",
                "SINNER_FACTOR_DOUBLE",
                "CV_RATIO",
                "SPARSE_MODE_NO_MASK",
                "SPARSE_MODE_ALL_MASK",
                "SPARSE_MODE_LEFT_UP",
                "SPARSE_MODE_RIGHT_DOWN",
                "SPARSE_MODE_BAND",
                "SPARSE_MODE_INT_MAX",
            },
        )
        python_mapping = [
            (
                "PromptFlashAttentionV2Replayer._select_split_factors",
                "PromptFlashAttentionTilingV2::AdjustCVTilingCVDiff",
            ),
            (
                "PromptFlashAttentionV2Replayer._sparse_tokens",
                "PromptFlashAttentionTilingV2::SetSparseModeData",
            ),
            (
                "PromptFlashAttentionV2Replayer._get_pre_next_tokens_left_up",
                "PromptFlashAttentionTilingV2::GetPreNextTokensLeftUp",
            ),
            (
                "PromptFlashAttentionV2Replayer._get_actual_inner_block_nums",
                "PromptFlashAttentionTilingV2::GetActualInnerBlockNums",
            ),
            (
                "PromptFlashAttentionV2Replayer._get_cut_block_nums",
                "PromptFlashAttentionTilingV2::GetCutBlockNums",
            ),
            (
                "PromptFlashAttentionV2Replayer._fix_param_with_row_invalid",
                "PromptFlashAttentionTilingV2::FixParamWithRowInvalid",
            ),
            (
                "PromptFlashAttentionV2Replayer._get_calc_block_nums_one_head",
                "PromptFlashAttentionTilingV2::GetCalcBlockNumsOneHead",
            ),
            (
                "PromptFlashAttentionV2Replayer._build_units",
                "PromptFlashAttentionTilingV2::ComputeSplitNBSeq",
            ),
            (
                "PromptFlashAttentionV2Replayer.replay_case",
                "PromptFlashAttentionTilingV2::PromptFlashAttentionSplitNBSeq",
            ),
        ]
        return {
            "operator": "prompt_flash_attention_v2",
            "source_root": str(self.source_root),
            "fixture_snapshot": self._fixture_snapshot(),
            "structs": [
                {
                    "name": struct.name,
                    "source": struct.source,
                    "fields": struct.fields,
                }
                for struct in structs
            ],
            "seq_params_aliases": alias_by_field,
            "seq_param_mappings": mappings,
            "constants": constants,
            "python_mapping": [
                {
                    "python_method": python_method,
                    "cpp_function": self._function_span(cpp_function),
                }
                for python_method, cpp_function in python_mapping
            ],
            "kernel_analysis": self.kernel_extractor.analyze(
                self.kernel_entry_path,
                self.kernel_dispatch_path,
                self.kernel_template_key_path,
            ),
        }

    def replay_csv(self, csv_path: Path, platform: PlatformProfile) -> dict[str, Any]:
        results = [self.replay_case(case, platform) for case in self.load_cases(csv_path)]
        return {
            "operator": "prompt_flash_attention_v2",
            "case_count": len(results),
            "platform": {
                "aiv_num": platform.aiv_num,
                "aic_num": platform.aic_num,
                "fa_run_flag": platform.fa_run_flag,
            },
            "source_analysis": self.analyze_source(),
            "cases": results,
        }

    def replay_case(self, case: PromptFlashAttentionCase, platform: PlatformProfile) -> dict[str, Any]:
        s_outer_factor, s_inner_factor, softmax_s_outer_factor = self._select_split_factors(case, platform)
        if self._enable_dn(case, s_outer_factor):
            if (
                case.query_d == case.value_d
                and case.query_d == 64
                and all(length % 32 == 0 for length in case.actual_seq_lengths)
                and all(length > 128 for length in case.actual_seq_lengths_kv)
            ):
                s_inner_factor = 256

        effective_s_outer = s_outer_factor * CV_RATIO
        sparse_pre_tokens, sparse_next_tokens = self._sparse_tokens(case)
        units = self._build_units(
            case,
            effective_s_outer,
            s_inner_factor,
            sparse_pre_tokens,
            sparse_next_tokens,
        )
        total_weight = sum(unit.weight for unit in units)
        vector_core_num = max(platform.aiv_num // CV_RATIO, 1)
        core_weight_target = total_weight / vector_core_num if vector_core_num else 0.0
        logical_assignments = self._split_units(units, core_weight_target, pair_size=CV_RATIO)
        physical_assignments = self._expand_physical_assignments(logical_assignments, CV_RATIO)
        validation = self._validate(case, units, logical_assignments)
        multi_smaxs_inner_loop_times = max(
            ceil_div(length, s_inner_factor) for length in case.actual_seq_lengths_kv
        )
        kernel_context = self._kernel_case_context(case, s_outer_factor, s_inner_factor, platform)
        physical_assignments = self._attach_kernel_execution(physical_assignments, kernel_context)
        source_hash = self._source_hash()
        return {
            "case_id": case.case_id,
            "api_name": case.api_name,
            "layout": case.layout,
            "query_shape": list(case.query_shape),
            "key_shape": list(case.key_shape),
            "value_shape": list(case.value_shape),
            "output_dtype": case.output_dtype,
            "actual_seq_lengths": case.actual_seq_lengths,
            "actual_seq_lengths_kv": case.actual_seq_lengths_kv,
            "available_physical_cores": platform.aiv_num,
            "used_physical_cores": len(physical_assignments),
            "logical_core_groups": len(logical_assignments),
            "core_pair_size": CV_RATIO,
            "split_core_mode": "SPLIT_NBS_CUBE",
            "tiling": {
                "singleProcessSOuterSize": s_outer_factor,
                "singleProcessSInnerSize": s_inner_factor,
                "effectiveSplitSOuterSize": effective_s_outer,
                "softmaxSOuterSize": softmax_s_outer_factor,
                "multiSmaxsInnerLoopTimes": multi_smaxs_inner_loop_times,
                "headNumRatio": case.head_num // case.kv_head_num if case.kv_head_num else 1,
                "sparseMode": case.sparse_mode,
                "sparsePreTokens": sparse_pre_tokens,
                "sparseNextTokens": sparse_next_tokens,
            },
            "logical_core_assignments": logical_assignments,
            "core_assignments": physical_assignments,
            "validation": validation,
            "kernel_execution_model": kernel_context,
            "traceability": {
                "source_hash": source_hash,
                "seq_params_struct": self._struct_span("PromptAttentionSeqParams"),
                "split_function": self._function_span(
                    "PromptFlashAttentionTilingV2::PromptFlashAttentionSplitNBSeq"
                ),
                "compute_function": self._function_span(
                    "PromptFlashAttentionTilingV2::ComputeSplitNBSeq"
                ),
                "tiling_key_setter": self._function_span("PromptFlashAttentionTilingV2::SetTilingKey"),
                "kernel_entry": self.kernel_extractor.find_function_span(
                    self.kernel_entry_path,
                    "prompt_flash_attention_FIAS",
                ),
                "kernel_dispatch": self.kernel_extractor.find_function_span(
                    self.kernel_dispatch_path,
                    "prompt_flash_attention_FIAS_arch32",
                ),
                "block_dim_setter": self._find_line_span(self.source_path, "context_->SetBlockDim(", "SetBlockDim"),
            },
        }

    def summary_rows(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for case in payload["cases"]:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "layout": case["layout"],
                    "used_physical_cores": case["used_physical_cores"],
                    "logical_core_groups": case["logical_core_groups"],
                    "s_outer": case["tiling"]["singleProcessSOuterSize"],
                    "s_inner": case["tiling"]["singleProcessSInnerSize"],
                    "effective_s_outer": case["tiling"]["effectiveSplitSOuterSize"],
                    "coverage_ok": case["validation"]["coverage_ok"],
                    "weight_ok": case["validation"]["weighted_coverage_ok"],
                }
            )
        return rows

    def render_case_svg(self, case_result: dict[str, Any], output_path: Path) -> None:
        assignments = case_result["core_assignments"]
        units = int(case_result["validation"]["total_unit_count"])
        width = 1840
        margin = 32
        header_height = 98
        content_width = width - (margin * 2)
        coverage_bar_width = 180
        cell_size = self._heatmap_cell_size(case_result)
        pane_content_width = content_width - coverage_bar_width - 44
        sections: list[dict[str, Any]] = []
        current_y = header_height
        for index, assignment in enumerate(assignments):
            panes = self._assignment_heatmap_panes(assignment)
            pane_layouts, pane_height = self._layout_heatmap_panes(
                panes,
                start_x=margin + coverage_bar_width + 28,
                start_y=current_y + 46,
                max_width=pane_content_width,
                cell_size=cell_size,
            )
            section_height = max(84, 58 + pane_height)
            sections.append(
                {
                    "index": index,
                    "assignment": assignment,
                    "panes": pane_layouts,
                    "top": current_y,
                    "height": section_height,
                }
            )
            current_y += section_height + 12
        height = current_y + 24
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc" />',
            f'<text x="{margin}" y="28" font-size="18" font-family="monospace" fill="#111827">{html.escape(case_result["case_id"])}</text>',
            (
                f'<text x="{margin}" y="48" font-size="12" font-family="monospace" fill="#334155">'
                f'used_physical_cores={case_result["used_physical_cores"]}, '
                f'logical_core_groups={case_result["logical_core_groups"]}, '
                f's_outer={case_result["tiling"]["singleProcessSOuterSize"]}, '
                f's_inner={case_result["tiling"]["singleProcessSInnerSize"]}'
                "</text>"
            ),
            (
                f'<text x="{margin}" y="68" font-size="12" font-family="monospace" fill="#334155">'
                "Each section is one physical core. Left: unit-index coverage. Right: per-(batch,head) Q x KV block heatmaps."
                "</text>"
            ),
            (
                f'<text x="{margin}" y="86" font-size="11" font-family="monospace" fill="#475569">'
                "Inside each pane, y = Souter block index, x = Sinner block index, filled cells = blocks executed by this core."
                "</text>"
            ),
        ]
        for section in sections:
            assignment = section["assignment"]
            top = int(section["top"])
            section_height = int(section["height"])
            row_fill = "#ffffff" if section["index"] % 2 == 0 else "#f1f5f9"
            logical_core_id = int(assignment["logical_core_id"])
            start = int(assignment["range_start_unit_index"])
            end = int(assignment["range_end_unit_index"])
            scale = coverage_bar_width / max(units, 1)
            bar_x = margin
            bar_y = top + 40
            bar_height = 14
            cover_x = bar_x + (start * scale)
            cover_width = max((end - start) * scale, 1.5)
            core_label = (
                f'C{int(assignment["core_id"]):02d} {assignment["lane_role"]} '
                f'| L{logical_core_id} | units[{start},{end}) | weight={assignment["total_weight"]}'
            )
            summary = self._truncate_text(str(assignment["task_summary"]), 160)
            parts.append(
                f'<rect x="{margin - 8}" y="{top}" width="{content_width + 16}" height="{section_height}" rx="8" fill="{row_fill}" stroke="#cbd5e1" stroke-width="1" />'
            )
            parts.append(
                f'<text x="{margin}" y="{top + 20}" font-size="12" font-family="monospace" fill="#0f172a">{html.escape(core_label)}</text>'
            )
            parts.append(
                f'<text x="{margin}" y="{top + 34}" font-size="10" font-family="monospace" fill="#475569">{html.escape(summary)}</text>'
            )
            parts.append(
                f'<rect x="{bar_x}" y="{bar_y}" width="{coverage_bar_width}" height="{bar_height}" rx="4" fill="#e2e8f0" />'
            )
            parts.append(
                f'<rect x="{cover_x:.2f}" y="{bar_y}" width="{cover_width:.2f}" height="{bar_height}" rx="4" fill="{self._color_for_index(logical_core_id)}" stroke="#1f2937" stroke-width="0.5" />'
            )
            for tick in range(0, units + 1, max(1, units // 8 or 1)):
                tick_x = bar_x + (tick * scale)
                parts.append(
                    f'<line x1="{tick_x:.2f}" y1="{bar_y + bar_height + 2}" x2="{tick_x:.2f}" y2="{bar_y + bar_height + 6}" stroke="#64748b" stroke-width="1" />'
                )
                parts.append(
                    f'<text x="{tick_x:.2f}" y="{bar_y + bar_height + 18}" text-anchor="middle" font-size="9" fill="#64748b">{tick}</text>'
                )
            for pane_layout in section["panes"]:
                self._render_heatmap_pane(
                    parts,
                    pane_layout,
                    cell_size,
                    self._color_for_index(logical_core_id),
                )
        parts.append("</svg>")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(parts), encoding="utf-8")

    def _shape_info(self, shape: tuple[int, ...], layout: str, head_num: int) -> tuple[int, int, int, int]:
        if layout == "BNSD":
            return int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
        if layout == "BSND":
            return int(shape[0]), int(shape[2]), int(shape[1]), int(shape[3])
        if layout == "BSH":
            batch_size = int(shape[0])
            seq = int(shape[1])
            hidden = int(shape[2])
            dim = hidden // head_num
            return batch_size, head_num, seq, dim
        raise ValueError(f"unsupported layout for replay: {layout}")

    def _expand_lengths(self, raw_value: Any, batch_size: int, fallback: int) -> list[int]:
        if raw_value is None:
            values = [fallback]
        else:
            values = [int(item) for item in tuple(raw_value)]
            if not values:
                values = [fallback]
        if len(values) == 1 and batch_size > 1:
            return values * batch_size
        if len(values) != batch_size:
            raise ValueError(
                f"actual sequence length size {len(values)} does not match batch size {batch_size}"
            )
        return values

    def _select_split_factors(
        self, case: PromptFlashAttentionCase, platform: PlatformProfile
    ) -> tuple[int, int, int]:
        min_factor = SOUTER_FACTOR_DEFAULT
        rectangle_factor = SINNER_FACTOR_DEFAULT
        softmax_s_outer_factor = SOUTER_FACTOR_DEFAULT
        sparse_pre_tokens, sparse_next_tokens = self._sparse_tokens(case)
        if case.value_d <= 128:
            check_dtype = case.input_dtype in {"float16", "bfloat16"}
            check_query_and_value_s = case.seq_q <= SOUTER_FACTOR_DEFAULT and case.seq_kv > SINNER_FACTOR_DEFAULT
            sparse_sum = sparse_pre_tokens + sparse_next_tokens
            check_sparse_mode = (
                case.sparse_mode in {SPARSE_MODE_ALL_MASK, SPARSE_MODE_RIGHT_DOWN}
                or (
                    case.sparse_mode in {SPARSE_MODE_NO_MASK, SPARSE_MODE_BAND}
                    and sparse_sum > SINNER_FACTOR_DEFAULT
                )
            )
            if check_dtype and check_query_and_value_s and check_sparse_mode and not case.has_pse_shift:
                min_factor = SOUTER_FACTOR_SUB
                rectangle_factor = SINNER_FACTOR_DOUBLE
                softmax_s_outer_factor = SOUTER_FACTOR_SUB
        elif not platform.fa_run_flag:
            min_factor = SOUTER_FACTOR_SUB
            rectangle_factor = SINNER_FACTOR_SUB
            softmax_s_outer_factor = SOUTER_FACTOR_SUB
        else:
            softmax_s_outer_factor = SOUTER_FACTOR_SUB
        return min_factor, rectangle_factor, softmax_s_outer_factor

    def _enable_dn(self, case: PromptFlashAttentionCase, s_outer_factor: int) -> bool:
        return (
            not case.has_attention_mask
            and not case.has_pse_shift
            and case.query_d <= 128
            and case.value_d <= 128
            and case.input_dtype in {"float16", "bfloat16"}
            and (s_outer_factor * 2 > 64)
        )

    def _sparse_tokens(self, case: PromptFlashAttentionCase) -> tuple[int, int]:
        sparse_pre = max(min(case.pre_tokens, SPARSE_MODE_INT_MAX), -SPARSE_MODE_INT_MAX)
        sparse_next = max(min(case.next_tokens, SPARSE_MODE_INT_MAX), -SPARSE_MODE_INT_MAX)
        if case.sparse_mode == SPARSE_MODE_LEFT_UP:
            return SPARSE_MODE_INT_MAX, 0
        if case.sparse_mode == SPARSE_MODE_RIGHT_DOWN:
            return SPARSE_MODE_INT_MAX, sparse_next
        if case.sparse_mode == SPARSE_MODE_ALL_MASK:
            return SPARSE_MODE_INT_MAX, SPARSE_MODE_INT_MAX
        if case.sparse_mode == SPARSE_MODE_NO_MASK and not case.has_attention_mask:
            return SPARSE_MODE_INT_MAX, SPARSE_MODE_INT_MAX
        return sparse_pre, sparse_next

    def _get_pre_next_tokens_left_up(
        self,
        sparse_mode: int,
        sparse_pre_tokens: int,
        sparse_next_tokens: int,
        actual_seq_length: int,
        actual_seq_length_kv: int,
    ) -> tuple[int, int]:
        if sparse_mode == SPARSE_MODE_RIGHT_DOWN:
            return SPARSE_MODE_INT_MAX, actual_seq_length_kv - actual_seq_length
        if sparse_mode == SPARSE_MODE_BAND:
            return (
                sparse_pre_tokens - actual_seq_length_kv + actual_seq_length,
                sparse_next_tokens + actual_seq_length_kv - actual_seq_length,
            )
        return sparse_pre_tokens, sparse_next_tokens

    def _get_actual_inner_block_nums(self, start: int, end: int, inner_block_nums: int) -> int:
        if end < 0:
            return 0
        if end < inner_block_nums:
            return end + 1 if start < 0 else end - start + 1
        if start < 0:
            return inner_block_nums
        return inner_block_nums - start if start < inner_block_nums else 0

    def _sum_of_arithmetic_series(self, an: int, step: int) -> int:
        if step == 0 or an <= 0:
            return 0
        return (an % step + an) * (an // step + 1) // 2

    def _get_cut_block_nums(
        self,
        block_seq_length_kv: int,
        block_seq_length: int,
        s_inner: int,
        s_outer: int,
        token: int,
    ) -> int:
        block_token = (
            ceil_div(token, s_inner) * s_inner if token > 0 else cpp_trunc_div(token, s_inner) * s_inner
        )
        out_div_in = s_outer // s_inner if s_outer > s_inner else 1
        in_div_out = s_inner // s_outer if s_inner > s_outer else 1
        tolerance = out_div_in if out_div_in >= 1 else in_div_out
        small_size = s_inner if out_div_in >= 1 else s_outer
        block_nums = 0
        block_nums += self._sum_of_arithmetic_series(
            (block_seq_length_kv - block_token) // small_size - tolerance,
            tolerance,
        )
        block_nums -= self._sum_of_arithmetic_series(
            (-block_token) // small_size - tolerance,
            tolerance,
        )
        block_nums -= self._sum_of_arithmetic_series(
            (block_seq_length_kv - block_seq_length - block_token) // small_size - tolerance,
            tolerance,
        )
        block_nums += self._sum_of_arithmetic_series(
            (-block_token - block_seq_length) // small_size - tolerance,
            tolerance,
        )
        return block_nums

    def _fix_param_with_row_invalid(
        self,
        actual_seq_length: int,
        actual_seq_length_kv: int,
        pre_tokens_left_up: int,
        next_tokens_left_up: int,
    ) -> tuple[int, int, int]:
        next_tokens_error = min(-next_tokens_left_up if next_tokens_left_up < 0 else 0, actual_seq_length)
        pre_tokens_error = (
            actual_seq_length - actual_seq_length_kv - pre_tokens_left_up
            if actual_seq_length > actual_seq_length_kv + pre_tokens_left_up
            else 0
        )
        pre_tokens_error = min(pre_tokens_error, actual_seq_length)
        next_tokens_left_up += next_tokens_error
        pre_tokens_left_up -= next_tokens_error
        actual_seq_length -= next_tokens_error
        actual_seq_length -= pre_tokens_error
        return actual_seq_length, pre_tokens_left_up, next_tokens_left_up

    def _get_calc_block_nums_one_head(
        self,
        actual_seq_length: int,
        actual_seq_length_kv: int,
        s_outer_size: int,
        s_inner_size: int,
        pre_tokens_left_up: int,
        next_tokens_left_up: int,
        is_attention_mask_used: bool,
    ) -> int:
        if not is_attention_mask_used:
            return ceil_div(actual_seq_length_kv, s_inner_size) * ceil_div(actual_seq_length, s_outer_size)
        inner_block_nums = ceil_div(actual_seq_length_kv, s_inner_size)
        block_seq_length_kv = inner_block_nums * s_inner_size
        outer_block_nums = ceil_div(actual_seq_length, s_outer_size)
        block_seq_length = outer_block_nums * s_outer_size
        to_calc_block_nums = inner_block_nums * outer_block_nums
        to_calc_block_nums -= self._get_cut_block_nums(
            block_seq_length_kv,
            block_seq_length,
            s_inner_size,
            s_outer_size,
            next_tokens_left_up,
        )
        to_calc_block_nums -= self._get_cut_block_nums(
            block_seq_length_kv,
            block_seq_length,
            s_inner_size,
            s_outer_size,
            block_seq_length_kv - block_seq_length + pre_tokens_left_up,
        )
        return to_calc_block_nums

    def _build_units(
        self,
        case: PromptFlashAttentionCase,
        effective_s_outer: int,
        s_inner_factor: int,
        sparse_pre_tokens: int,
        sparse_next_tokens: int,
    ) -> list[BlockUnit]:
        units: list[BlockUnit] = []
        for sid in range(case.batch_size):
            for nid in range(case.head_num):
                pre_tokens_left_up, next_tokens_left_up = self._get_pre_next_tokens_left_up(
                    case.sparse_mode,
                    sparse_pre_tokens,
                    sparse_next_tokens,
                    case.actual_seq_lengths[sid],
                    case.actual_seq_lengths_kv[sid],
                )
                actual_seq_length = case.actual_seq_lengths[sid]
                actual_seq_length_kv = case.actual_seq_lengths_kv[sid]
                actual_seq_length, pre_tokens_left_up, next_tokens_left_up = self._fix_param_with_row_invalid(
                    actual_seq_length,
                    actual_seq_length_kv,
                    pre_tokens_left_up,
                    next_tokens_left_up,
                )
                outer_block_nums = ceil_div(actual_seq_length, effective_s_outer)
                inner_block_nums = ceil_div(actual_seq_length_kv, s_inner_factor)
                for spos in range(outer_block_nums):
                    q_token_start = spos * effective_s_outer
                    q_token_end = min(q_token_start + effective_s_outer, actual_seq_length)
                    if case.has_attention_mask:
                        start_index = -(
                            ceil_div(pre_tokens_left_up, s_inner_factor)
                            if pre_tokens_left_up > 0
                            else cpp_trunc_div(pre_tokens_left_up, s_inner_factor)
                        )
                        end_index = (
                            ceil_div(next_tokens_left_up, s_inner_factor)
                            if next_tokens_left_up > 0
                            else cpp_trunc_div(next_tokens_left_up, s_inner_factor)
                        )
                        active_inner_ranges = self._active_inner_ranges(
                            start_index,
                            end_index,
                            inner_block_nums,
                        )
                    else:
                        active_inner_ranges = ((0, inner_block_nums),) if inner_block_nums > 0 else tuple()
                    active_kv_ranges = self._inner_ranges_to_token_ranges(
                        active_inner_ranges,
                        s_inner_factor,
                        actual_seq_length_kv,
                    )
                    weight = sum(range_end - range_start for range_start, range_end in active_inner_ranges)
                    units.append(
                        BlockUnit(
                            sid=sid,
                            nid=nid,
                            spos=spos,
                            weight=weight,
                            q_token_start=q_token_start,
                            q_token_end=q_token_end,
                            outer_block_count=outer_block_nums,
                            inner_block_count=inner_block_nums,
                            active_inner_block_ranges=active_inner_ranges,
                            active_kv_token_ranges=active_kv_ranges,
                        )
                    )
                    pre_tokens_left_up -= effective_s_outer
                    next_tokens_left_up += effective_s_outer
        return units

    def _split_units(
        self, units: list[BlockUnit], core_weight_target: float, pair_size: int
    ) -> list[dict[str, Any]]:
        if not units:
            return []
        assignments: list[dict[str, Any]] = []
        cur_core = 0
        current_start_unit_index = 0
        previous_unit: BlockUnit | None = None
        current_weight = 0
        for unit_index, unit in enumerate(units):
            dif = int(core_weight_target * float(cur_core + 1)) - current_weight
            if unit.weight - dif > dif and previous_unit is not None:
                assignments.append(
                    self._make_assignment(
                        logical_core_id=cur_core,
                        start_unit=units[current_start_unit_index],
                        end_unit=previous_unit,
                        start_index=current_start_unit_index,
                        end_index=unit_index,
                        units=units[current_start_unit_index:unit_index],
                        pair_size=pair_size,
                    )
                )
                cur_core += 1
                current_start_unit_index = unit_index
            previous_unit = unit
            current_weight += unit.weight
        assignments.append(
            self._make_assignment(
                logical_core_id=cur_core,
                start_unit=units[current_start_unit_index],
                end_unit=previous_unit or units[-1],
                start_index=current_start_unit_index,
                end_index=len(units),
                units=units[current_start_unit_index:],
                pair_size=pair_size,
            )
        )
        return assignments

    def _make_assignment(
        self,
        logical_core_id: int,
        start_unit: BlockUnit,
        end_unit: BlockUnit,
        start_index: int,
        end_index: int,
        units: list[BlockUnit],
        pair_size: int,
    ) -> dict[str, Any]:
        total_weight = sum(unit.weight for unit in units)
        task_units = [self._unit_payload(unit) for unit in units]
        task_segments = self._segment_payloads(units)
        return {
            "logical_core_id": logical_core_id,
            "group_size": pair_size,
            "coreNidStart": start_unit.nid,
            "coreNidEnd": end_unit.nid + 1,
            "coreSidStart": start_unit.sid,
            "coreSidEnd": end_unit.sid + 1,
            "coreSposStart": start_unit.spos,
            "coreSposEnd": end_unit.spos + 1,
            "Nid": [start_unit.nid, end_unit.nid + 1],
            "Sid": [start_unit.sid, end_unit.sid + 1],
            "Spos": [start_unit.spos, end_unit.spos + 1],
            "range_start_unit_index": start_index,
            "range_end_unit_index": end_index,
            "unit_count": end_index - start_index,
            "total_weight": total_weight,
            "task_units": task_units,
            "task_segments": task_segments,
            "task_summary": self._format_assignment_summary(task_segments),
        }

    def _expand_physical_assignments(
        self, logical_assignments: list[dict[str, Any]], pair_size: int
    ) -> list[dict[str, Any]]:
        expanded: list[dict[str, Any]] = []
        lane_roles = ["vector", "cube"]
        for assignment in logical_assignments:
            for lane_id in range(pair_size):
                item = dict(assignment)
                item["core_id"] = assignment["logical_core_id"] * pair_size + lane_id
                item["lane_id"] = lane_id
                item["lane_role"] = lane_roles[lane_id] if lane_id < len(lane_roles) else f"lane-{lane_id}"
                expanded.append(item)
        return expanded

    def _validate(
        self,
        case: PromptFlashAttentionCase,
        units: list[BlockUnit],
        logical_assignments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        coverage_ok = bool(logical_assignments) and logical_assignments[0]["range_start_unit_index"] == 0
        no_overlap = True
        no_gap = True
        previous_end = 0
        for assignment in logical_assignments:
            start = int(assignment["range_start_unit_index"])
            end = int(assignment["range_end_unit_index"])
            if start < previous_end:
                no_overlap = False
            if start != previous_end:
                no_gap = False
            previous_end = end
        if previous_end != len(units):
            coverage_ok = False
            no_gap = False
        else:
            coverage_ok = coverage_ok and no_overlap and no_gap
        total_weight = sum(unit.weight for unit in units)
        covered_weight = sum(int(item["total_weight"]) for item in logical_assignments)
        return {
            "coverage_ok": coverage_ok,
            "no_gap": no_gap,
            "no_overlap": no_overlap,
            "weighted_coverage_ok": total_weight == covered_weight,
            "total_unit_count": len(units),
            "covered_unit_count": sum(int(item["unit_count"]) for item in logical_assignments),
            "total_weight": total_weight,
            "covered_weight": covered_weight,
            "batch_size": case.batch_size,
            "head_num": case.head_num,
        }

    def _active_inner_ranges(
        self,
        start_index: int,
        end_index: int,
        inner_block_nums: int,
    ) -> tuple[tuple[int, int], ...]:
        start = max(start_index, 0)
        end = min(end_index + 1, inner_block_nums)
        if end <= start:
            return tuple()
        return ((start, end),)

    def _inner_ranges_to_token_ranges(
        self,
        inner_ranges: tuple[tuple[int, int], ...],
        s_inner_factor: int,
        actual_seq_length_kv: int,
    ) -> tuple[tuple[int, int], ...]:
        token_ranges: list[tuple[int, int]] = []
        for range_start, range_end in inner_ranges:
            token_start = range_start * s_inner_factor
            token_end = min(range_end * s_inner_factor, actual_seq_length_kv)
            token_ranges.append((token_start, token_end))
        return tuple(token_ranges)

    def _unit_payload(self, unit: BlockUnit) -> dict[str, Any]:
        return {
            "sid": unit.sid,
            "nid": unit.nid,
            "spos": unit.spos,
            "Spos": [unit.spos, unit.spos + 1],
            "query_tokens": [unit.q_token_start, unit.q_token_end],
            "grid_shape": {
                "q_blocks": unit.outer_block_count,
                "kv_blocks": unit.inner_block_count,
            },
            "active_inner_blocks": [list(item) for item in unit.active_inner_block_ranges],
            "active_kv_tokens": [list(item) for item in unit.active_kv_token_ranges],
            "weight": unit.weight,
            "task_brief": self._format_unit_brief(unit),
        }

    def _segment_payloads(self, units: list[BlockUnit]) -> list[dict[str, Any]]:
        if not units:
            return []
        segments: list[dict[str, Any]] = []
        for unit in units:
            if (
                segments
                and segments[-1]["sid"] == unit.sid
                and segments[-1]["nid"] == unit.nid
                and segments[-1]["active_inner_blocks"] == [list(item) for item in unit.active_inner_block_ranges]
                and segments[-1]["active_kv_tokens"] == [list(item) for item in unit.active_kv_token_ranges]
                and segments[-1]["spos"][1] == unit.spos
                and segments[-1]["query_tokens"][1] == unit.q_token_start
            ):
                segments[-1]["spos"][1] = unit.spos + 1
                segments[-1]["query_tokens"][1] = unit.q_token_end
                segments[-1]["unit_count"] += 1
                segments[-1]["total_weight"] += unit.weight
                segments[-1]["task_brief"] = self._format_segment_brief(segments[-1])
                continue
            segment = {
                "sid": unit.sid,
                "nid": unit.nid,
                "spos": [unit.spos, unit.spos + 1],
                "query_tokens": [unit.q_token_start, unit.q_token_end],
                "grid_shape": {
                    "q_blocks": unit.outer_block_count,
                    "kv_blocks": unit.inner_block_count,
                },
                "active_inner_blocks": [list(item) for item in unit.active_inner_block_ranges],
                "active_kv_tokens": [list(item) for item in unit.active_kv_token_ranges],
                "unit_count": 1,
                "total_weight": unit.weight,
            }
            segment["task_brief"] = self._format_segment_brief(segment)
            segments.append(segment)
        return segments

    def _format_assignment_summary(
        self,
        task_segments: list[dict[str, Any]],
        max_segments: int = 3,
    ) -> str:
        if not task_segments:
            return "no work assigned"
        rendered = [str(item["task_brief"]) for item in task_segments[:max_segments]]
        if len(task_segments) > max_segments:
            rendered.append(f"... +{len(task_segments) - max_segments} more")
        return " | ".join(rendered)

    def _format_unit_brief(self, unit: BlockUnit) -> str:
        return (
            f'B{unit.sid} H{unit.nid} Spos[{unit.spos},{unit.spos + 1}) '
            f'Q[{unit.q_token_start},{unit.q_token_end}) '
            f'KV{self._format_ranges(unit.active_kv_token_ranges)} '
            f'w={unit.weight}'
        )

    def _format_segment_brief(self, segment: dict[str, Any]) -> str:
        return (
            f'B{segment["sid"]} H{segment["nid"]} '
            f'Spos[{segment["spos"][0]},{segment["spos"][1]}) '
            f'Q[{segment["query_tokens"][0]},{segment["query_tokens"][1]}) '
            f'KV{self._format_ranges(tuple(tuple(item) for item in segment["active_kv_tokens"]))} '
            f'w={segment["total_weight"]}'
        )

    def _format_ranges(self, ranges: tuple[tuple[int, int], ...]) -> str:
        if not ranges:
            return "[]"
        return "[" + ",".join(f"[{start},{end})" for start, end in ranges) + "]"

    def _heatmap_cell_size(self, case_result: dict[str, Any]) -> int:
        max_kv_blocks = 1
        for assignment in case_result["core_assignments"]:
            for unit in assignment["task_units"]:
                max_kv_blocks = max(max_kv_blocks, int(unit["grid_shape"]["kv_blocks"]))
        return max(6, min(12, int(240 / max_kv_blocks) if max_kv_blocks else 12))

    def _assignment_heatmap_panes(self, assignment: dict[str, Any]) -> list[dict[str, Any]]:
        panes: list[dict[str, Any]] = []
        lookup: dict[tuple[int, int], dict[str, Any]] = {}
        for unit in assignment["task_units"]:
            key = (int(unit["sid"]), int(unit["nid"]))
            if key not in lookup:
                pane = {
                    "sid": key[0],
                    "nid": key[1],
                    "q_blocks": int(unit["grid_shape"]["q_blocks"]),
                    "kv_blocks": int(unit["grid_shape"]["kv_blocks"]),
                    "units": [],
                }
                lookup[key] = pane
                panes.append(pane)
            pane = lookup[key]
            pane["q_blocks"] = max(pane["q_blocks"], int(unit["grid_shape"]["q_blocks"]))
            pane["kv_blocks"] = max(pane["kv_blocks"], int(unit["grid_shape"]["kv_blocks"]))
            pane["units"].append(unit)
        return panes

    def _layout_heatmap_panes(
        self,
        panes: list[dict[str, Any]],
        start_x: int,
        start_y: int,
        max_width: int,
        cell_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if not panes:
            return ([], 0)
        layouts: list[dict[str, Any]] = []
        current_x = start_x
        current_y = start_y
        row_height = 0
        for pane in panes:
            pane_width = 40 + (int(pane["kv_blocks"]) * cell_size)
            pane_height = 26 + (int(pane["q_blocks"]) * cell_size)
            if current_x > start_x and current_x + pane_width > start_x + max_width:
                current_x = start_x
                current_y += row_height + 14
                row_height = 0
            layouts.append(
                {
                    "pane": pane,
                    "x": current_x,
                    "y": current_y,
                    "width": pane_width,
                    "height": pane_height,
                }
            )
            current_x += pane_width + 18
            row_height = max(row_height, pane_height)
        return (layouts, (current_y - start_y) + row_height)

    def _render_heatmap_pane(
        self,
        parts: list[str],
        pane_layout: dict[str, Any],
        cell_size: int,
        fill_color: str,
    ) -> None:
        pane = pane_layout["pane"]
        x = int(pane_layout["x"])
        y = int(pane_layout["y"])
        q_blocks = int(pane["q_blocks"])
        kv_blocks = int(pane["kv_blocks"])
        grid_x = x + 28
        grid_y = y + 16
        grid_width = kv_blocks * cell_size
        grid_height = q_blocks * cell_size
        parts.append(
            f'<rect x="{x}" y="{y}" width="{pane_layout["width"]}" height="{pane_layout["height"]}" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x + 6}" y="{y + 12}" font-size="10" font-family="monospace" fill="#0f172a">B{pane["sid"]} H{pane["nid"]}</text>'
        )
        parts.append(
            f'<rect x="{grid_x}" y="{grid_y}" width="{grid_width}" height="{grid_height}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" />'
        )
        for unit in pane["units"]:
            row = int(unit["spos"])
            if row >= q_blocks:
                continue
            for block_start, block_end in unit["active_inner_blocks"]:
                fill_x = grid_x + (int(block_start) * cell_size)
                fill_width = max((int(block_end) - int(block_start)) * cell_size, 1)
                fill_y = grid_y + (row * cell_size)
                parts.append(
                    f'<rect x="{fill_x}" y="{fill_y}" width="{fill_width}" height="{cell_size}" fill="{fill_color}" fill-opacity="0.78" />'
                )
        for row in range(q_blocks + 1):
            line_y = grid_y + (row * cell_size)
            parts.append(
                f'<line x1="{grid_x}" y1="{line_y}" x2="{grid_x + grid_width}" y2="{line_y}" stroke="#cbd5e1" stroke-width="0.6" />'
            )
        for col in range(kv_blocks + 1):
            line_x = grid_x + (col * cell_size)
            parts.append(
                f'<line x1="{line_x}" y1="{grid_y}" x2="{line_x}" y2="{grid_y + grid_height}" stroke="#cbd5e1" stroke-width="0.6" />'
            )
        if q_blocks <= 8:
            for row in range(q_blocks):
                label_y = grid_y + (row * cell_size) + (cell_size * 0.72)
                parts.append(
                    f'<text x="{x + 4}" y="{label_y:.2f}" font-size="8" font-family="monospace" fill="#64748b">{row}</text>'
                )
        else:
            parts.append(
                f'<text x="{x + 4}" y="{grid_y + 8}" font-size="8" font-family="monospace" fill="#64748b">0</text>'
            )
            parts.append(
                f'<text x="{x + 4}" y="{grid_y + grid_height - 2}" font-size="8" font-family="monospace" fill="#64748b">{q_blocks - 1}</text>'
            )
        if kv_blocks <= 12:
            for col in range(kv_blocks):
                label_x = grid_x + (col * cell_size) + (cell_size / 2)
                parts.append(
                    f'<text x="{label_x:.2f}" y="{grid_y - 3}" text-anchor="middle" font-size="8" font-family="monospace" fill="#64748b">{col}</text>'
                )
        else:
            parts.append(
                f'<text x="{grid_x}" y="{grid_y - 3}" font-size="8" font-family="monospace" fill="#64748b">0</text>'
            )
            parts.append(
                f'<text x="{grid_x + grid_width}" y="{grid_y - 3}" text-anchor="end" font-size="8" font-family="monospace" fill="#64748b">{kv_blocks - 1}</text>'
            )
        parts.append(
            f'<text x="{grid_x}" y="{grid_y + grid_height + 12}" font-size="8" font-family="monospace" fill="#64748b">qblk x kvblk</text>'
        )

    def _fixture_snapshot(self) -> dict[str, Any]:
        expected_dirs = [
            "docs",
            "examples",
            "framework",
            "op_api",
            "op_graph",
            "op_host",
            "op_kernel",
            "tests",
        ]
        present_dirs = [name for name in expected_dirs if (self.source_root / name).exists()]
        fixture_manifest = self._manifest_for_root(self.source_root)
        return {
            "full_operator_snapshot": present_dirs == expected_dirs,
            "top_level_dirs": present_dirs,
            "file_count": len(fixture_manifest),
            "manifest_sha256": self._manifest_sha256(fixture_manifest),
            "origin": "https://gitcode.com/cann/ops-transformer/tree/master/attention/prompt_flash_attention",
            "workspace_sync": self._workspace_sync_status(),
        }

    def _kernel_case_context(
        self,
        case: PromptFlashAttentionCase,
        s_outer_factor: int,
        s_inner_factor: int,
        platform: PlatformProfile,
    ) -> dict[str, Any]:
        precision_mode = self._precision_mode_hint(case)
        dispatch_candidates = self.kernel_extractor.match_dispatch_candidates(
            self.kernel_dispatch_path,
            case_layout=case.layout,
            split_mode="CUBEVECTORDIFF",
            input_dtype=case.input_dtype,
            key_dtype=case.key_dtype,
            output_dtype=case.output_dtype,
            precision_mode=precision_mode,
            high_level_api=case.api_name.startswith("aclnn"),
            has_attention_mask=case.has_attention_mask,
        )
        return {
            "entry_function": "prompt_flash_attention_FIAS",
            "arch_dispatch": "prompt_flash_attention_FIAS_arch32",
            "task_type": self.kernel_extractor.extract_task_type(self.kernel_dispatch_path),
            "shared_logical_assignment": True,
            "lane_roles": ["vector", "cube"],
            "pair_size": CV_RATIO,
            "block_dim": platform.aiv_num,
            "tiling_key_components": {
                "layout": case.layout,
                "split_mode": "CUBEVECTORDIFF",
                "precision_mode": precision_mode,
                "config": self._tiling_key_config_hint(case, s_outer_factor, s_inner_factor),
                "query_dtype": case.input_dtype,
                "key_dtype": case.key_dtype,
                "output_dtype": case.output_dtype,
                "has_attention_mask": case.has_attention_mask,
                "has_pse_shift": case.has_pse_shift,
            },
            "candidate_dispatches": dispatch_candidates,
            "notes": [
                "SPLIT_NBS_CUBE expands one logical tiling group into a vector lane and a cube lane.",
                "The host-side tiling arrays describe the shared logical region, while the kernel dispatch decides which mixed vector/cube implementation consumes it.",
            ],
        }

    def _attach_kernel_execution(
        self,
        assignments: list[dict[str, Any]],
        kernel_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        attached: list[dict[str, Any]] = []
        for assignment in assignments:
            item = dict(assignment)
            lane_role = str(item["lane_role"])
            paired_core_id = (
                item["core_id"] + 1 if int(item["lane_id"]) % 2 == 0 else item["core_id"] - 1
            )
            item["kernel_execution"] = {
                "entry_function": kernel_context["entry_function"],
                "arch_dispatch": kernel_context["arch_dispatch"],
                "task_type": kernel_context["task_type"],
                "lane_role": lane_role,
                "paired_core_id": paired_core_id,
                "logical_core_id": item["logical_core_id"],
                "shared_logical_assignment": kernel_context["shared_logical_assignment"],
                "execution_note": (
                    "Shares the same logical split window as its pair core; lane role reflects the mixed vector/cube kernel contract."
                ),
            }
            attached.append(item)
        return attached

    def _precision_mode_hint(self, case: PromptFlashAttentionCase) -> str | None:
        raw_value = case.attributes.get("innerPrecise")
        if raw_value is None:
            return None
        try:
            precise = int(raw_value)
        except (TypeError, ValueError):
            return None
        return "HIGHPRECISION" if precise & 1 else "HIGHPERFORMANCE"

    def _tiling_key_config_hint(
        self,
        case: PromptFlashAttentionCase,
        s_outer_factor: int,
        s_inner_factor: int,
    ) -> str | None:
        s_outer = s_outer_factor * CV_RATIO
        mapping = {
            (64, 64, 256, 256): "Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256",
            (64, 64, 512, 512): "Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512",
            (64, 256, 64, 64): "Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64",
            (64, 256, 128, 128): "Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128",
            (128, 128, 64, 64): "Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64",
            (128, 128, 128, 128): "Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128",
            (128, 128, 192, 128): "Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128",
            (128, 128, 256, 128): "Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128",
            (128, 128, 256, 256): "Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256",
            (128, 128, 512, 512): "Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512",
            (128, 256, 64, 64): "Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64",
            (64, 128, 576, 512): "Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512",
            (64, 256, 256, 256): "Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256",
            (128, 256, 128, 128): "Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128",
            (128, 128, 128, 64): "Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned64",
            (128, 128, 64, 128): "Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned128",
            (64, 256, 128, 64): "Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned64",
            (64, 256, 64, 128): "Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned128",
        }
        return mapping.get((s_outer, s_inner_factor, case.query_d, case.value_d))

    def _source_hash(self) -> str:
        payload = (
            self.source_path.read_bytes()
            + self.header_path.read_bytes()
            + self.kernel_entry_path.read_bytes()
            + self.kernel_dispatch_path.read_bytes()
        )
        return hashlib.sha256(payload).hexdigest()[:12]

    def _find_line_span(self, path: Path, marker: str, label: str) -> dict[str, Any]:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for index, line in enumerate(lines, start=1):
            if marker in line:
                return {
                    "path": str(path.resolve()),
                    "start_line": index,
                    "end_line": index,
                    "label": label,
                }
        raise ValueError(f"marker '{marker}' not found in {path}")

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit - 3]}..."

    def _color_for_index(self, index: int) -> str:
        base = (index * 53) % 255
        return f"rgb({(base + 70) % 255},{(base + 130) % 255},{(base + 190) % 255})"

    def _manifest_for_root(
        self,
        root: Path,
        ignored_relative_paths: set[str] | None = None,
    ) -> dict[str, str]:
        manifest: dict[str, str] = {}
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative_path = path.relative_to(root).as_posix()
            if ignored_relative_paths and relative_path in ignored_relative_paths:
                continue
            manifest[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return manifest

    def _manifest_sha256(self, manifest: dict[str, str]) -> str:
        payload = "\n".join(f"{path}:{digest}" for path, digest in sorted(manifest.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _workspace_sync_status(self) -> dict[str, Any]:
        fixture_root = self.repo_root / "fixtures" / "prompt_flash_attention"
        workspace_root = self.repo_root.parent / "prompt_flash_attention"
        ignored_extra_files = {"FIXTURE_SOURCE.md"}
        if not fixture_root.exists():
            return {
                "available": False,
                "reason": "fixture_root_missing",
                "reference_root": str(workspace_root),
            }
        if not workspace_root.exists():
            return {
                "available": False,
                "reason": "workspace_source_missing",
                "reference_root": str(workspace_root),
            }

        fixture_manifest = self._manifest_for_root(
            fixture_root,
            ignored_relative_paths=ignored_extra_files,
        )
        workspace_manifest = self._manifest_for_root(workspace_root)
        missing_in_fixture = sorted(set(workspace_manifest) - set(fixture_manifest))
        extra_in_fixture = sorted(set(fixture_manifest) - set(workspace_manifest))
        mismatch_files = sorted(
            relative_path
            for relative_path in sorted(set(fixture_manifest) & set(workspace_manifest))
            if fixture_manifest[relative_path] != workspace_manifest[relative_path]
        )
        return {
            "available": True,
            "reference_root": str(workspace_root),
            "ignored_extra_files": sorted(ignored_extra_files),
            "fixture_file_count": len(fixture_manifest),
            "reference_file_count": len(workspace_manifest),
            "missing_in_fixture": missing_in_fixture,
            "extra_in_fixture": extra_in_fixture,
            "mismatch_files": mismatch_files,
            "content_aligned": not missing_in_fixture and not extra_in_fixture and not mismatch_files,
        }

    def _function_span(self, function_name: str) -> dict[str, Any]:
        span = self.extractor.find_function_span(self.source_path, function_name)
        return {
            "path": span.path,
            "start_line": span.start_line,
            "end_line": span.end_line,
            "label": span.label,
        }

    def _struct_span(self, marker: str) -> dict[str, Any]:
        lines = self.header_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start_line = 0
        end_line = 0
        for index, line in enumerate(lines, start=1):
            if marker in line:
                start_line = index
                break
        for index in range(start_line, len(lines) + 1):
            if start_line and "END_TILING_DATA_DEF;" in lines[index - 1]:
                end_line = index
                break
        return {
            "path": str(self.header_path.resolve()),
            "start_line": start_line,
            "end_line": end_line or start_line,
            "label": marker,
        }
