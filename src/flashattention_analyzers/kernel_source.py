from __future__ import annotations

import re
from pathlib import Path
from typing import Any


ENTRY_RE = re.compile(r'extern "C"\s+__global__\s+__aicore__\s+void\s+(?P<name>\w+)\s*\(')
TASK_TYPE_RE = re.compile(r"KERNEL_TASK_TYPE_DEFAULT\((?P<task_type>\w+)\)")
DISPATCH_BRANCH_RE = re.compile(
    r"#(?:el)?if\s+TILING_KEY_VAR\s*==\s*(?P<tiling_key>\w+)(?P<body>.*?)(?=(?:#elif\s+TILING_KEY_VAR\s*==|#endif))",
    re.DOTALL,
)
INVOKE_RE = re.compile(
    r"(?P<invoke_macro>INVOKE_[A-Z0-9_]+)\(\s*(?P<implementation>\w+)\s*(?:,\s*(?P<template_args>.*?))?\);",
    re.DOTALL,
)


class KernelSourceExtractor:
    def analyze(
        self,
        entry_path: Path,
        dispatch_path: Path,
        template_key_path: Path,
    ) -> dict[str, Any]:
        branches = self.extract_dispatch_branches(dispatch_path)
        split_counts: dict[str, int] = {}
        layout_counts: dict[str, int] = {}
        for branch in branches:
            split_mode = str(branch["split_mode"])
            layout = str(branch["layout"])
            split_counts[split_mode] = split_counts.get(split_mode, 0) + 1
            layout_counts[layout] = layout_counts.get(layout, 0) + 1
        return {
            "entrypoints": self.extract_entrypoints(entry_path),
            "dispatch_function": self.find_function_span(dispatch_path, "prompt_flash_attention_FIAS_arch32"),
            "task_type": self.extract_task_type(dispatch_path),
            "dispatch_branch_count": len(branches),
            "dispatch_branches": branches,
            "dispatch_split_counts": split_counts,
            "dispatch_layout_counts": layout_counts,
            "tiling_key_template": self.find_macro_block_span(template_key_path, "ASCENDC_TPL_ARGS_DECL"),
        }

    def extract_entrypoints(self, path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        matches = list(ENTRY_RE.finditer(text))
        entrypoints: list[dict[str, Any]] = []
        for index, match in enumerate(matches):
            body_start = match.start()
            body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[body_start:body_end]
            helper_targets: list[str] = []
            for callee in (
                "prompt_flash_attention_FIAS_regbase",
                "prompt_flash_attention_FIAS_arch32",
                "prompt_flash_attention_FIAS",
            ):
                if f"{callee}(" in body:
                    helper_targets.append(callee)
            entrypoints.append(
                {
                    "name": match.group("name"),
                    "source": self._span_from_offsets(path, text, match.start(), body_end, match.group("name")),
                    "helper_targets": helper_targets,
                }
            )
        return entrypoints

    def extract_task_type(self, dispatch_path: Path) -> str | None:
        text = dispatch_path.read_text(encoding="utf-8", errors="ignore")
        match = TASK_TYPE_RE.search(text)
        if match is None:
            return None
        return match.group("task_type")

    def extract_dispatch_branches(self, dispatch_path: Path) -> list[dict[str, Any]]:
        text = dispatch_path.read_text(encoding="utf-8", errors="ignore")
        branches: list[dict[str, Any]] = []
        for match in DISPATCH_BRANCH_RE.finditer(text):
            tiling_key = match.group("tiling_key")
            body = match.group("body")
            invoke_match = INVOKE_RE.search(body)
            if invoke_match is None:
                continue
            traits = self._key_traits(tiling_key)
            branches.append(
                {
                    "tiling_key": tiling_key,
                    "layout": traits["layout"],
                    "split_mode": traits["split_mode"],
                    "precision_mode": traits["precision_mode"],
                    "api_family": traits["api_family"],
                    "matmul_family": traits["matmul_family"],
                    "prefix_enabled": traits["prefix_enabled"],
                    "paged_attention_enabled": traits["paged_attention_enabled"],
                    "flash_decode_enabled": traits["flash_decode_enabled"],
                    "invoke_macro": invoke_match.group("invoke_macro"),
                    "implementation": invoke_match.group("implementation"),
                    "template_args": self._normalize_whitespace(invoke_match.group("template_args") or ""),
                    "source": self._span_from_offsets(
                        dispatch_path,
                        text,
                        match.start(),
                        match.end(),
                        tiling_key,
                    ),
                }
            )
        return branches

    def match_dispatch_candidates(
        self,
        dispatch_path: Path,
        case_layout: str,
        split_mode: str,
        input_dtype: str,
        key_dtype: str,
        output_dtype: str,
        precision_mode: str | None,
        high_level_api: bool,
        has_attention_mask: bool,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        expected_layout = case_layout.upper()
        query_token = self._dtype_signature_token(input_dtype, "Q")
        key_token = self._dtype_signature_token(key_dtype, "KV")
        output_token = self._dtype_signature_token(output_dtype, "OUT")
        candidates: list[dict[str, Any]] = []
        for branch in self.extract_dispatch_branches(dispatch_path):
            if branch["layout"] != expected_layout:
                continue
            if branch["split_mode"] != split_mode:
                continue
            key_name = str(branch["tiling_key"])
            if query_token and query_token not in key_name:
                continue
            if key_token and key_token not in key_name:
                continue
            if output_token and output_token not in key_name:
                continue
            if high_level_api and branch["api_family"] != "HIGHLEVELAPI":
                continue
            if not has_attention_mask and "ENABLE_MASK" in key_name:
                continue
            if branch["prefix_enabled"] or branch["paged_attention_enabled"] or branch["flash_decode_enabled"]:
                continue
            score = 0
            if precision_mode and branch["precision_mode"] == precision_mode:
                score += 4
            if "MDL" in key_name:
                score += 2
            if "NORMAL" not in key_name:
                score += 1
            candidate = dict(branch)
            candidate["score"] = score
            candidates.append(candidate)
        candidates.sort(
            key=lambda item: (
                -int(item["score"]),
                str(item["tiling_key"]),
            )
        )
        return candidates[:limit]

    def find_function_span(self, path: Path, function_name: str) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        marker = f"{function_name}("
        offset = text.find(marker)
        if offset < 0:
            raise ValueError(f"function '{function_name}' not found in {path}")
        body_end = text.find("\n}", offset)
        if body_end < 0:
            body_end = len(text)
        else:
            body_end += 3
        return self._span_from_offsets(path, text, offset, body_end, function_name)

    def find_macro_block_span(self, path: Path, marker: str) -> dict[str, Any]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        offset = text.find(marker)
        if offset < 0:
            raise ValueError(f"marker '{marker}' not found in {path}")
        end = text.find("ASCENDC_TPL_SEL(", offset)
        if end < 0:
            end = len(text)
        return self._span_from_offsets(path, text, offset, end, marker)

    def _dtype_signature_token(self, dtype_name: str, role: str) -> str | None:
        normalized = dtype_name.lower()
        mapping = {
            "float16": "FP16",
            "bfloat16": "BF16",
            "int8": "INT8",
            "float8_e4m3fn": "FP8",
            "hifloat8": "FP8",
        }
        suffix = mapping.get(normalized)
        if suffix is None:
            return None
        return f"{role}{suffix}"

    def _key_traits(self, tiling_key: str) -> dict[str, Any]:
        layout = "UNKNOWN"
        for candidate in ("BNSD", "BSH", "BSND", "TND", "NTD"):
            if f"_{candidate}_" in tiling_key:
                layout = candidate
                break
        split_mode = "CUBEVECTORDIFF" if "CUBEVECTORDIFF" in tiling_key else "NS_SPLIT"
        if "TAIL" in tiling_key or "NOTAIL" in tiling_key:
            split_mode = "NS_SPLIT"
        precision_mode = "HIGHPRECISION" if "HIGHPRECISION" in tiling_key else "HIGHPERFORMANCE"
        api_family = "BASICAPI" if "BASICAPI" in tiling_key else "HIGHLEVELAPI"
        matmul_family = "MLA" if "MLA" in tiling_key else ("MDL" if "MDL" in tiling_key else "GENERIC")
        return {
            "layout": layout,
            "split_mode": split_mode,
            "precision_mode": precision_mode,
            "api_family": api_family,
            "matmul_family": matmul_family,
            "prefix_enabled": "PREFIX" in tiling_key,
            "paged_attention_enabled": "_PA_" in tiling_key or "_PA_ND_" in tiling_key or "_PA_NZ_" in tiling_key,
            "flash_decode_enabled": "_FD_" in tiling_key,
        }

    def _span_from_offsets(
        self,
        path: Path,
        text: str,
        start_offset: int,
        end_offset: int,
        label: str,
    ) -> dict[str, Any]:
        start_line = text.count("\n", 0, start_offset) + 1
        end_line = text.count("\n", 0, end_offset) + 1
        return {
            "path": str(path.resolve()),
            "start_line": start_line,
            "end_line": end_line,
            "label": label,
        }

    def _normalize_whitespace(self, value: str) -> str:
        return " ".join(value.split())
