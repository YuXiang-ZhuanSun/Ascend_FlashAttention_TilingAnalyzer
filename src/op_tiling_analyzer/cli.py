from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from op_tiling_analyzer.analyzers.fpa_v2 import PromptFlashAttentionV2Replayer
from op_tiling_analyzer.models import PlatformProfile
from op_tiling_analyzer.utils import ensure_parent, write_json


SUPPORTED_OPERATORS = {"prompt_flash_attention_v2"}


def build_parser() -> argparse.ArgumentParser:
    default_source_root = Path("fixtures") / "prompt_flash_attention"
    default_cases = Path("testcases") / "fa_testcases.csv"
    parser = argparse.ArgumentParser(
        prog="tiling_tool.py",
        description="Build and validate an operator tiling replay tool from source code and testcase inputs.",
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser("analyze-source", help="Parse source and emit a structured analysis report.")
    analyze_parser.add_argument("--operator", default="prompt_flash_attention_v2", choices=sorted(SUPPORTED_OPERATORS))
    analyze_parser.add_argument("--source-root", type=Path, default=default_source_root)
    analyze_parser.add_argument("--output", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay-cases", help="Replay tiling logic for a CSV case list.")
    replay_parser.add_argument("--operator", default="prompt_flash_attention_v2", choices=sorted(SUPPORTED_OPERATORS))
    replay_parser.add_argument("--source-root", type=Path, default=default_source_root)
    replay_parser.add_argument("--cases", type=Path, default=default_cases)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument("--summary-csv", type=Path)
    replay_parser.add_argument("--visualize-dir", type=Path)
    replay_parser.add_argument("--aiv-num", type=int, default=32)
    replay_parser.add_argument("--aic-num", type=int, default=32)
    replay_parser.add_argument("--disable-fa-run-flag", action="store_true")

    visualize_parser = subparsers.add_parser("visualize", help="Render SVG block diagrams from replay JSON.")
    visualize_parser.add_argument("--input", type=Path, required=True)
    visualize_parser.add_argument("--output-dir", type=Path, required=True)
    visualize_parser.add_argument("--case-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in {"analyze-source", "replay-cases", "visualize", "-h", "--help"}:
        args_list = ["replay-cases", *args_list]
    parser = build_parser()
    if not args_list:
        parser.print_help()
        return 0
    args = parser.parse_args(args_list)

    if args.command == "analyze-source":
        replayer = PromptFlashAttentionV2Replayer(Path.cwd(), source_root=args.source_root)
        write_json(args.output, replayer.analyze_source())
        return 0

    if args.command == "replay-cases":
        replayer = PromptFlashAttentionV2Replayer(Path.cwd(), source_root=args.source_root)
        platform = PlatformProfile(
            aiv_num=args.aiv_num,
            aic_num=args.aic_num,
            fa_run_flag=not args.disable_fa_run_flag,
        )
        payload = replayer.replay_csv(args.cases, platform)
        write_json(args.output, payload)
        if args.summary_csv is not None:
            _write_summary_csv(args.summary_csv, replayer.summary_rows(payload))
        if args.visualize_dir is not None:
            _render_visualizations(replayer, payload["cases"], args.visualize_dir)
        return 0

    if args.command == "visualize":
        replayer = PromptFlashAttentionV2Replayer(Path.cwd(), source_root=Path.cwd())
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        cases = payload["cases"]
        if args.case_id:
            cases = [case for case in cases if case["case_id"] == args.case_id]
        _render_visualizations(replayer, cases, args.output_dir)
        return 0

    parser.print_help()
    return 0


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_visualizations(
    replayer: PromptFlashAttentionV2Replayer,
    cases: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name_counts: dict[str, int] = {}
    for case in cases:
        case_id = str(case["case_id"])
        file_name_counts[case_id] = file_name_counts.get(case_id, 0) + 1
        suffix = "" if file_name_counts[case_id] == 1 else f'__{file_name_counts[case_id]}'
        replayer.render_case_svg(case, output_dir / f"{case_id}{suffix}.svg")
