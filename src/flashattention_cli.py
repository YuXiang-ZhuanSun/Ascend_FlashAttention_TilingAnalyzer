from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from flashattention_analyzers.fpa_v2 import PromptFlashAttentionV2Replayer
from flashattention_models import PlatformProfile
from flashattention_utils import ensure_parent, write_json


SUPPORTED_OPERATORS = {"prompt_flash_attention_v2"}
COMMANDS = {"analyze-source", "replay-cases", "visualize"}


def _add_replay_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_source_root: Path,
    default_cases: Path,
) -> None:
    parser.add_argument("--operator", default="prompt_flash_attention_v2", choices=sorted(SUPPORTED_OPERATORS))
    parser.add_argument("--source-root", type=Path, default=default_source_root)
    parser.add_argument(
        "--input",
        "--cases",
        dest="cases",
        type=Path,
        default=default_cases,
        help="CSV testcase input path.",
    )
    parser.add_argument("--output", type=Path, help="Replay JSON output path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for replay.json, summary.csv, and visualizations/ outputs.",
    )
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--visualize-dir", type=Path)
    parser.add_argument("--aiv-num", type=int, default=32)
    parser.add_argument("--aic-num", type=int, default=32)
    parser.add_argument("--disable-fa-run-flag", action="store_true")


def _resolve_replay_outputs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path | None, Path | None]:
    if args.output_dir is not None:
        output = args.output or args.output_dir / "replay.json"
        summary_csv = args.summary_csv or args.output_dir / "summary.csv"
        visualize_dir = args.visualize_dir or args.output_dir / "visualizations"
        return output, summary_csv, visualize_dir
    if args.output is None:
        parser.error("replay-cases requires --output or --output-dir")
    return args.output, args.summary_csv, args.visualize_dir


def build_parser() -> argparse.ArgumentParser:
    default_source_root = Path("fixtures") / "prompt_flash_attention"
    default_cases = Path("testcases") / "fa_testcases.csv"
    parser = argparse.ArgumentParser(
        description="Analyze and replay FlashAttention tiling from source code and testcase inputs.",
        epilog=(
            "Quick start:\n"
            "  python cli.py --input testcases/fa_testcases.csv --output-dir results/quickstart\n"
            "Advanced source analysis:\n"
            "  python tiling_tool.py analyze-source --output docs/fpa_source_analysis.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser("analyze-source", help="Parse source and emit a structured analysis report.")
    analyze_parser.add_argument("--operator", default="prompt_flash_attention_v2", choices=sorted(SUPPORTED_OPERATORS))
    analyze_parser.add_argument("--source-root", type=Path, default=default_source_root)
    analyze_parser.add_argument("--output", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay-cases", help="Replay tiling logic for a CSV case list.")
    _add_replay_arguments(
        replay_parser,
        default_source_root=default_source_root,
        default_cases=default_cases,
    )

    visualize_parser = subparsers.add_parser("visualize", help="Render SVG block diagrams from replay JSON.")
    visualize_parser.add_argument("--input", type=Path, required=True)
    visualize_parser.add_argument("--output-dir", type=Path, required=True)
    visualize_parser.add_argument("--case-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in COMMANDS | {"-h", "--help"}:
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
        output_path, summary_csv_path, visualize_dir = _resolve_replay_outputs(parser, args)
        platform = PlatformProfile(
            aiv_num=args.aiv_num,
            aic_num=args.aic_num,
            fa_run_flag=not args.disable_fa_run_flag,
        )
        payload = replayer.replay_csv(args.cases, platform)
        write_json(output_path, payload)
        if summary_csv_path is not None:
            _write_summary_csv(summary_csv_path, replayer.summary_rows(payload))
        if visualize_dir is not None:
            _render_visualizations(replayer, payload["cases"], visualize_dir)
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
