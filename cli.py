from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashattention_cli import main as advanced_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay FlashAttention tiling from testcase CSV inputs.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("testcases") / "fa_testcases.csv",
        help="CSV testcase input path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for replay.json, summary.csv, and visualizations/ outputs.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("fixtures") / "prompt_flash_attention",
        help="Operator source tree or fixture snapshot root.",
    )
    parser.add_argument("--output", type=Path, help="Optional custom replay JSON output path.")
    parser.add_argument("--summary-csv", type=Path, help="Optional custom summary CSV output path.")
    parser.add_argument("--visualize-dir", type=Path, help="Optional custom SVG output directory.")
    parser.add_argument("--aiv-num", type=int, default=32)
    parser.add_argument("--aic-num", type=int, default=32)
    parser.add_argument("--disable-fa-run-flag", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    forwarded = [
        "replay-cases",
        "--input",
        str(args.input),
        "--output-dir",
        str(args.output_dir),
        "--source-root",
        str(args.source_root),
        "--aiv-num",
        str(args.aiv_num),
        "--aic-num",
        str(args.aic_num),
    ]
    if args.output is not None:
        forwarded.extend(["--output", str(args.output)])
    if args.summary_csv is not None:
        forwarded.extend(["--summary-csv", str(args.summary_csv)])
    if args.visualize_dir is not None:
        forwarded.extend(["--visualize-dir", str(args.visualize_dir)])
    if args.disable_fa_run_flag:
        forwarded.append("--disable-fa-run-flag")
    return advanced_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
