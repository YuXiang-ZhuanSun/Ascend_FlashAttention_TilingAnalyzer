from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CliIntegrationTests(unittest.TestCase):
    def test_simple_cli_generates_default_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_csv = tmp_path / "quickstart_cases.csv"
            source_lines = (ROOT / "testcases" / "fa_testcases.csv").read_text(encoding="utf-8").splitlines()
            input_csv.write_text("\n".join(source_lines[:2]) + "\n", encoding="utf-8")
            output_dir = tmp_path / "quickstart"
            command = [
                sys.executable,
                str(ROOT / "cli.py"),
                "--input",
                str(input_csv),
                "--output-dir",
                str(output_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0)
            replay_json = output_dir / "replay.json"
            summary_csv = output_dir / "summary.csv"
            visualize_dir = output_dir / "visualizations"
            self.assertTrue(replay_json.exists())
            self.assertTrue(summary_csv.exists())
            self.assertTrue(visualize_dir.exists())
            payload = json.loads(replay_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["case_count"], 1)
            self.assertEqual(len(payload["cases"]), 1)
            self.assertEqual(len(list(visualize_dir.glob("*.svg"))), 1)

    def test_replay_cli_generates_json_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_json = Path(tmp_dir) / "result.json"
            output_svg_dir = Path(tmp_dir) / "svg"
            command = [
                sys.executable,
                str(ROOT / "tiling_tool.py"),
                "--cases",
                str(ROOT / "testcases" / "fa_testcases.csv"),
                "--output",
                str(output_json),
                "--visualize-dir",
                str(output_svg_dir),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertGreater(payload["case_count"], 0)
            self.assertTrue(payload["source_analysis"]["fixture_snapshot"]["full_operator_snapshot"])
            self.assertTrue(payload["source_analysis"]["fixture_snapshot"]["workspace_sync"]["content_aligned"])
            first_case = payload["cases"][0]
            self.assertIn("task_summary", first_case["core_assignments"][0])
            self.assertIn("kernel_execution", first_case["core_assignments"][0])
            self.assertGreater(len(first_case["core_assignments"][0]["task_units"]), 0)
            self.assertIn("grid_shape", first_case["core_assignments"][0]["task_units"][0])
            self.assertTrue((output_svg_dir / "aclnnPromptFlashAttentionV3_test122_m.svg").exists())
            svg_files = sorted(output_svg_dir.glob("*.svg"))
            self.assertEqual(len(svg_files), payload["case_count"])
            self.assertTrue((output_svg_dir / "PFAV3_case7.svg").exists())
            self.assertTrue((output_svg_dir / "PFAV3_case7__2.svg").exists())
            svg_content = (output_svg_dir / "aclnnPromptFlashAttentionV3_test122_m.svg").read_text(encoding="utf-8")
            self.assertIn("C00 vector", svg_content)
            self.assertIn("B0 H0", svg_content)
            self.assertIn("qblk x kvblk", svg_content)
            self.assertIn("host_branch=default_value_d_le_128", svg_content)
            self.assertIn("dn_branch=no_dn_sinner_override", svg_content)
            self.assertIn("tiling_key=QFP16_KVFP16_OUTFP16_BNSD_HIGHPRECISION_HIGHLEVELAPI_MDL_CUBEVECTORDIFF_NEWTILING", svg_content)
            self.assertIn("value=1000000000000001612", svg_content)


if __name__ == "__main__":
    unittest.main()
