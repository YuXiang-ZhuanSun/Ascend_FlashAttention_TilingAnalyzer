from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashattention_analyzers.fpa_v2 import PromptFlashAttentionV2Replayer
from flashattention_models import PlatformProfile


class PromptFlashAttentionV2ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.replayer = PromptFlashAttentionV2Replayer(
            ROOT,
            source_root=ROOT / "fixtures" / "prompt_flash_attention",
        )
        cls.cases = cls.replayer.load_cases(ROOT / "testcases" / "fa_testcases.csv")
        cls.case_lookup = {case.case_id: case for case in cls.cases}
        cls.platform = PlatformProfile(aiv_num=32, aic_num=32)

    def test_default_source_root_prefers_shipped_fixture_snapshot(self) -> None:
        replayer = PromptFlashAttentionV2Replayer(ROOT)
        self.assertEqual(
            replayer.source_root,
            (ROOT / "fixtures" / "prompt_flash_attention").resolve(),
        )

    def test_case_loader_broadcasts_actual_seq_lengths(self) -> None:
        case = self.case_lookup["aclnnPromptFlashAttentionV3_test122_m5_B8"]
        self.assertEqual(case.batch_size, 8)
        self.assertEqual(case.actual_seq_lengths, [128] * 8)
        self.assertEqual(case.actual_seq_lengths_kv, [1024] * 8)
        self.assertEqual(case.key_dtype, "float16")
        self.assertEqual(case.output_dtype, "float16")

    def test_replay_standard_bnsd_case(self) -> None:
        case = self.case_lookup["aclnnPromptFlashAttentionV3_test122_m"]
        result = self.replayer.replay_case(case, self.platform)
        self.assertEqual(result["used_physical_cores"], 32)
        self.assertEqual(result["logical_core_groups"], 16)
        self.assertTrue(result["validation"]["coverage_ok"])
        self.assertTrue(result["validation"]["weighted_coverage_ok"])

        first_group = result["logical_core_assignments"][0]
        self.assertEqual(first_group["coreNidStart"], 0)
        self.assertEqual(first_group["coreNidEnd"], 2)
        self.assertEqual(first_group["coreSidStart"], 0)
        self.assertEqual(first_group["coreSidEnd"], 1)
        self.assertEqual(first_group["coreSposStart"], 0)
        self.assertEqual(first_group["coreSposEnd"], 8)
        self.assertEqual(first_group["range_start_unit_index"], 0)
        self.assertEqual(first_group["range_end_unit_index"], 16)
        self.assertEqual(len(first_group["task_segments"]), 2)
        self.assertEqual(first_group["task_segments"][0]["sid"], 0)
        self.assertEqual(first_group["task_segments"][0]["nid"], 0)
        self.assertEqual(first_group["task_segments"][0]["spos"], [0, 8])
        self.assertEqual(first_group["task_segments"][0]["query_tokens"], [0, 1024])
        self.assertEqual(first_group["task_segments"][0]["active_kv_tokens"], [[0, 1024]])
        self.assertEqual(first_group["task_units"][0]["grid_shape"], {"q_blocks": 8, "kv_blocks": 8})
        self.assertIn("B0 H0", first_group["task_summary"])

        first_core = result["core_assignments"][0]
        self.assertEqual(first_core["core_id"], 0)
        self.assertEqual(first_core["lane_role"], "vector")
        self.assertEqual(first_core["task_units"][0]["query_tokens"], [0, 128])
        self.assertEqual(first_core["task_units"][0]["active_kv_tokens"], [[0, 1024]])
        self.assertEqual(first_core["task_units"][0]["grid_shape"], {"q_blocks": 8, "kv_blocks": 8})
        self.assertEqual(first_core["kernel_execution"]["paired_core_id"], 1)
        self.assertEqual(first_core["kernel_execution"]["arch_dispatch"], "prompt_flash_attention_FIAS_arch32")
        self.assertEqual(result["kernel_execution_model"]["pair_size"], 2)
        self.assertGreater(len(result["kernel_execution_model"]["candidate_dispatches"]), 0)
        self.assertEqual(
            result["kernel_execution_model"]["candidate_dispatches"][0]["implementation"],
            "PromptFlashAttentionS1s2Bns1X910",
        )

    def test_replay_multi_batch_case(self) -> None:
        case = self.case_lookup["aclnnPromptFlashAttentionV3_test122_m5_B2"]
        result = self.replayer.replay_case(case, self.platform)
        self.assertEqual(result["used_physical_cores"], 32)
        self.assertEqual(result["logical_core_groups"], 16)
        self.assertEqual(result["validation"]["total_unit_count"], 64)
        self.assertTrue(result["validation"]["coverage_ok"])
        self.assertEqual(
            result["kernel_execution_model"]["tiling_key_components"]["config"],
            "Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128",
        )

    def test_replay_all_csv_rows_have_full_validation(self) -> None:
        payload = self.replayer.replay_csv(ROOT / "testcases" / "fa_testcases.csv", self.platform)
        self.assertEqual(payload["case_count"], 23)
        self.assertEqual(len(payload["cases"]), 23)
        self.assertTrue(all(case["validation"]["coverage_ok"] for case in payload["cases"]))
        self.assertTrue(all(case["validation"]["weighted_coverage_ok"] for case in payload["cases"]))
        self.assertTrue(
            all(case["kernel_execution_model"]["candidate_dispatches"] for case in payload["cases"])
        )
        self.assertEqual({case["split_core_mode"] for case in payload["cases"]}, {"SPLIT_NBS_CUBE"})


if __name__ == "__main__":
    unittest.main()
