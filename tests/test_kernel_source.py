from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashattention_analyzers.fpa_v2 import PromptFlashAttentionV2Replayer
from flashattention_analyzers.kernel_source import KernelSourceExtractor


class KernelSourceExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.replayer = PromptFlashAttentionV2Replayer(
            ROOT,
            source_root=ROOT / "fixtures" / "prompt_flash_attention",
        )
        cls.case = cls.replayer.load_cases(ROOT / "testcases" / "fa_testcases.csv")[0]
        cls.extractor = KernelSourceExtractor()
        cls.dispatch_path = ROOT / "fixtures" / "prompt_flash_attention" / "op_kernel" / "prompt_flash_attention_arch32.h"

    def test_extract_entrypoints(self) -> None:
        entrypoints = self.extractor.extract_entrypoints(
            ROOT / "fixtures" / "prompt_flash_attention" / "op_kernel" / "prompt_flash_attention.cpp"
        )
        self.assertEqual(entrypoints[0]["name"], "prompt_flash_attention_FIAS")
        self.assertIn("prompt_flash_attention_FIAS_arch32", entrypoints[0]["helper_targets"])

    def test_match_default_bnsd_kernel_candidates(self) -> None:
        candidates = self.extractor.match_dispatch_candidates(
            self.dispatch_path,
            case_layout=self.case.layout,
            split_mode="CUBEVECTORDIFF",
            input_dtype=self.case.input_dtype,
            key_dtype=self.case.key_dtype,
            output_dtype=self.case.output_dtype,
            precision_mode="HIGHPRECISION",
            high_level_api=True,
            has_attention_mask=self.case.has_attention_mask,
        )
        self.assertGreater(len(candidates), 0)
        self.assertTrue(all(candidate["layout"] == "BNSD" for candidate in candidates))
        self.assertTrue(all(candidate["split_mode"] == "CUBEVECTORDIFF" for candidate in candidates))
        self.assertEqual(candidates[0]["implementation"], "PromptFlashAttentionS1s2Bns1X910")


if __name__ == "__main__":
    unittest.main()
