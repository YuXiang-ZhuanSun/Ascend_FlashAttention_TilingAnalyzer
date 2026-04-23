from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashattention_analyzers.fpa_v2 import PromptFlashAttentionV2Replayer


class FixtureSyncTests(unittest.TestCase):
    def test_fixture_snapshot_matches_workspace_source_when_available(self) -> None:
        replayer = PromptFlashAttentionV2Replayer(
            ROOT,
            source_root=ROOT / "fixtures" / "prompt_flash_attention",
        )
        sync_status = replayer.analyze_source()["fixture_snapshot"]["workspace_sync"]
        if not sync_status["available"]:
            self.skipTest(sync_status["reason"])
        self.assertTrue(sync_status["content_aligned"])
        self.assertEqual(sync_status["missing_in_fixture"], [])
        self.assertEqual(sync_status["extra_in_fixture"], [])
        self.assertEqual(sync_status["mismatch_files"], [])


if __name__ == "__main__":
    unittest.main()
