from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashattention_analyzers.cpp_tiling import CppTilingExtractor


class CppTilingExtractorTests(unittest.TestCase):
    def test_extract_prompt_attention_seq_params(self) -> None:
        extractor = CppTilingExtractor()
        header_path = ROOT / "fixtures" / "prompt_flash_attention" / "op_host" / "prompt_flash_attention_tiling.h"
        structs = extractor.extract_tiling_structs(header_path)
        struct_lookup = {struct.name: struct for struct in structs}
        self.assertIn("PromptAttentionSeqParams", struct_lookup)

        seq_params = struct_lookup["PromptAttentionSeqParams"]
        field_lookup = {field.name: field for field in seq_params.fields}
        self.assertEqual(field_lookup["CoreHeadNumTail"].comment, "coreNStart")
        self.assertEqual(field_lookup["actualCoreNums"].comment, "coreSidStart")
        self.assertEqual(field_lookup["coreSeqPosEnd"].array_length, 64)

    def test_extract_assignment_mapping(self) -> None:
        extractor = CppTilingExtractor()
        header_path = ROOT / "fixtures" / "prompt_flash_attention" / "op_host" / "prompt_flash_attention_tiling.h"
        source_path = ROOT / "fixtures" / "prompt_flash_attention" / "op_host" / "prompt_flash_attention_tiling_v2.cpp"
        structs = extractor.extract_tiling_structs(header_path)
        seq_struct = {struct.name: struct for struct in structs}["PromptAttentionSeqParams"]
        alias_by_field = {field.name: field.comment or field.name for field in seq_struct.fields}
        mappings = extractor.extract_assignment_mapping(source_path, alias_by_field=alias_by_field)
        mapping_lookup = {entry["struct_field"]: entry for entry in mappings}
        self.assertEqual(mapping_lookup["CoreHeadNumTail"]["source_variable"], "coreNidStart")
        self.assertEqual(mapping_lookup["actualS1"]["source_variable"], "coreNidEnd")
        self.assertEqual(mapping_lookup["coreSeqPosStart"]["source_variable"], "coreSposStart")


if __name__ == "__main__":
    unittest.main()
