from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from op_tiling_analyzer.models import SourceSpan, StructDefinition, StructField


BEGIN_RE = re.compile(r"BEGIN_TILING_DATA_DEF\((?P<name>\w+)\)")
END_RE = re.compile(r"END_TILING_DATA_DEF;")
FIELD_RE = re.compile(
    r"TILING_DATA_FIELD_DEF\((?P<cpp_type>[^,]+),\s*(?P<name>\w+)\);(?:\s*//\s*(?P<comment>.*))?"
)
ARRAY_FIELD_RE = re.compile(
    r"TILING_DATA_FIELD_DEF_ARR\((?P<cpp_type>[^,]+),\s*(?P<length>\d+),\s*(?P<name>\w+)\);(?:\s*//\s*(?P<comment>.*))?"
)
SETTER_RE = re.compile(
    r"(?P<receiver>\w+)->set_(?P<field>\w+)\((?P<value>\w+)\.data\(\)\);"
)
CONST_RE = re.compile(
    r"constexpr\s+(?P<cpp_type>[\w:<>]+)\s+(?P<name>\w+)\s*=\s*(?P<value>[^;]+);"
)
FUNCTION_RE = re.compile(r"^\s*[\w:<>*&\s]+\b(?P<name>[\w:]+)::(?P<method>\w+)\s*\(")


class CppTilingExtractor:
    def extract_tiling_structs(self, header_path: Path) -> list[StructDefinition]:
        structs: list[StructDefinition] = []
        lines = header_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        current_name: str | None = None
        current_fields: list[StructField] = []
        start_line = 0
        for index, line in enumerate(lines, start=1):
            if current_name is None:
                match = BEGIN_RE.search(line)
                if match:
                    current_name = match.group("name")
                    current_fields = []
                    start_line = index
                continue

            match = ARRAY_FIELD_RE.search(line)
            if match:
                current_fields.append(
                    StructField(
                        name=match.group("name"),
                        cpp_type=match.group("cpp_type").strip(),
                        array_length=int(match.group("length")),
                        comment=(match.group("comment") or "").strip() or None,
                        source=SourceSpan(
                            path=str(header_path.resolve()),
                            start_line=index,
                            end_line=index,
                        ),
                    )
                )
                continue

            match = FIELD_RE.search(line)
            if match:
                current_fields.append(
                    StructField(
                        name=match.group("name"),
                        cpp_type=match.group("cpp_type").strip(),
                        array_length=None,
                        comment=(match.group("comment") or "").strip() or None,
                        source=SourceSpan(
                            path=str(header_path.resolve()),
                            start_line=index,
                            end_line=index,
                        ),
                    )
                )
                continue

            if END_RE.search(line):
                structs.append(
                    StructDefinition(
                        name=current_name,
                        fields=current_fields,
                        source=SourceSpan(
                            path=str(header_path.resolve()),
                            start_line=start_line,
                            end_line=index,
                        ),
                    )
                )
                current_name = None
                current_fields = []
                start_line = 0

        return structs

    def extract_constants(
        self, paths: Iterable[Path], constant_names: set[str] | None = None
    ) -> dict[str, dict[str, str | int]]:
        constants: dict[str, dict[str, str | int]] = {}
        for path in paths:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            for index, line in enumerate(lines, start=1):
                match = CONST_RE.search(line)
                if not match:
                    continue
                name = match.group("name")
                if constant_names is not None and name not in constant_names:
                    continue
                constants[name] = {
                    "value": match.group("value").strip(),
                    "cpp_type": match.group("cpp_type").strip(),
                    "path": str(path.resolve()),
                    "line": index,
                }
        return constants

    def extract_assignment_mapping(
        self,
        source_path: Path,
        receiver: str = "seqParams",
        alias_by_field: dict[str, str] | None = None,
    ) -> list[dict[str, str | int]]:
        mappings: list[dict[str, str | int]] = []
        lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for index, line in enumerate(lines, start=1):
            match = SETTER_RE.search(line)
            if not match or match.group("receiver") != receiver:
                continue
            field_name = match.group("field")
            mappings.append(
                {
                    "struct_field": field_name,
                    "semantic_field": (alias_by_field or {}).get(field_name, field_name),
                    "source_variable": match.group("value"),
                    "path": str(source_path.resolve()),
                    "line": index,
                }
            )
        return mappings

    def find_function_span(self, source_path: Path, function_name: str) -> SourceSpan:
        lines = source_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        start_line = 0
        for index, line in enumerate(lines, start=1):
            if function_name in line and "::" in line and "(" in line:
                start_line = index
                break
        if start_line == 0:
            raise ValueError(f"function '{function_name}' not found in {source_path}")

        end_line = len(lines)
        for index in range(start_line, len(lines)):
            if index + 1 <= start_line:
                continue
            match = FUNCTION_RE.search(lines[index])
            if match:
                end_line = index
                break

        return SourceSpan(
            path=str(source_path.resolve()),
            start_line=start_line,
            end_line=end_line,
            label=function_name,
        )
