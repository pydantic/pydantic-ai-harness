"""`FileSystem` as a `FileReader`: whether its `read_file` can be pointed at files others keep in the workspace."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.filesystem import FileSystem

SPILLS = '.pydantic-ai-harness/tool-output'
SPILL = f'{SPILLS}/run/call.0'
CAPPED = {'max_read_chars': 1_000}


class TestFileSystemAsFileReader:
    @pytest.mark.parametrize(
        ('settings', 'path', 'tool'),
        [
            pytest.param(CAPPED, SPILL, 'read_file', id='capped-reads-in-the-working-directory'),
            pytest.param({'max_read_chars': 50_000}, SPILL, 'read_file', id='cap-equal-to-the-limit'),
            pytest.param({}, SPILL, None, id='uncapped-reads'),
            pytest.param({'max_read_chars': 50_001}, SPILL, None, id='cap-above-the-limit'),
            pytest.param({**CAPPED, 'tools': ['write_file']}, SPILL, None, id='no-read-file-tool'),
            pytest.param({**CAPPED, 'root_dir': '..'}, SPILL, None, id='explicit-root-dir'),
            pytest.param({**CAPPED, 'denied_patterns': [f'{SPILLS}/**']}, SPILL, None, id='denied'),
            pytest.param({**CAPPED, 'allowed_patterns': ['src/**']}, SPILL, None, id='not-allowed'),
            pytest.param({**CAPPED, 'allowed_patterns': [f'{SPILLS}/**']}, SPILL, 'read_file', id='allowed'),
            pytest.param(CAPPED, '../elsewhere', None, id='outside-the-working-directory'),
            pytest.param(CAPPED, '/var/spills', None, id='absolute-path'),
        ],
    )
    def test_answers_from_configuration(self, settings: dict[str, Any], path: str, tool: str | None) -> None:
        filesystem = FileSystem[None](**settings)
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

        assert filesystem.file_read_tool(ctx, path, max_chars=50_000) == tool
