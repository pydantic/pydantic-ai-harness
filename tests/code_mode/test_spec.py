"""`CodeMode` spec support: the published `AgentSpec` schema entry and `from_spec`.

The `os_access: CodeModeOS | None` field has no JSON representation, and one such
field used to erase the capability's whole schema entry. `from_spec` names the
spec-expressible parameters, turns `CodeModeMountSpec` mappings into real `MountDir`
instances, and rejects `os_access` by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_monty import MountDir

from pydantic_ai_harness.code_mode import CodeMode


class TestSchema:
    def test_schema_publishes_the_capability_entry(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([CodeMode])
        params: dict[str, Any] = schema['$defs']['spec_params_CodeMode']['properties']
        assert {
            'tools',
            'max_retries',
            'max_tool_calls',
            'mount',
            'resource_limits',
            'dynamic_catalog',
            'id',
            'description',
            'defer_loading',
        } <= set(params)
        assert 'os_access' not in params

    def test_mounts_are_published_as_their_spec_shape(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([CodeMode])
        mount_spec: dict[str, Any] = schema['$defs']['CodeModeMountSpec']
        assert set(mount_spec['properties']) == {
            'host_path',
            'virtual_path',
            'mode',
            'write_bytes_limit',
            'memory_usage_limit',
        }
        assert set(mount_spec['required']) == {'host_path', 'virtual_path'}


class TestFromSpec:
    def test_options_are_forwarded(self) -> None:
        capability = CodeMode.from_spec(
            tools=['search', 'fetch'],
            max_retries=1,
            max_tool_calls=7,
            resource_limits={'max_duration_secs': 5.0},
            dynamic_catalog=True,
            id='cm',
            description='sandboxed tools',
            defer_loading=True,
        )
        assert capability.tools == ['search', 'fetch']
        assert capability.max_retries == 1
        assert capability.max_tool_calls == 7
        assert capability.resource_limits == {'max_duration_secs': 5.0}
        assert capability.dynamic_catalog is True
        assert capability.id == 'cm'
        assert capability.description == 'sandboxed tools'
        assert capability.defer_loading is True

    def test_a_single_mount_mapping_becomes_a_mount_dir(self, tmp_path: Path) -> None:
        capability = CodeMode.from_spec(
            mount={'host_path': str(tmp_path), 'virtual_path': '/work', 'mode': 'read-only'},
        )
        assert isinstance(capability.mount, MountDir)
        assert capability.mount.virtual_path == '/work'
        assert capability.mount.mode == 'read-only'

    def test_a_mount_list_becomes_mount_dirs(self, tmp_path: Path) -> None:
        capability = CodeMode.from_spec(
            mount=[
                {'host_path': str(tmp_path), 'virtual_path': '/a'},
                {'host_path': str(tmp_path), 'virtual_path': '/b', 'write_bytes_limit': 1024},
            ],
        )
        assert isinstance(capability.mount, list)
        assert [mount.virtual_path for mount in capability.mount] == ['/a', '/b']
        assert capability.mount[1].write_bytes_limit == 1024

    def test_a_mount_missing_a_required_key_fails_validation(self, tmp_path: Path) -> None:
        bad: dict[str, Any] = {'mount': {'host_path': str(tmp_path)}}
        with pytest.raises(ValidationError, match='Field required'):
            CodeMode.from_spec(**bad)

    def test_an_unknown_mount_key_is_rejected(self, tmp_path: Path) -> None:
        bad: dict[str, Any] = {
            'mount': {
                'host_path': str(tmp_path),
                'virtual_path': '/work',
                'write_bytes_limt': 1024,
            },
        }
        with pytest.raises(ValueError, match=r"Unknown mount spec key\(s\): \['write_bytes_limt'\]"):
            CodeMode.from_spec(**bad)

    @pytest.mark.parametrize('mount', ['not-a-mapping', ['not-a-mapping']])
    def test_a_non_mapping_mount_entry_fails_validation(self, mount: Any) -> None:
        with pytest.raises(ValidationError):
            CodeMode.from_spec(**{'mount': mount})

    def test_live_os_access_is_rejected_by_name(self) -> None:
        bad: dict[str, Any] = {'os_access': object()}
        with pytest.raises(UserError, match='cannot be built from a spec with `os_access`'):
            CodeMode.from_spec(**bad)

    def test_unknown_fields_are_rejected(self) -> None:
        bad: dict[str, Any] = {'sandbox': 'monty'}
        with pytest.raises(UserError, match=r"no spec field\(s\) \['sandbox'\]"):
            CodeMode.from_spec(**bad)
