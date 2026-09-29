"""`ModalSandbox` spec support: the published `AgentSpec` schema entry and `from_spec`.

The `session: ModalSandboxSession | None` field has no JSON representation, and one
such field used to erase the capability's whole schema entry. `from_spec` names the
spec-expressible parameters and rejects `session` by name.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import AgentSpec
from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.modal_sandbox import ModalSandbox


class TestSchema:
    def test_schema_publishes_the_capability_entry(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([ModalSandbox])
        params: dict[str, Any] = schema['$defs']['spec_params_ModalSandbox']['properties']
        assert {
            'image',
            'sandbox_id',
            'app_name',
            'create_app_if_missing',
            'sandbox_timeout',
            'workdir',
            'env',
            'default_command_timeout',
            'max_command_timeout',
            'max_output_bytes',
            'max_output_lines',
            'max_read_bytes',
            'instructions',
            'id',
            'description',
            'defer_loading',
        } <= set(params)
        assert 'session' not in params


class TestFromSpec:
    def test_owned_sandbox_settings_are_forwarded(self) -> None:
        capability = ModalSandbox.from_spec(
            image='ubuntu:24.04',
            app_name='spec-app',
            create_app_if_missing=False,
            sandbox_timeout=600,
            workdir='/work',
            env={'TOKEN': 'x'},
            default_command_timeout=30.0,
            max_command_timeout=120,
            max_output_bytes=1024,
            max_output_lines=50,
            max_read_bytes=2048,
            instructions='use the sandbox',
            id='ms',
            description='cloud sandbox',
            defer_loading=True,
        )
        assert capability.image == 'ubuntu:24.04'
        assert capability.app_name == 'spec-app'
        assert capability.create_app_if_missing is False
        assert capability.sandbox_timeout == 600
        assert capability.workdir == '/work'
        assert capability.env == {'TOKEN': 'x'}
        assert capability.default_command_timeout == 30.0
        assert capability.max_command_timeout == 120
        assert capability.max_output_bytes == 1024
        assert capability.max_output_lines == 50
        assert capability.max_read_bytes == 2048
        assert capability.instructions == 'use the sandbox'
        assert capability.id == 'ms'
        assert capability.description == 'cloud sandbox'
        assert capability.defer_loading is True

    def test_attach_mode_round_trips(self) -> None:
        assert ModalSandbox.from_spec(sandbox_id='sb-1').sandbox_id == 'sb-1'

    def test_construction_validation_still_applies(self) -> None:
        with pytest.raises(ValueError, match='only apply when creating a sandbox'):
            ModalSandbox.from_spec(sandbox_id='sb-1', image='ubuntu:24.04')

    def test_a_live_session_is_rejected_by_name(self) -> None:
        bad: dict[str, Any] = {'session': object()}
        with pytest.raises(UserError, match='cannot be built from a spec with `session`'):
            ModalSandbox.from_spec(**bad)

    def test_unknown_fields_are_rejected(self) -> None:
        bad: dict[str, Any] = {'region': 'us-east'}
        with pytest.raises(UserError, match=r"no spec field\(s\) \['region'\]"):
            ModalSandbox.from_spec(**bad)
