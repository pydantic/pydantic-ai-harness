"""Tests for the `context_policy` channel: identity/context-driven access decisions."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, RunContext

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pydantic_ai_harness.permission_policy import PermissionPolicy, Rule, Verdict

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _one_shot_command_model(command: str) -> FunctionModel:
    """Call `run_command` once, then echo the tool return as text."""
    state = {'called': False}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not state['called']:
            state['called'] = True
            return ModelResponse(parts=[ToolCallPart(tool_name='run_command', args={'command': command})])
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        ret = next(p for p in last.parts if isinstance(p, ToolReturnPart) and p.tool_name == 'run_command')
        return ModelResponse(parts=[TextPart(str(ret.content))])

    return FunctionModel(model_fn)


def _agent(
    policy: PermissionPolicy[object],
    *,
    command: str = 'ls -la',
    output_type: Any = str,
) -> Agent[object, Any]:
    agent = Agent(_one_shot_command_model(command), capabilities=[policy], output_type=output_type)

    @agent.tool_plain
    def run_command(command: str) -> str:  # pyright: ignore[reportUnusedFunction]
        """Run a shell command."""
        return f'EXECUTED: {command}'

    return agent


class TestContextPolicy:
    async def test_context_deny_blocks_an_otherwise_allowed_call(self) -> None:
        # `ls` is on the read-only safelist (allowed); the context policy denies it anyway.
        policy = PermissionPolicy[object](context_policy=lambda ctx, name, args: 'deny')
        result = await _agent(policy).run('go')
        assert 'Permission denied for `run_command`' in result.output

    async def test_context_ask_defers_an_otherwise_allowed_call(self) -> None:
        policy = PermissionPolicy[object](context_policy=lambda ctx, name, args: 'ask')
        result = await _agent(policy, output_type=[str, DeferredToolRequests]).run('go')
        assert isinstance(result.output, DeferredToolRequests)
        assert [c.tool_name for c in result.output.approvals] == ['run_command']

    async def test_context_none_abstains_and_falls_back_to_rules(self) -> None:
        policy = PermissionPolicy[object](
            rules=[Rule('deny', tool='run_command', command='ls')],
            context_policy=lambda ctx, name, args: None,
        )
        result = await _agent(policy).run('go')
        assert 'Permission denied for `run_command`' in result.output

    async def test_context_deny_overrides_a_rule_allow(self) -> None:
        # Most-restrictive: a broad rule `allow` cannot green-light a context `deny`.
        policy = PermissionPolicy[object](
            rules=[Rule('allow', tool='run_command', command='ls')],
            context_policy=lambda ctx, name, args: 'deny',
        )
        result = await _agent(policy).run('go')
        assert 'Permission denied for `run_command`' in result.output

    async def test_context_allow_cannot_loosen_a_rule_deny(self) -> None:
        # Most-restrictive: a context `allow` cannot override a rule `deny`.
        policy = PermissionPolicy[object](
            rules=[Rule('deny', tool='run_command', command='ls')],
            context_policy=lambda ctx, name, args: 'allow',
        )
        result = await _agent(policy).run('go')
        assert 'Permission denied for `run_command`' in result.output

    async def test_async_context_policy_is_awaited(self) -> None:
        async def policy_fn(ctx: RunContext[object], name: str, args: dict[str, Any]) -> Verdict:
            return 'deny'

        policy = PermissionPolicy[object](context_policy=policy_fn)
        result = await _agent(policy).run('go')
        assert 'Permission denied for `run_command`' in result.output

    async def test_identity_driven_decision_from_deps(self) -> None:
        # The context policy reads the caller's role from `deps`: admin runs, viewer is denied.
        def by_role(ctx: RunContext[object], name: str, args: dict[str, Any]) -> Verdict | None:
            return 'deny' if ctx.deps == 'viewer' else None

        policy = PermissionPolicy[object](context_policy=by_role)

        admin = await _agent(policy).run('go', deps='admin')
        assert admin.output == 'EXECUTED: ls -la'

        viewer = await _agent(policy).run('go', deps='viewer')
        assert 'Permission denied for `run_command`' in viewer.output
