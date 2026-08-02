"""End-to-end tests for the `PermissionPolicy` capability through `Agent.run`."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ApprovalRequired, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from pydantic_ai_harness.permission_policy import (
        PermissionPolicy,
        PermissionRequest,
        Rule,
    )

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _one_shot_command_model(command: str) -> FunctionModel:
    """A model that calls `run_command` once, then echoes the tool return as text."""
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
    model: FunctionModel,
    policy: PermissionPolicy[object],
    *,
    output_type: Any = str,
) -> Agent[object, Any]:
    agent = Agent(model, capabilities=[policy], output_type=output_type)

    @agent.tool_plain
    def run_command(command: str) -> str:  # pyright: ignore[reportUnusedFunction]
        """Run a shell command."""
        return f'EXECUTED: {command}'

    return agent


class TestAllow:
    async def test_allowed_command_runs(self) -> None:
        policy = PermissionPolicy[object]()  # safelist auto-allows `ls`
        result = await _agent(_one_shot_command_model('ls -la'), policy).run('go')
        assert result.output == 'EXECUTED: ls -la'

    async def test_rule_allow_runs(self) -> None:
        policy = PermissionPolicy[object](rules=[Rule('allow', tool='run_command', command='npm')])
        result = await _agent(_one_shot_command_model('npm run build'), policy).run('go')
        assert result.output == 'EXECUTED: npm run build'


class TestDeny:
    async def test_user_deny_returns_retryable_message(self) -> None:
        policy = PermissionPolicy[object](rules=[Rule('deny', tool='run_command', command='ls')])
        result = await _agent(_one_shot_command_model('ls -la'), policy).run('go')
        assert 'Permission denied for `run_command`' in result.output
        assert 'you may restate the request' in result.output

    async def test_dangerous_command_is_never_allowed(self) -> None:
        policy = PermissionPolicy[object]()
        result = await _agent(_one_shot_command_model('rm -rf /'), policy).run('go')
        assert 'will not be allowed even with justification' in result.output

    async def test_deny_short_circuits_execution(self) -> None:
        # The wrapped tool must never run when denied: the output is the denial text,
        # not `EXECUTED: ...`.
        policy = PermissionPolicy[object](rules=[Rule('deny', tool='run_command')])
        result = await _agent(_one_shot_command_model('ls'), policy).run('go')
        assert 'EXECUTED' not in result.output

    async def test_deny_return_carries_provenance_marker(self) -> None:
        # A denial must be attributable to the harness policy layer, not read as a genuine
        # (successful) tool result. The prose marker is load-bearing for the model; the
        # `metadata` key is load-bearing for the app. (`outcome` can't be set from the
        # `wrap_tool_execute` return path in current core -- see `_DENY_METADATA_KEY`.)
        policy = PermissionPolicy[object](rules=[Rule('deny', tool='run_command')])
        result = await _agent(_one_shot_command_model('ls'), policy).run('go')
        deny_part = next(
            p
            for m in result.all_messages()
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, ToolReturnPart) and p.tool_name == 'run_command'
        )
        assert deny_part.metadata == {'pydantic_ai_harness_permission_denied': True}
        assert isinstance(deny_part.content, str)
        assert deny_part.content.startswith('[permission-policy] Permission denied for `run_command`')


class TestDenyRemovesTool:
    async def test_bare_deny_removes_tool_from_toolset(self) -> None:
        seen: list[list[str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append([t.name for t in info.function_tools])
            return ModelResponse(parts=[TextPart('done')])

        policy = PermissionPolicy[object](rules=[Rule('deny', tool='run_command')], deny_removes_tool=True)
        await _agent(FunctionModel(model_fn), policy).run('go')
        assert 'run_command' not in seen[0]

    async def test_specific_deny_keeps_tool(self) -> None:
        seen: list[list[str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append([t.name for t in info.function_tools])
            return ModelResponse(parts=[TextPart('done')])

        # A command-scoped deny is not a bare-name deny, so the tool stays available.
        policy = PermissionPolicy[object](
            rules=[Rule('deny', tool='run_command', command='git')], deny_removes_tool=True
        )
        await _agent(FunctionModel(model_fn), policy).run('go')
        assert 'run_command' in seen[0]


class TestEscalationNote:
    async def test_note_added_to_guarded_tool(self) -> None:
        descriptions: list[str | None] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descriptions.append(info.function_tools[0].description)
            return ModelResponse(parts=[TextPart('done')])

        policy = PermissionPolicy[object]()
        await _agent(FunctionModel(model_fn), policy).run('go')
        assert descriptions[0] is not None
        assert 'permission policy' in descriptions[0]

    async def test_note_can_be_disabled(self) -> None:
        descriptions: list[str | None] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descriptions.append(info.function_tools[0].description)
            return ModelResponse(parts=[TextPart('done')])

        policy = PermissionPolicy[object](add_escalation_note=False)
        await _agent(FunctionModel(model_fn), policy).run('go')
        assert descriptions[0] == 'Run a shell command.'

    async def test_note_has_no_leading_space_when_tool_has_no_description(self) -> None:
        # `_ESCALATION_NOTE` itself starts with a leading space (a separator for the common
        # case of appending after real prose). When `tool_def.description` is empty,
        # `(description or '').rstrip() + _ESCALATION_NOTE` left that leading space in place,
        # so the guarded tool's description started with a stray blank before "Note:".
        descriptions: list[str | None] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descriptions.append(info.function_tools[0].description)
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[object, str] = Agent(FunctionModel(model_fn), capabilities=[PermissionPolicy[object]()])

        @agent.tool_plain(description='')
        def run_command(command: str) -> str:  # pyright: ignore[reportUnusedFunction]
            return f'EXECUTED: {command}'  # pragma: no cover - never called; only the description is inspected

        await agent.run('go')
        assert descriptions[0] is not None
        assert descriptions[0].startswith('Note:')


class TestAskViaOnAsk:
    async def test_on_ask_approve_runs_tool(self) -> None:
        seen: list[PermissionRequest] = []

        def on_ask(ctx: RunContext[object], req: PermissionRequest) -> bool:
            seen.append(req)
            return True

        policy = PermissionPolicy[object](rules=[Rule('ask', tool='run_command', command='git status')], on_ask=on_ask)
        result = await _agent(_one_shot_command_model('git status'), policy).run('go')
        assert result.output == 'EXECUTED: git status'
        assert seen[0].tool_name == 'run_command'
        assert seen[0].command == 'git status'
        assert seen[0].args == {'command': 'git status'}

    async def test_on_ask_deny_bool_returns_default_message(self) -> None:
        policy = PermissionPolicy[object](
            rules=[Rule('ask', tool='run_command', command='git status')], on_ask=lambda ctx, req: False
        )
        result = await _agent(_one_shot_command_model('git status'), policy).run('go')
        assert 'not approved' in result.output
        assert 'EXECUTED' not in result.output

    async def test_on_ask_deny_string_returns_custom_message(self) -> None:
        policy = PermissionPolicy[object](
            rules=[Rule('ask', tool='run_command', command='git status')],
            on_ask=lambda ctx, req: 'use the API instead',
        )
        result = await _agent(_one_shot_command_model('git status'), policy).run('go')
        assert 'use the API instead' in result.output

    async def test_async_on_ask(self) -> None:
        async def on_ask(ctx: RunContext[object], req: PermissionRequest) -> bool:
            return True

        policy = PermissionPolicy[object](rules=[Rule('ask', tool='run_command', command='git status')], on_ask=on_ask)
        result = await _agent(_one_shot_command_model('git status'), policy).run('go')
        assert result.output == 'EXECUTED: git status'


class TestAskViaDeferred:
    async def test_ask_surfaces_as_deferred_requests(self) -> None:
        policy = PermissionPolicy[object](rules=[Rule('ask', tool='run_command', command='git status')])
        agent = _agent(_one_shot_command_model('git status'), policy, output_type=[str, DeferredToolRequests])
        result = await agent.run('go')
        assert isinstance(result.output, DeferredToolRequests)
        assert len(result.output.approvals) == 1
        assert result.output.approvals[0].tool_name == 'run_command'

    async def test_ask_without_handling_raises(self) -> None:
        policy = PermissionPolicy[object](rules=[Rule('ask', tool='run_command', command='git status')])
        agent = _agent(_one_shot_command_model('git status'), policy)  # output_type=str only
        with pytest.raises(UserError, match='DeferredToolRequests'):
            await agent.run('go')

    async def test_on_ask_only_resolves_its_own_marked_asks(self) -> None:
        # A `PermissionPolicy` with no `on_ask` declines resolution, so its asks bubble up
        # rather than being silently swallowed.
        policy = PermissionPolicy[object](rules=[Rule('ask', tool='run_command')])
        agent = _agent(_one_shot_command_model('anything'), policy, output_type=[str, DeferredToolRequests])
        result = await agent.run('go')
        assert isinstance(result.output, DeferredToolRequests)


class TestComposition:
    async def test_leaves_foreign_approvals_untouched(self) -> None:
        # A core `requires_approval=True` tool raises `ApprovalRequired` with no marker of
        # ours; even with an approve-everything `on_ask`, this policy must decline to resolve
        # it so the foreign approval bubbles up rather than being silently swallowed.
        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(p, ToolReturnPart) for m in messages for p in getattr(m, 'parts', [])):
                return ModelResponse(parts=[ToolCallPart(tool_name='danger', args={})])
            return ModelResponse(parts=[TextPart('done')])  # pragma: no cover - approval halts first

        policy = PermissionPolicy[object](on_ask=lambda ctx, req: True)
        agent: Agent[object, Any] = Agent(
            FunctionModel(model_fn), capabilities=[policy], output_type=[str, DeferredToolRequests]
        )

        @agent.tool_plain(requires_approval=True)
        def danger() -> str:  # pyright: ignore[reportUnusedFunction]
            """A tool that always needs approval."""
            return 'ran'  # pragma: no cover - approval halts before execution

        result = await agent.run('go')
        assert isinstance(result.output, DeferredToolRequests)
        assert result.output.approvals[0].tool_name == 'danger'


class TestApprovalProvenance:
    """`ctx.tool_call_approved` is scoped to the tool call, not to which mechanism approved
    it. This policy must not treat an approval it never itself requested as satisfying its
    own outstanding `ask` -- see `_pending_asks` in `_capability.py`.
    """

    @staticmethod
    def _ctx(*, tool_call_id: str, tool_call_approved: bool = False) -> RunContext[object]:
        return RunContext[object](
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            tool_call_id=tool_call_id,
            tool_call_approved=tool_call_approved,
        )

    async def test_unrecognized_approval_does_not_satisfy_this_policys_own_ask(self) -> None:
        handler_calls: list[dict[str, Any]] = []

        async def handler(args: dict[str, Any]) -> str:
            handler_calls.append(args)
            return 'ran'

        policy = PermissionPolicy[object](rules=[Rule('ask', tool='run_command')])
        tool_def = ToolDefinition(name='run_command')
        call = ToolCallPart(tool_name='run_command', args={'command': 'ls'}, tool_call_id='call-1')

        # First attempt: not yet approved -- must raise, and record `call-1` as its own.
        with pytest.raises(ApprovalRequired):
            await policy.wrap_tool_execute(
                self._ctx(tool_call_id='call-1'), call=call, tool_def=tool_def, args={'command': 'ls'}, handler=handler
            )
        assert handler_calls == []

        # A *different* tool_call_id, reported approved by something else -- this policy
        # never put it on hold, so the approval must not be trusted.
        with pytest.raises(ApprovalRequired):
            await policy.wrap_tool_execute(
                self._ctx(tool_call_id='call-2', tool_call_approved=True),
                call=call,
                tool_def=tool_def,
                args={'command': 'ls'},
                handler=handler,
            )
        assert handler_calls == []

        # The *same* tool_call_id this policy itself put on hold, now approved: must run.
        result = await policy.wrap_tool_execute(
            self._ctx(tool_call_id='call-1', tool_call_approved=True),
            call=call,
            tool_def=tool_def,
            args={'command': 'ls'},
            handler=handler,
        )
        assert result == 'ran'
        assert handler_calls == [{'command': 'ls'}]

    async def test_approval_with_no_tool_call_id_is_never_trusted(self) -> None:
        # Defensive: without a `tool_call_id` there is nothing to key `_pending_asks` on, so
        # an approved-but-unidentified call must still re-decide rather than assume it's ours.
        handler_calls: list[dict[str, Any]] = []

        async def handler(args: dict[str, Any]) -> str:
            handler_calls.append(args)  # pragma: no cover - must never be reached
            return 'ran'  # pragma: no cover

        policy = PermissionPolicy[object](rules=[Rule('ask', tool='run_command')])
        tool_def = ToolDefinition(name='run_command')
        call = ToolCallPart(tool_name='run_command', args={'command': 'ls'}, tool_call_id='call-1')
        ctx = RunContext[object](deps=None, model=TestModel(), usage=RunUsage(), tool_call_approved=True)

        with pytest.raises(ApprovalRequired):
            await policy.wrap_tool_execute(ctx, call=call, tool_def=tool_def, args={'command': 'ls'}, handler=handler)
        assert handler_calls == []


class TestInternals:
    def test_prepare_ignores_non_string_command(self) -> None:
        policy = PermissionPolicy[object]()
        assert policy._prepare('run_command', {'command': None}) == (None, None)  # pyright: ignore[reportPrivateUsage]
        assert policy._prepare('not_a_shell_tool', {}) == (None, None)  # pyright: ignore[reportPrivateUsage]

    def test_invalid_default_verdict_falls_back_to_ask(self) -> None:
        policy = PermissionPolicy[object](default_verdict='bogus')
        decision, _ = policy._decide('unknown_tool', {})  # pyright: ignore[reportPrivateUsage]
        assert decision.verdict == 'ask'


class TestMisc:
    def test_not_serializable(self) -> None:
        assert PermissionPolicy.get_serialization_name() is None

    async def test_non_shell_tool_uses_rules_and_default(self) -> None:
        # A tool that is not shell-class is governed by rules/default; a bare deny blocks it.
        agent: Agent[object, str] = Agent(
            _one_shot_non_shell_model(),
            capabilities=[PermissionPolicy[object](rules=[Rule('deny', tool='lookup')])],
        )

        @agent.tool_plain
        def lookup(key: str) -> str:  # pyright: ignore[reportUnusedFunction]
            """Look something up."""
            return f'VALUE:{key}'  # pragma: no cover - denied before execution

        result = await agent.run('go')
        assert 'Permission denied for `lookup`' in result.output
        assert 'VALUE' not in result.output


def _one_shot_non_shell_model() -> FunctionModel:
    state = {'called': False}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not state['called']:
            state['called'] = True
            return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={'key': 'x'})])
        last = messages[-1]
        assert isinstance(last, ModelRequest)
        ret = next(p for p in last.parts if isinstance(p, ToolReturnPart) and p.tool_name == 'lookup')
        return ModelResponse(parts=[TextPart(str(ret.content))])

    return FunctionModel(model_fn)
