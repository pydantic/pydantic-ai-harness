"""Managed AWS MCP Server capability.

External contract, verified 2026-09-04:

- The managed server is GA in `us-east-1` and `eu-central-1`; SigV4 connections
  can configure a separate default Region for AWS operations.
- Public knowledge tools accept unauthenticated remote connections. Authenticated
  connections use a caller-owned AWS Sign-In OAuth 2.1 transport or AWS's MCP Proxy
  for AWS, which owns credential resolution and SigV4 signing.
- MCP tools carry `readOnlyHint`; the proxy's read-only mode uses that hint too.
- The service has no additional charge. Called AWS services and data transfer
  retain their normal charges.

Sources:
https://docs.aws.amazon.com/agent-toolkit/latest/userguide/getting-started-aws-mcp-server.html
https://docs.aws.amazon.com/agent-toolkit/latest/userguide/understanding-mcp-server-tools.html
https://docs.aws.amazon.com/general/latest/gr/aws-mcp.html
https://aws.amazon.com/about-aws/whats-new/2026/05/aws-mcp-server/
https://github.com/aws/mcp-proxy-for-aws

Re-check the endpoint table, authentication decision guide, tool annotations,
and launch status before changing transport, access, or Region behavior.
"""

from __future__ import annotations

import re
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import BinaryContent, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool
from pydantic_core import to_json

try:
    from fastmcp.client.transports import ClientTransport
    from pydantic_ai.mcp import CallToolFunc, MCPToolset, MCPToolsetClient, ToolResult
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for the AWS capability. Install `pydantic-ai-harness[aws]`.'
    ) from _import_error

AWSAccess = Literal['read_only', 'approval_required', 'unrestricted']
"""Which managed AWS MCP tools the agent may execute."""

_ENDPOINTS = {
    'us-east-1': 'https://aws-mcp.us-east-1.api.aws/mcp',
    'eu-central-1': 'https://aws-mcp.eu-central-1.api.aws/mcp',
}
_ACCOUNT_ID_PATTERN = re.compile(r'[0-9]{12}')
_REGION_PATTERN = re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)+')
_DEFAULT_DESCRIPTION = 'Use the managed AWS MCP Server for one AWS account and target Region.'
_DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
_DEFAULT_MAX_OUTPUT_LINES = 2000
_OUTPUT_TRUNCATED = '[... AWS MCP output truncated ...]'
_INSTRUCTIONS = (
    'This AWS capability is scoped to account `{account_id}` and target Region `{region}`. Treat both values '
    'as required context for every tool from this capability. {identity_authority} Do not use this capability to switch '
    'accounts or target Regions. '
    'Prefer AWS documentation '
    'and read operations before proposing changes. After a failed change with an unknown outcome, inspect current state '
    'before retrying. This is real AWS, not the LocalStack emulator. Access mode is `{access}` and authentication mode '
    'is `{authentication}`.'
)


def _validate_configuration(
    account_id: object,
    region: object,
    endpoint_region: str,
    access: str,
    authentication: str,
    managed_transport: ClientTransport | None,
    max_output_bytes: int,
    max_output_lines: int,
) -> tuple[AWSAccess, Literal['unauthenticated', 'oauth', 'sigv4']]:
    if not isinstance(account_id, str) or _ACCOUNT_ID_PATTERN.fullmatch(account_id) is None:
        raise UserError('`account_id` must be a 12-digit AWS account ID.')
    if not isinstance(region, str) or _REGION_PATTERN.fullmatch(region) is None:
        raise UserError('`region` must be an AWS Region identifier such as `us-west-2`.')
    if endpoint_region not in _ENDPOINTS:
        supported = ', '.join(f'`{value}`' for value in _ENDPOINTS)
        raise UserError(f'`endpoint_region` must be one of {supported}.')
    if access not in ('read_only', 'approval_required', 'unrestricted'):
        raise UserError('`access` must be `read_only`, `approval_required`, or `unrestricted`.')
    if authentication not in ('unauthenticated', 'oauth', 'sigv4'):
        raise UserError('`authentication` must be `unauthenticated`, `oauth`, or `sigv4`.')
    if authentication == 'unauthenticated' and managed_transport is not None:
        raise UserError('`managed_transport` requires `authentication="oauth"` or `authentication="sigv4"`.')
    if authentication != 'unauthenticated' and managed_transport is None:
        raise UserError(f'`authentication="{authentication}"` requires a caller-owned `managed_transport`.')
    if managed_transport is not None and not isinstance(managed_transport, ClientTransport):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise UserError('Authenticated AWS connections require a pre-built FastMCP `ClientTransport`.')
    for name, value in (('max_output_bytes', max_output_bytes), ('max_output_lines', max_output_lines)):
        if type(value) is not int or value <= 0:
            raise UserError(f'`{name}` must be a positive integer.')
    if max_output_bytes < len(_OUTPUT_TRUNCATED.encode()):
        raise UserError(f'`max_output_bytes` must be at least {len(_OUTPUT_TRUNCATED.encode())}.')
    return access, authentication


def _is_explicitly_read_only(tool_def: ToolDefinition) -> bool:
    metadata = tool_def.metadata
    if metadata is None:  # pragma: no cover - MCPToolset always supplies a metadata mapping
        return False
    annotations = metadata.get('annotations')
    match annotations:
        case {'readOnlyHint': True}:
            return True
        case _:
            return False


def _requires_approval(_ctx: RunContext[AgentDepsT], tool_def: ToolDefinition, _tool_args: dict[str, object]) -> bool:
    return not _is_explicitly_read_only(tool_def)


def _limit_text(text: str, *, max_bytes: int, max_lines: int) -> str:
    if len(text.encode()) <= max_bytes and len(text.splitlines()) <= max_lines:
        return text
    if max_lines == 1:
        return _OUTPUT_TRUNCATED
    preview_bytes = max(0, max_bytes - len(_OUTPUT_TRUNCATED.encode()) - 1)
    preview = '\n'.join(text.splitlines()[: max_lines - 1])
    preview = preview.encode()[:preview_bytes].decode(errors='ignore')
    return f'{preview}\n{_OUTPUT_TRUNCATED}' if preview else _OUTPUT_TRUNCATED


def _limit_tool_result(result: ToolResult, *, tool_name: str, max_bytes: int, max_lines: int) -> ToolResult:
    rendered, user_content = ToolReturnPart(tool_name=tool_name, content=result).model_response_str_and_user_content()
    text = '\n'.join([rendered, *(item for item in user_content if isinstance(item, str))])
    if any(isinstance(item, BinaryContent) for item in user_content):
        serialized = to_json(result)
        visible_bytes = len(text.encode()) + len(serialized)
        visible_lines = max(1, len(text.splitlines())) + max(1, serialized.count(b'\n') + 1)
        return result if visible_bytes <= max_bytes and visible_lines <= max_lines else _OUTPUT_TRUNCATED
    limited = _limit_text(text, max_bytes=max_bytes, max_lines=max_lines)
    return result if limited == text else limited


@dataclass(frozen=True)
class _LimitToolOutput:
    max_bytes: int
    max_lines: int

    async def __call__(
        self,
        _ctx: RunContext[object],
        call_tool: CallToolFunc,
        name: str,
        args: dict[str, object],
    ) -> ToolResult:
        try:
            result = await call_tool(name, args)
        except ModelRetry as exc:
            message = str(exc)
            limited = _limit_text(message, max_bytes=self.max_bytes, max_lines=self.max_lines)
            if limited == message:
                raise
            raise ModelRetry(limited) from exc
        return _limit_tool_result(result, tool_name=name, max_bytes=self.max_bytes, max_lines=self.max_lines)


class _AWSToolset(MCPToolset[AgentDepsT]):
    """Managed AWS MCP connection with conservative annotation filtering."""

    def __init__(
        self,
        *,
        endpoint_region: Literal['us-east-1', 'eu-central-1'],
        read_only: bool,
        client: MCPToolsetClient | None,
        id: str,
        account_id: str,
        region: str,
        max_output_bytes: int,
        max_output_lines: int,
    ) -> None:
        output_limiter = _LimitToolOutput(max_bytes=max_output_bytes, max_lines=max_output_lines)
        if client is None:
            super().__init__(
                _ENDPOINTS[endpoint_region], id=id, include_instructions=False, process_tool_call=output_limiter
            )
        else:
            super().__init__(client, id=id, include_instructions=False, process_tool_call=output_limiter)
        self._read_only = read_only
        self._scope_description = f'AWS scope: account {account_id}, target Region {region}.'

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        if not tools:
            raise UserError(
                'The managed AWS MCP Server returned no tools. Check authentication and retry the connection; '
                'an empty catalog can indicate throttled initialization.'
            )
        tools = {
            name: replace(
                tool,
                tool_def=replace(
                    tool.tool_def,
                    description=f'{self._scope_description} {tool.tool_def.description or ""}'.rstrip(),
                ),
            )
            for name, tool in tools.items()
        }
        if not self._read_only:
            return tools
        read_tools = {name: tool for name, tool in tools.items() if _is_explicitly_read_only(tool.tool_def)}
        if not read_tools:
            raise UserError(
                'The managed AWS MCP Server returned no tools explicitly marked read-only. '
                'Its safety annotations may have changed.'
            )
        return read_tools


@dataclass
class AWS(AbstractCapability[AgentDepsT]):
    """Access one real AWS account and target Region through the managed AWS MCP Server.

    Direct connections use unauthenticated public knowledge tools. Pass a
    trusted caller-owned MCP transport for AWS Sign-In OAuth or AWS's SigV4 MCP
    proxy so the transport retains its identity lifecycle.
    The default exposes only tools whose MCP annotation explicitly marks them
    read-only. `approval_required` uses Pydantic AI's tool approval wrapper for
    every new non-read tool call, including a model-initiated retry, and `unrestricted`
    requires an explicit opt-in.
    """

    account_id: str
    """Declared 12-digit AWS account for this capability instance."""

    region: str
    """Declared target Region for AWS operations, for example `us-west-2`."""

    _: KW_ONLY

    endpoint_region: Literal['us-east-1', 'eu-central-1'] = 'us-east-1'
    """Managed endpoint for direct unauthenticated connections."""

    access: AWSAccess = 'read_only'
    """Expose read-only tools, approval-gate other tools, or expose all tools."""

    authentication: Literal['unauthenticated', 'oauth', 'sigv4'] = 'unauthenticated'
    """Use public knowledge tools, or identify the caller-owned OAuth or SigV4 transport."""

    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES
    """Maximum UTF-8 payload bytes retained from each managed tool result. Must be at least 34."""

    max_output_lines: int = _DEFAULT_MAX_OUTPUT_LINES
    """Maximum payload lines retained from each managed tool result."""

    id: str | None = None
    """Stable capability ID, derived from account, target Region, and endpoint Region when omitted."""

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    managed_transport: ClientTransport | None = field(default=None, repr=False)
    """Trusted caller-owned transport connected to the managed AWS MCP Server.

    For SigV4 one-account scope, configure exactly one proxy profile. Passing
    this value asserts that the transport reaches AWS's managed endpoint and that
    its identity matches `account_id`. Harness does not inspect credentials,
    sign requests, refresh tokens, or alter the transport.
    """

    def __post_init__(self) -> None:
        self.access, self.authentication = _validate_configuration(
            self.account_id,
            self.region,
            self.endpoint_region,
            self.access,
            self.authentication,
            self.managed_transport,
            self.max_output_bytes,
            self.max_output_lines,
        )
        self.id = self._derived_id()

    def _derived_id(self) -> str:
        return self.id if self.id is not None else f'aws-{self.account_id}-{self.region}-{self.endpoint_region}'

    def get_toolset(self) -> AbstractToolset[AgentDepsT]:
        """Build the managed AWS MCP toolset and apply the selected access policy."""
        toolset = _AWSToolset[AgentDepsT](
            endpoint_region=self.endpoint_region,
            read_only=self.access == 'read_only',
            client=self.managed_transport,
            id=self._derived_id(),
            account_id=self.account_id,
            region=self.region,
            max_output_bytes=self.max_output_bytes,
            max_output_lines=self.max_output_lines,
        )
        if self.access == 'approval_required':
            return toolset.approval_required(_requires_approval)
        return toolset

    def get_instructions(self) -> str | None:
        """Return the declared account, Region, identity, and access scope."""
        identity_authority = (
            'There is no authenticated IAM identity; use only public knowledge tools.'
            if self.authentication == 'unauthenticated'
            else 'The authenticated IAM identity is the authority; do not claim access that its policies deny.'
        )
        return _INSTRUCTIONS.format(
            account_id=self.account_id,
            region=self.region,
            identity_authority=identity_authority,
            access=self.access,
            authentication=self.authentication,
        )

    @classmethod
    def from_spec(
        cls,
        account_id: str,
        region: str,
        *,
        endpoint_region: Literal['us-east-1', 'eu-central-1'] = 'us-east-1',
        access: AWSAccess = 'read_only',
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        max_output_lines: int = _DEFAULT_MAX_OUTPUT_LINES,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
    ) -> AWS[AgentDepsT]:
        """Construct from serializable options, excluding the runtime-only managed transport."""
        return cls(
            account_id=account_id,
            region=region,
            endpoint_region=endpoint_region,
            access=access,
            authentication='unauthenticated',
            max_output_bytes=max_output_bytes,
            max_output_lines=max_output_lines,
            id=id,
            description=description,
            defer_loading=defer_loading,
        )

    @classmethod
    def get_serialization_name(cls) -> str:
        """Return the agent-spec capability name."""
        return 'AWS'
