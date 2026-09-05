"""Cloudflare managed MCP transport and policy.

External contract, verified 2026-09-04:

- Cloudflare recommends `https://mcp.cloudflare.com/mcp` for broad API access. It exposes
  `docs`, `search`, and `execute`; `execute` can read or mutate and is marked destructive.
- Focused `*.mcp.cloudflare.com/mcp` servers expose typed product tools. Their
  `readOnlyHint` annotations distinguish calls safe to expose without mutation opt-in.
- Managed servers support browser OAuth and bearer API tokens. Focused authenticated
  servers expose account selection through explicit tool arguments when the credential
  can access multiple accounts. Code Mode accepts `account_id` on `execute` instead.
- Cloudflare exposes no managed-server zone header. Focused tool schemas use `zone_id`,
  `zoneId`, or `zone`; this toolset restricts and fills those arguments when `zone_id` is set.

Sources: https://github.com/cloudflare/mcp,
https://github.com/cloudflare/mcp-server-cloudflare, and
https://github.com/cloudflare/agents. Re-check their server registrations and the
Code Mode `src/tools` implementations when the catalog or policy changes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from fractions import Fraction
from math import lcm
from pathlib import Path
from typing import Any

from pydantic import AnyUrl, TypeAdapter
from pydantic_ai.exceptions import ApprovalRequired, ModelRetry, ToolFailed, UserError
from pydantic_ai.tools import AgentDepsT, ObjectJsonSchema, RunContext
from pydantic_ai.toolsets import ToolsetTool
from pydantic_core import to_json

try:
    from pydantic_ai.mcp import MCPToolset, MCPToolsetClient
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'MCP support is required for Cloudflare. Install it with: uv add "pydantic-ai-harness[cloudflare]"'
    ) from _import_error

__all__ = ['CloudflareServer', 'CloudflareToolset', 'MCPToolsetClient']


class CloudflareServer(str, Enum):
    """Official Cloudflare managed MCP server selection."""

    API = 'api'
    DOCS = 'docs'
    AGENTS_SDK_DOCS = 'agents_sdk_docs'
    WORKERS_BINDINGS = 'workers_bindings'
    WORKERS_BUILDS = 'workers_builds'
    OBSERVABILITY = 'observability'
    CONTAINERS = 'containers'
    BROWSER = 'browser'
    LOGPUSH = 'logpush'
    AI_GATEWAY = 'ai_gateway'
    AUDIT_LOGS = 'audit_logs'
    DNS_ANALYTICS = 'dns_analytics'
    DEX = 'dex'
    CASB = 'casb'
    DEVELOPER_STACK = 'developer_stack'
    BLOG = 'blog'
    DEMO_DAY = 'demo_day'


@dataclass(frozen=True)
class _ServerConfig:
    url: str
    public: bool = False
    safe_tools: frozenset[str] = frozenset()


_SERVERS: dict[CloudflareServer, _ServerConfig] = {
    CloudflareServer.API: _ServerConfig('https://mcp.cloudflare.com/mcp', safe_tools=frozenset({'docs', 'search'})),
    CloudflareServer.DOCS: _ServerConfig('https://docs.mcp.cloudflare.com/mcp', public=True),
    CloudflareServer.AGENTS_SDK_DOCS: _ServerConfig(
        'https://agents.cloudflare.com/mcp', public=True, safe_tools=frozenset({'search-agent-docs'})
    ),
    CloudflareServer.WORKERS_BINDINGS: _ServerConfig('https://bindings.mcp.cloudflare.com/mcp'),
    CloudflareServer.WORKERS_BUILDS: _ServerConfig('https://builds.mcp.cloudflare.com/mcp'),
    CloudflareServer.OBSERVABILITY: _ServerConfig('https://observability.mcp.cloudflare.com/mcp'),
    CloudflareServer.CONTAINERS: _ServerConfig('https://containers.mcp.cloudflare.com/mcp'),
    CloudflareServer.BROWSER: _ServerConfig('https://browser.mcp.cloudflare.com/mcp'),
    CloudflareServer.LOGPUSH: _ServerConfig('https://logs.mcp.cloudflare.com/mcp'),
    CloudflareServer.AI_GATEWAY: _ServerConfig('https://ai-gateway.mcp.cloudflare.com/mcp'),
    CloudflareServer.AUDIT_LOGS: _ServerConfig('https://auditlogs.mcp.cloudflare.com/mcp'),
    CloudflareServer.DNS_ANALYTICS: _ServerConfig('https://dns-analytics.mcp.cloudflare.com/mcp'),
    CloudflareServer.DEX: _ServerConfig('https://dex.mcp.cloudflare.com/mcp'),
    CloudflareServer.CASB: _ServerConfig('https://casb.mcp.cloudflare.com/mcp'),
    CloudflareServer.DEVELOPER_STACK: _ServerConfig(
        'https://stack.mcp.cloudflare.com/mcp',
        public=True,
        safe_tools=frozenset({'list_libraries', 'search_dev_stack'}),
    ),
    CloudflareServer.BLOG: _ServerConfig('https://blog.mcp.cloudflare.com/mcp', public=True),
    CloudflareServer.DEMO_DAY: _ServerConfig(
        'https://demo-day.mcp.cloudflare.com/mcp', public=True, safe_tools=frozenset({'mcp_demo_day_info'})
    ),
}

_ACCOUNT_KEYS = ('account_id', 'accountId')
_ZONE_KEYS = ('zone_id', 'zoneId', 'zone')
_PAGE_KEYS = ('limit', 'per_page', 'perPage', 'page_size', 'pageSize', 'first', 'k', 'limitPerGroup')
_PAGE_CONTAINER_KEYS = ('query', 'keysQuery', 'valuesQuery')
_TRUNCATION_MARKER = '[... Cloudflare result truncated ...]'
_ERROR_ENVELOPE_BYTES = len(to_json({'error': ''}))
_MAX_SCHEMA_DEPTH = 64
_REF_ANNOTATION_KEYS = frozenset(
    {
        '$anchor',
        '$comment',
        '$defs',
        '$dynamicAnchor',
        '$id',
        '$schema',
        '$vocabulary',
        'default',
        'definitions',
        'deprecated',
        'description',
        'examples',
        'readOnly',
        'title',
        'writeOnly',
    }
)
_OBJECT_SCHEMA_KEYS = _REF_ANNOTATION_KEYS | frozenset(
    {'type', 'properties', 'required', 'additionalProperties', 'minProperties', 'maxProperties'}
)
_OBJECT_DICT = TypeAdapter(dict[str, object])
_STRING_LIST = TypeAdapter(list[str])
_OBJECT_LIST = TypeAdapter(list[object])


@dataclass(frozen=True)
class _PageConstraints:
    minimum: int = 1
    maximum: int | None = None
    multiple: int = 1
    allowed: frozenset[int] | None = None

    def merge(self, other: _PageConstraints) -> _PageConstraints:
        maximum = (
            min(self.maximum, other.maximum)
            if self.maximum is not None and other.maximum is not None
            else self.maximum
            if self.maximum is not None
            else other.maximum
        )
        allowed = (
            self.allowed & other.allowed
            if self.allowed is not None and other.allowed is not None
            else self.allowed
            if self.allowed is not None
            else other.allowed
        )
        return _PageConstraints(
            minimum=max(self.minimum, other.minimum),
            maximum=maximum,
            multiple=lcm(self.multiple, other.multiple),
            allowed=allowed,
        )

    def accepts(self, value: object) -> bool:
        integer = _page_integer(value)
        if integer is None:
            return False
        if integer < self.minimum or (self.maximum is not None and integer > self.maximum):
            return False
        if integer % self.multiple != 0:
            return False
        return self.allowed is None or integer in self.allowed

    def limit(self, configured: int) -> int | None:
        upper = min(configured, self.maximum) if self.maximum is not None else configured
        if self.allowed is not None:
            candidates = [value for value in self.allowed if value <= upper and self.accepts(value)]
            return max(candidates, default=None)
        candidate = upper - (upper % self.multiple)
        return candidate if self.accepts(candidate) else None


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return _OBJECT_DICT.validate_python(value)


def _resolve_schema(root: dict[str, object], schema: object, seen: frozenset[str] = frozenset()) -> object:
    field = _object_dict(schema)
    reference = field.get('$ref')
    if (
        not isinstance(reference, str)
        or not reference.startswith('#/')
        or reference in seen
        or len(seen) >= _MAX_SCHEMA_DEPTH
    ):
        return schema
    if any(key not in _REF_ANNOTATION_KEYS and not key.startswith('x-') for key in field if key != '$ref'):
        return schema
    target: object = root
    for raw_part in reference[2:].split('/'):
        part = raw_part.replace('~1', '/').replace('~0', '~')
        target_dict = _object_dict(target)
        if part not in target_dict:
            return schema
        target = target_dict[part]
    resolved_target = _resolve_schema(root, target, seen | {reference})
    if not isinstance(resolved_target, dict):
        return schema
    resolved = _OBJECT_DICT.validate_python(resolved_target)
    return {**resolved, **{key: value for key, value in field.items() if key != '$ref'}}


def _annotations(tool: ToolsetTool[AgentDepsT]) -> dict[str, object]:
    metadata = tool.tool_def.metadata or {}
    value: object = metadata.get('annotations')
    return _object_dict(value)


def _is_read_only(
    server: CloudflareServer,
    tool: ToolsetTool[AgentDepsT],
    *,
    official_client: bool,
    trust_server_annotations: bool,
) -> bool:
    annotations = _annotations(tool)
    if annotations.get('readOnlyHint') is False or annotations.get('destructiveHint') is True:
        return False
    if official_client and tool.tool_def.name in _SERVERS[server].safe_tools:
        return True
    return (
        trust_server_annotations
        and annotations.get('readOnlyHint') is True
        and annotations.get('destructiveHint') is not True
    )


def _properties(tool: ToolsetTool[AgentDepsT]) -> dict[str, object]:
    root = tool.tool_def.parameters_json_schema
    resolved_root = _object_dict(_resolve_schema(root, root))
    value: object = resolved_root.get('properties')
    return _object_dict(value)


def _scope_keys(tool: ToolsetTool[AgentDepsT], candidates: tuple[str, ...]) -> tuple[str, ...]:
    properties = _properties(tool)
    return tuple(key for key in candidates if key in properties)


def _is_api_safe_tool(server: CloudflareServer, tool: ToolsetTool[AgentDepsT], *, official_client: bool) -> bool:
    return official_client and server is CloudflareServer.API and tool.tool_def.name in _SERVERS[server].safe_tools


def _page_integer(value: object) -> int | None:
    if type(value) is int:
        return value
    if type(value) is float and value.is_integer():
        return int(value)
    return None


def _fraction(value: object) -> Fraction | None:
    if type(value) not in (int, float):
        return None
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None


def _union_constraints(variants: list[object], depth: int) -> _PageConstraints | None:
    numeric_variants: list[_PageConstraints] = []
    for variant in variants:
        if variant is False:
            continue
        if variant is True:
            return None
        variant_schema = _object_dict(variant)
        if variant_schema.get('type') == 'null':
            continue
        variant_constraints = _page_constraints(variant, depth + 1)
        if variant_constraints is None:
            return None
        numeric_variants.append(variant_constraints)
    return numeric_variants[0] if len(numeric_variants) == 1 else None


def _range_constraints(field: dict[str, object]) -> _PageConstraints | None:
    constraints = _PageConstraints()
    for keyword, lower, exclusive in (
        ('minimum', True, False),
        ('exclusiveMinimum', True, True),
        ('maximum', False, False),
        ('exclusiveMaximum', False, True),
    ):
        if keyword not in field:
            continue
        value = _fraction(field[keyword])
        if value is None:
            return None
        if lower:
            minimum = (
                value.numerator // value.denominator + 1 if exclusive else -(-value.numerator // value.denominator)
            )
            constraints = constraints.merge(_PageConstraints(minimum=minimum))
        else:
            maximum = (
                -(-value.numerator // value.denominator) - 1 if exclusive else value.numerator // value.denominator
            )
            constraints = constraints.merge(_PageConstraints(maximum=maximum))
    return constraints


def _discrete_constraints(field: dict[str, object]) -> _PageConstraints | None:
    constraints = _PageConstraints()
    if 'multipleOf' in field:
        multiple = _fraction(field['multipleOf'])
        if multiple is None or multiple <= 0:
            return None
        constraints = constraints.merge(_PageConstraints(multiple=multiple.numerator))
    if 'enum' in field:
        values = field['enum']
        if not isinstance(values, list):
            return None
        constraints = constraints.merge(
            _PageConstraints(
                allowed=frozenset(
                    value for item in _OBJECT_LIST.validate_python(values) if (value := _page_integer(item)) is not None
                )
            )
        )
    if 'const' in field:
        value = _page_integer(field['const'])
        constraints = constraints.merge(_PageConstraints(allowed=frozenset() if value is None else frozenset({value})))
    return constraints


def _page_constraints(schema: object, depth: int = 0) -> _PageConstraints | None:
    if depth >= _MAX_SCHEMA_DEPTH:
        return None
    field = _object_dict(schema)
    if not field:
        return None
    field_type = field.get('type', 'integer')
    if isinstance(field_type, list):
        raw_types = _OBJECT_LIST.validate_python(field_type)
        if not all(isinstance(item, str) for item in raw_types):
            return None
        types = [item for item in raw_types if isinstance(item, str)]
        field_type = (
            next((item for item in types if item != 'null'), None)
            if set(types) <= {'integer', 'number', 'null'}
            else None
        )
    if (
        any(keyword in field for keyword in ('$ref', '$dynamicRef', '$recursiveRef'))
        or field_type not in ('integer', 'number')
        or any(keyword in field for keyword in ('not', 'if', 'then', 'else'))
    ):
        return None
    range_constraints = _range_constraints(field)
    discrete_constraints = _discrete_constraints(field)
    if range_constraints is None or discrete_constraints is None:
        return None
    constraints = range_constraints.merge(discrete_constraints)
    for keyword in ('anyOf', 'oneOf'):
        if keyword not in field:
            continue
        variants = field[keyword]
        if not isinstance(variants, list):
            return None
        variant_constraints = _union_constraints(_OBJECT_LIST.validate_python(variants), depth)
        if variant_constraints is None:
            return None
        constraints = constraints.merge(variant_constraints)
    if 'allOf' in field:
        variants = field['allOf']
        if not isinstance(variants, list):
            return None
        for variant in _OBJECT_LIST.validate_python(variants):
            variant_constraints = _page_constraints(variant, depth + 1)
            if variant_constraints is None:
                return None
            constraints = constraints.merge(variant_constraints)
    return constraints


def _page_limit(schema: object, configured: int) -> int | None:
    constraints = _page_constraints(schema)
    return constraints.limit(configured) if constraints is not None else None


def _object_schema_type(field: dict[str, object]) -> bool | None:
    field_type = field.get('type')
    if field_type is None or field_type == 'object':
        return True
    if isinstance(field_type, str):
        return False
    if not isinstance(field_type, list):
        return None
    raw_types = _OBJECT_LIST.validate_python(field_type)
    if not raw_types or not all(isinstance(item, str) for item in raw_types):
        return None
    types = set(raw_types)
    if len(types) != len(raw_types):
        return None
    if 'object' not in types:
        return False
    return True if types <= {'object', 'null'} else None


def _pagination_fields(tool: ToolsetTool[AgentDepsT]) -> list[tuple[tuple[str, ...], object]]:
    properties = _properties(tool)
    root = tool.tool_def.parameters_json_schema
    fields: list[tuple[tuple[str, ...], object]] = [
        ((key,), _resolve_schema(root, properties[key])) for key in _PAGE_KEYS if key in properties
    ]
    for container_key in _PAGE_CONTAINER_KEYS:
        container = _object_dict(_resolve_schema(root, properties.get(container_key)))
        if _object_schema_type(container) is not True:
            continue
        nested = _object_dict(container.get('properties'))
        fields.extend(((container_key, key), _resolve_schema(root, nested[key])) for key in _PAGE_KEYS if key in nested)
    return fields


def _supported_object_schema(schema: object) -> dict[str, object] | None:
    field = _object_dict(schema)
    if not isinstance(schema, dict):
        return None
    if any(key not in _OBJECT_SCHEMA_KEYS and not key.startswith('x-') for key in field):
        return None
    if _object_schema_type(field) is not True:
        return None
    properties = field.get('properties')
    if properties is not None and not isinstance(properties, dict):
        return None
    required = field.get('required')
    if required is not None:
        if not isinstance(required, list):
            return None
        required_items = _OBJECT_LIST.validate_python(required)
        if not all(isinstance(key, str) for key in required_items):
            return None
    return field


def _supports_result_limit(tool: ToolsetTool[AgentDepsT], configured: int) -> bool:
    root = tool.tool_def.parameters_json_schema
    resolved_root = _supported_object_schema(_resolve_schema(root, root))
    if resolved_root is None:
        return False
    properties = _object_dict(resolved_root.get('properties'))
    for container_key in _PAGE_CONTAINER_KEYS:
        if container_key not in properties:
            continue
        resolved_container = _resolve_schema(root, properties[container_key])
        container_field = _object_dict(resolved_container)
        object_type = _object_schema_type(container_field)
        if object_type is False:
            continue
        if object_type is None:
            return False
        container = _supported_object_schema(resolved_container)
        if container is None:
            return False
    return all(_page_limit(schema, configured) is not None for _, schema in _pagination_fields(tool))


def _bounded_page_schema(schema: object, configured: int) -> dict[str, object] | None:
    field = _object_dict(schema)
    if not field:
        return None
    effective_limit = _page_limit(field, configured)
    if effective_limit is None:  # pragma: no cover - filtered before schema preparation
        return None
    existing_maximum = field.get('maximum')
    if not isinstance(existing_maximum, (int, float)) or existing_maximum > effective_limit:
        field['maximum'] = effective_limit
    field['default'] = effective_limit
    return field


def _take_utf8_prefix(text: str, byte_limit: int) -> str:
    return text.encode('utf-8')[:byte_limit].decode('utf-8', errors='ignore')


def _bounded_text(text: str, *, max_bytes: int, max_lines: int) -> str:
    lines = text.splitlines()
    lines_exceeded = len(lines) > max_lines
    bytes_exceeded = len(text.encode('utf-8')) > max_bytes
    if not lines_exceeded and not bytes_exceeded:
        return text

    marker = _TRUNCATION_MARKER
    if max_lines == 1:
        return _take_utf8_prefix(marker, max_bytes)
    marker_bytes = len(marker.encode('utf-8')) + 1
    if marker_bytes >= max_bytes:
        return _take_utf8_prefix(marker, max_bytes)
    body_lines = lines[: max_lines - 1]
    body = _take_utf8_prefix('\n'.join(body_lines), max_bytes - marker_bytes).rstrip('\n')
    return f'{body}\n{marker}' if body else marker


def _bounded_error_text(text: str, *, max_bytes: int, max_lines: int) -> str:
    candidate = _bounded_text(text, max_bytes=max_bytes, max_lines=max_lines)
    if len(to_json({'error': candidate})) <= max_bytes:
        return candidate
    best = ''
    lower = 0
    upper = min(max_bytes, len(text.encode('utf-8')))
    while lower <= upper:
        budget = (lower + upper) // 2
        candidate = _bounded_text(text, max_bytes=budget, max_lines=max_lines)
        if len(to_json({'error': candidate})) <= max_bytes:
            best = candidate
            lower = budget + 1
        else:
            upper = budget - 1
    return best


class CloudflareToolset(MCPToolset[AgentDepsT]):
    """One official Cloudflare managed MCP server with client-side policy.

    The toolset selects one server, filters its tools to the read-safe set by
    default, injects configured resource boundaries, bounds result sizes, and
    sends mutation-capable tools through Pydantic AI's approval flow. It keeps
    the rest of the public `MCPToolset` surface for toolset composition.

    Use `Cloudflare` for capability instructions and agent-spec support. Use
    this class directly with toolset combinators.
    """

    def __init__(
        self,
        *,
        server: CloudflareServer | str = CloudflareServer.DOCS,
        account_id: str | None = None,
        zone_id: str | None = None,
        api_token: str | None = None,
        allow_mutations: bool = False,
        max_results: int = 20,
        max_output_bytes: int = 50 * 1024,
        max_output_lines: int = 500,
        client: MCPToolsetClient | None = None,
        trust_server_annotations: bool = False,
        id: str = 'cloudflare',
        include_instructions: bool = True,
    ) -> None:
        """Connect to one managed server with conservative execution policy.

        Args:
            server: Managed endpoint selected from Cloudflare's catalog.
            account_id: Account enforced through explicit focused-server tool
                arguments. Not supported by public servers.
            zone_id: Zone enforced through explicit tool arguments.
            api_token: Bearer token. Authenticated servers use OAuth when omitted.
            allow_mutations: Expose tools outside the read-safe set. Their calls
                still require Pydantic AI approval.
            max_results: Maximum value for recognized pagination arguments.
            max_output_bytes: Maximum serialized UTF-8 bytes returned per call.
            max_output_lines: Maximum serialized lines returned per call.
            client: Prebuilt MCP client or transport. It owns authentication and
                account selection; zone and execution policies still apply.
            trust_server_annotations: Treat a custom client's `readOnlyHint` as
                authorization to run without mutation opt-in or approval.
            id: Toolset identifier.
            include_instructions: Include remote MCP server instructions when
                no account or zone scope is configured. `Cloudflare` uses the
                same flag for its own capability guidance.
        """
        try:
            resolved_server = CloudflareServer(server)
        except ValueError as e:
            values = ', '.join(repr(item.value) for item in CloudflareServer)
            raise UserError(f'`server` must be one of: {values}.') from e
        for name, value in (
            ('max_results', max_results),
            ('max_output_bytes', max_output_bytes),
            ('max_output_lines', max_output_lines),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}.')
        if max_output_bytes < _ERROR_ENVELOPE_BYTES:
            raise ValueError(
                f'max_output_bytes must be at least {_ERROR_ENVELOPE_BYTES} bytes to contain a tool error result.'
            )
        if (
            resolved_server is CloudflareServer.API
            and (account_id is not None or zone_id is not None)
            and allow_mutations
        ):
            raise UserError(
                "Cloudflare's Code Mode `execute` tool accepts arbitrary JavaScript, so this client cannot enforce "
                'an account or zone boundary on it. Select a focused server with explicit resource arguments.'
            )
        server_config = _SERVERS[resolved_server]
        if server_config.public:
            if account_id is not None or zone_id is not None:
                raise UserError(f'The public Cloudflare `{resolved_server.value}` server has no account or zone scope.')
            if api_token is not None:
                raise UserError(f'The public Cloudflare `{resolved_server.value}` server does not accept `api_token`.')
        if isinstance(client, (str, Path, AnyUrl)):
            raise UserError(
                '`client` must be a prebuilt MCP client or transport, not an address. Omit it to use the selected '
                'managed server with configured OAuth or token authentication.'
            )

        resolved_client: MCPToolsetClient = client if client is not None else server_config.url
        if client is not None and (api_token is not None or account_id is not None):
            raise UserError(
                '`client` owns its authentication and account selection; do not also pass `api_token` or `account_id`.'
            )
        remote_instructions = include_instructions and account_id is None and zone_id is None
        if client is None:
            auth = None if server_config.public else (api_token if api_token is not None else 'oauth')
            super().__init__(
                resolved_client,
                id=id,
                include_instructions=remote_instructions,
                auth=auth,
                cache_tools=False,
            )
        else:
            super().__init__(resolved_client, id=id, include_instructions=remote_instructions, cache_tools=False)
        self._official_client = client is None
        self._trust_server_annotations = self._official_client or trust_server_annotations
        if not self._official_client:
            self.include_instructions = False
        self.server = resolved_server
        self.account_id = account_id
        self.zone_id = zone_id
        self.allow_mutations = allow_mutations
        self.max_results = max_results
        self.max_output_bytes = max_output_bytes
        self.max_output_lines = max_output_lines

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        selected: dict[str, ToolsetTool[AgentDepsT]] = {}
        for name, tool in tools.items():
            if not self._selects_tool(tool):
                continue
            selected[name] = replace(
                tool, tool_def=replace(tool.tool_def, parameters_json_schema=self._bounded_schema(tool))
            )
        return selected

    def _selects_tool(self, tool: ToolsetTool[AgentDepsT]) -> bool:
        read_only = _is_read_only(
            self.server,
            tool,
            official_client=self._official_client,
            trust_server_annotations=self._trust_server_annotations,
        )
        if not read_only and not self.allow_mutations:
            return False
        api_safe = _is_api_safe_tool(self.server, tool, official_client=self._official_client)
        if self.account_id is not None and not api_safe and not _scope_keys(tool, _ACCOUNT_KEYS):
            return False
        if self.zone_id is not None and not api_safe and not _scope_keys(tool, _ZONE_KEYS):
            return False
        return _supports_result_limit(tool, self.max_results)

    def _bounded_schema(self, tool: ToolsetTool[AgentDepsT]) -> ObjectJsonSchema:
        root = tool.tool_def.parameters_json_schema
        schema = {**root, **_object_dict(_resolve_schema(root, root))}
        schema.pop('$ref', None)
        properties = _properties(tool)
        bounded_properties: ObjectJsonSchema = dict(properties)
        for key in _PAGE_KEYS:
            bounded = _bounded_page_schema(_resolve_schema(root, properties.get(key)), self.max_results)
            if bounded is not None:
                bounded_properties[key] = bounded
        for container_key in _PAGE_CONTAINER_KEYS:
            container = _object_dict(_resolve_schema(root, properties.get(container_key)))
            if _object_schema_type(container) is not True:
                continue
            nested = _object_dict(container.get('properties'))
            if not nested:
                continue
            bounded_nested = dict(nested)
            for key in _PAGE_KEYS:
                bounded = _bounded_page_schema(_resolve_schema(root, nested.get(key)), self.max_results)
                if bounded is not None:
                    bounded_nested[key] = bounded
            container['properties'] = bounded_nested
            bounded_properties[container_key] = container
        schema['properties'] = bounded_properties
        required = schema.get('required')
        required_keys = _STRING_LIST.validate_python(required) if isinstance(required, list) else []
        scoped_keys: set[str] = set()
        required_scope_keys: list[str] = []
        read_only = _is_read_only(
            self.server,
            tool,
            official_client=self._official_client,
            trust_server_annotations=self._trust_server_annotations,
        )
        boundaries = (
            (_ACCOUNT_KEYS, self.account_id),
            (_ZONE_KEYS, self.zone_id),
        )
        for candidates, configured in boundaries:
            if configured is None:
                continue
            declared = _scope_keys(tool, candidates)
            scoped_keys.update(declared)
            if not read_only:
                selected = declared[0]
                required_scope_keys.append(selected)
                for key in declared[1:]:
                    bounded_properties.pop(key, None)
                field = _object_dict(bounded_properties[selected])
                field['const'] = configured
                field['default'] = configured
                bounded_properties[selected] = field
        schema['properties'] = bounded_properties
        schema['required'] = list(
            dict.fromkeys([*(key for key in required_keys if key not in scoped_keys), *required_scope_keys])
        )
        return schema

    def _pin_scope(
        self,
        args: dict[str, Any],
        tool: ToolsetTool[AgentDepsT],
        candidates: tuple[str, ...],
        configured: str,
        boundary: str,
    ) -> None:
        declared = _scope_keys(tool, candidates)
        selected = next((key for key in declared if key in args), declared[0] if declared else None)
        for key in candidates:
            if key in args and args[key] != configured:
                raise ModelRetry(f'The requested operation is outside the configured Cloudflare {boundary} boundary.')
            if key == selected:
                args[key] = configured
            else:
                args.pop(key, None)

    def _bound_page_arg(
        self, args: dict[str, Any], key: str, schema: object, *, display_name: str | None = None
    ) -> None:
        value = args.get(key)
        effective_limit = _page_limit(schema, self.max_results)
        constraints = _page_constraints(schema)
        label = display_name or key
        if effective_limit is None or constraints is None:  # pragma: no cover - filtered by `get_tools`
            raise ModelRetry(f'`{label}` does not support a safe bounded Cloudflare page size.')
        if isinstance(value, (int, float)) and value > effective_limit:
            raise ModelRetry(f'`{label}` cannot exceed the configured Cloudflare result limit of {effective_limit}.')
        if value is not None and not constraints.accepts(value):
            raise ModelRetry(f'`{label}` must be a valid integer page size no greater than {effective_limit}.')
        if value is None:
            args[key] = effective_limit

    def _scoped_args(self, tool_args: dict[str, Any], tool: ToolsetTool[AgentDepsT]) -> dict[str, Any]:
        args = dict(tool_args)
        properties = _properties(tool)
        root = tool.tool_def.parameters_json_schema
        if self.account_id is not None:
            self._pin_scope(args, tool, _ACCOUNT_KEYS, self.account_id, 'account')
        if self.zone_id is not None and not _is_api_safe_tool(self.server, tool, official_client=self._official_client):
            if not _scope_keys(tool, _ZONE_KEYS):  # pragma: no cover
                raise ModelRetry('The requested tool does not expose a Cloudflare zone boundary.')
            self._pin_scope(args, tool, _ZONE_KEYS, self.zone_id, 'zone')
        undeclared = next((key for key in _PAGE_KEYS if key in args and key not in properties), None)
        if undeclared is not None:
            raise ModelRetry(f'`{undeclared}` is not declared by the current Cloudflare tool schema.')
        for key in _PAGE_KEYS:
            if key not in properties:
                continue
            self._bound_page_arg(args, key, _resolve_schema(root, properties[key]))
        for container_key in _PAGE_CONTAINER_KEYS:
            container = _object_dict(_resolve_schema(root, properties.get(container_key)))
            if _object_schema_type(container) is not True:
                continue
            nested_properties = _object_dict(container.get('properties'))
            if container_key not in args or args[container_key] is None:
                continue
            raw_nested_args = args.get(container_key)
            if raw_nested_args is not None and not isinstance(raw_nested_args, dict):
                raise ModelRetry(f'`{container_key}` must be an object containing pagination arguments.')
            nested_args = {} if raw_nested_args is None else _OBJECT_DICT.validate_python(raw_nested_args)
            undeclared = next((key for key in _PAGE_KEYS if key in nested_args and key not in nested_properties), None)
            if undeclared is not None:
                raise ModelRetry(
                    f'`{container_key}.{undeclared}` is not declared by the current Cloudflare tool schema.'
                )
            for key in _PAGE_KEYS:
                if key in nested_properties:
                    self._bound_page_arg(
                        nested_args,
                        key,
                        _resolve_schema(root, nested_properties[key]),
                        display_name=f'{container_key}.{key}',
                    )
            args[container_key] = nested_args
        return args

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[AgentDepsT],
        tool: ToolsetTool[AgentDepsT],
    ) -> Any:
        if name != tool.tool_def.name:
            raise UserError('The requested Cloudflare tool is not available through this toolset policy.')
        authoritative_tool = (await super().get_tools(ctx)).get(name)
        if authoritative_tool is None or not self._selects_tool(authoritative_tool):
            raise UserError('The requested Cloudflare tool is not available through this toolset policy.')
        read_only = _is_read_only(
            self.server,
            authoritative_tool,
            official_client=self._official_client,
            trust_server_annotations=self._trust_server_annotations,
        )
        local_error: str | None = None
        scoped_args = tool_args
        try:
            scoped_args = self._scoped_args(tool_args, authoritative_tool)
        except ModelRetry as error:
            local_error = _bounded_error_text(
                error.message, max_bytes=self.max_output_bytes, max_lines=self.max_output_lines
            )
        if local_error is not None:
            raise ModelRetry(local_error) from None
        if scoped_args != tool_args and ctx.tool_call_approved:
            message = _bounded_error_text(
                'The current Cloudflare tool schema would change the approved provider arguments. Start a new '
                'approval request using the current tool schema.',
                max_bytes=self.max_output_bytes,
                max_lines=self.max_output_lines,
            )
            raise ToolFailed(message)
        if not read_only:
            if scoped_args != tool_args:
                message = _bounded_error_text(
                    'Repeat the mutation with the configured Cloudflare scope and result limits shown in the tool schema '
                    'so the approval request contains the exact provider arguments.',
                    max_bytes=self.max_output_bytes,
                    max_lines=self.max_output_lines,
                )
                raise ModelRetry(message)
            if not ctx.tool_call_approved:
                raise ApprovalRequired
        provider_error: str | None = None
        result: object | None = None
        try:
            use_task = bool((authoritative_tool.tool_def.metadata or {}).get('task'))
            result = await super().direct_call_tool(name, scoped_args, use_task=use_task)
        except ModelRetry as error:
            provider_error = _bounded_error_text(
                error.message, max_bytes=self.max_output_bytes, max_lines=self.max_output_lines
            )
        if provider_error is not None:
            if read_only and not ctx.tool_call_approved:
                raise ModelRetry(provider_error) from None
            raise ToolFailed(provider_error) from None
        if isinstance(result, str):
            return _bounded_text(result, max_bytes=self.max_output_bytes, max_lines=self.max_output_lines)

        serialized = to_json(result).decode('utf-8', errors='replace')
        if (
            len(serialized.encode('utf-8')) <= self.max_output_bytes
            and len(serialized.splitlines()) <= self.max_output_lines
        ):
            return result
        return _bounded_text(_TRUNCATION_MARKER, max_bytes=self.max_output_bytes, max_lines=self.max_output_lines)

    async def direct_call_tool(
        self,
        name: str,
        args: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        use_task: bool = False,
    ) -> Any:
        """Reject direct calls that cannot participate in this toolset's policy context."""
        raise UserError(
            '`CloudflareToolset.direct_call_tool()` is disabled because direct MCP calls bypass tool visibility, '
            'resource boundaries, and approval state. Execute Cloudflare tools through an `Agent`.'
        )
