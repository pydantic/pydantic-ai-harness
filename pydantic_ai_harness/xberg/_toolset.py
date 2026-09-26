"""Xberg REST API server wire contract, and the toolset that speaks it.

Wire contract, verified 2026-09-22 against Xberg 1.2.6:

- `xberg serve` listens on `http://127.0.0.1:8000`; `-H` and `-p` change host and port.
- `POST /extract` takes multipart form data: `files` (repeatable, required), `config` (a JSON
  object of extraction overrides), and `output_format` (`plain` by default, or `markdown`,
  `djot`, `html`, `json`, `doctags`). It answers `{"results": [...], "errors": [...],
  "summary": {...}}` with one `results` entry per accepted input. A result always carries
  `content` and `mime_type`, and may carry `metadata`, `tables`, `detected_languages`, `chunks`,
  and `images`; an error always carries `index`, `error_type`, and `message`, plus `source`, the
  uploaded basename.
- `results` arrives in input order with failed inputs left out, and an error's `index` is the
  input's position. Uploads carry basenames only, so pairing a result with the path it came
  from has to be positional, and every input is expected to appear in exactly one of the lists,
  each error at a distinct index.
- `POST /detect` takes the same `files` field and answers `{"mime_type", "filename"}`.
- `GET /formats` answers a list of `{"extension", "mime_type"}`.
- The `config` keys this toolset writes are `force_ocr` (top level) and `ocr.language` (nested
  under the same `ocr` object as the backend choice, which is why the merge is one level deep).
- A `tables` entry carries its rendering as `markdown`.
- Failures answer `{"error_type", "message", "status_code"}`: 400 `ValidationError`, 422
  `ParsingError` or `OcrError`, 500 for server errors.
- Uploads are capped server-side by `XBERG_MAX_REQUEST_BODY_BYTES` for the whole request and
  `XBERG_MAX_MULTIPART_FIELD_BYTES` for each file, both 104857600 bytes (100 MiB) by default.
- The server documents no authentication of its own.

Sources: <https://docs.xberg.io/guides/api-server/> for the endpoints, the limits, and the
`config` examples, <https://docs.xberg.io/reference/configuration/> for the config schema, and
the `xberg` package's own type stubs (`ExtractionResult`, `ExtractionErrorItem`,
`DetectResponse`, `SupportedFormat`, `Table`) for the field names it serializes. Re-check by
starting `xberg serve` and reading `GET /openapi.json`, which the server generates from the same
types.
"""

from __future__ import annotations

import base64
import errno
import json
import math
import os
import re
import stat
import sys
import traceback
from collections.abc import AsyncGenerator, Awaitable, Callable, Collection, Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, NoReturn, TypeVar, get_args
from urllib.parse import unquote, unquote_plus, urlsplit, urlunsplit

import anyio
import anyio.to_thread
import httpx
from opentelemetry.trace import Span, get_current_span
from pydantic import BaseModel, Field, JsonValue, RootModel, StrictInt, TypeAdapter, ValidationError, field_validator
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_core import to_json

__all__ = (
    'XBERG_DEFAULT_URL',
    'XBERG_TOOL_NAMES',
    'OutputFormat',
    'XbergDocument',
    'XbergExtraction',
    'XbergExtractionError',
    'XbergToolset',
)

XBERG_DEFAULT_URL = 'http://127.0.0.1:8000'
"""Where `xberg serve` listens unless `-H` and `-p` say otherwise."""

XBERG_TOOL_NAMES = ('extract', 'extract_batch', 'detect_mime_type', 'list_formats')
"""Every tool this capability serves, and what it registers by default."""

OutputFormat = Literal['plain', 'markdown', 'djot', 'html', 'json', 'doctags']
"""Content formats Xberg renders extracted documents as."""

DEFAULT_OUTPUT_FORMAT: OutputFormat = 'markdown'

DEFAULT_ROOT = '.'
DEFAULT_MAX_OUTPUT_BYTES = 20_000
DEFAULT_MAX_BATCH_INPUTS = 20
DEFAULT_MAX_UPLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_RESPONSE_VALUES = 1_000_000
DEFAULT_TIMEOUT = 120.0

_TRUNCATION_MARKER = '\n...[truncated]'
"""Appended to content cut to fit `max_output_bytes`, so the model can see the cut."""

_ERROR_TEXT_CHARS = 500
"""How much server or transport error text is quoted to the model."""

_ENVELOPE_BYTES = 1 << 20
"""The largest error body parsed as Xberg's envelope, a type and a message; a larger one is quoted by its head."""

_READ_CHUNK = 1 << 20
"""How much of an upload is read at a time, so allocation follows the file rather than the limit."""

_DETAILED_VALIDATION_ERRORS = 5
"""How many validation errors are quoted in full; the rest are counted, since each one quotes the body."""

_JSON_VALUE = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*(?:"|\\?\Z)|[\[{]|-?\d[\d.eE+-]*|-?[A-Za-z]+', re.DOTALL)
"""One JSON value outside string content: a string, an opening bracket, a number, or a bare word such as `null`
or the `NaN` the parser admits. An unterminated string runs to the end, so no start is tried twice."""

_FRAMING_HEADERS = frozenset({'content-length', 'content-type', 'transfer-encoding'})
"""Headers httpx computes for the multipart body; a caller's value would misframe it or defeat the upload limit."""

_RESERVED_HEADERS = _FRAMING_HEADERS | {'accept-encoding'}
"""Headers the toolset sets per request, so `headers` cannot name them."""

_GENERATED_HEADERS = frozenset(
    {('accept', '*/*'), ('connection', 'keep-alive'), ('user-agent', f'python-httpx/{httpx.__version__}')}
)
"""What httpx sends on its own, exempt from redaction only as these exact name and value pairs."""


def _written(request: httpx.Request) -> frozenset[tuple[str, str]]:
    """The header pairs exempt from redaction: `_GENERATED_HEADERS`, and the framing and encoding on `request`.

    Read before the send, since a hook writes into the same request, so a boundary or a length a hook
    wrote counts as a secret.
    """
    framing = {(name, value) for name, value in request.headers.multi_items() if name in _RESERVED_HEADERS}
    return _GENERATED_HEADERS | framing


_REDACTED = '[redacted]'
"""What stands in for a credential in any text the model reads."""

_NAMES = TypeAdapter(list[str])
"""The shape of `tools` and `redact`: a list of strings."""

_OPTIONS = TypeAdapter(dict[object, object])
"""Copies any mapping into a dict to check."""

_ITEMS = TypeAdapter(list[object])
"""Copies a list inside `config` so the mappings among its items can be walked."""

_HEADER_NAME = re.compile(r"[-!#$%&'*+.^_`|~0-9A-Za-z]+")
"""An HTTP token, the only shape of header name the transport sends."""

_HEADER_VALUE = re.compile(r'(?:[\x21-\x7e]+(?:[ \t]+[\x21-\x7e]+)*)?')
"""What the transport sends as a header value: printable ASCII with blanks only between words."""

_MULTIPART_NAMES = str.maketrans({'"': '%22', '\\': '\\\\'} | {chr(c): f'%{c:02X}' for c in range(0x20) if c != 0x1B})
"""How httpx escapes a filename in a multipart part, which is the name the server echoes."""

_LIMIT_CEILING = 2**63 - 1
"""The largest cap or deadline: what an OpenTelemetry attribute holds and a float deadline carries."""

_CLOSE_TIMEOUT = 5.0
"""Seconds a client or response gets to close under cancellation before it is abandoned."""

_UPLOAD_OPEN_FLAGS = os.O_BINARY if os.name == 'nt' else os.O_NONBLOCK | os.O_NOFOLLOW
"""`O_NOFOLLOW` refuses a symlink swapped in after the root check, and `O_NONBLOCK` keeps a swapped-in FIFO
from waiting for a writer. Windows has neither flag and gets `O_BINARY`; `_open_directly` checks the open there."""

_EXTRACT_SPAN = 'xberg_extract'
_DETECT_SPAN = 'xberg_detect'
_FORMATS_SPAN = 'xberg_formats'

_ModelT = TypeVar('_ModelT', bound=BaseModel)


class XbergDocument(BaseModel):
    """One extracted document, sized to fit the context window it is being read into."""

    source: str
    """The path the extraction was asked for."""

    mime_type: str
    """MIME type Xberg detected for the file."""

    content: str
    """Extracted text, rendered in the requested output format."""

    metadata: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    """Format-specific metadata Xberg read from the file, such as page counts or authors."""

    tables: list[str] = Field(default_factory=list[str])
    """Each table Xberg found, as markdown."""

    detected_languages: list[str] = Field(default_factory=list[str])
    """Languages detected in the content, when the server ran language detection."""

    truncated: bool = False
    """Whether `content` is a prefix of the extracted text rather than all of it."""

    omitted: list[str] = Field(default_factory=list[str])
    """Parts dropped whole because the document still did not fit, in the order they go:
    `tables`, `metadata`, `detected_languages`."""


class XbergExtractionError(BaseModel):
    """One input the server could not extract."""

    source: str
    """The path that failed, clipped to a few hundred characters like `message`."""

    error_type: str
    """Xberg's error class, such as `ParsingError` or `OcrError`."""

    message: str
    """What the server said went wrong, clipped to a few hundred characters."""


class XbergExtraction(BaseModel):
    """The outcome of one batch: the documents that came back, and the inputs that failed."""

    documents: list[XbergDocument] = Field(default_factory=list[XbergDocument])
    errors: list[XbergExtractionError] = Field(default_factory=list[XbergExtractionError])


class _ApiTable(BaseModel):
    """One table of a result."""

    markdown: str


def _finite(value: JsonValue) -> bool:
    """Whether `value` holds no NaN or infinity, walked without recursion."""
    pending: list[JsonValue] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            return False
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return True


class _ApiDocument(BaseModel):
    """One `/extract` result."""

    content: str
    mime_type: str
    metadata: dict[str, JsonValue] | None = None
    tables: list[_ApiTable] | None = None
    detected_languages: list[str] | None = None

    @field_validator('metadata')
    @classmethod
    def _finite_metadata(cls, metadata: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
        """Refuse a NaN or an infinity, which the parser admits and a tool result would write as null."""
        if metadata is not None and not _finite(metadata):
            raise ValueError('metadata holds a number that is not finite')
        return metadata


class _ApiError(BaseModel):
    """One `/extract` error, every field required and `index` strict, so a malformed one is drift."""

    index: StrictInt
    source: str
    error_type: str
    message: str


class _ApiExtraction(BaseModel):
    """The `/extract` envelope; unknown fields are ignored."""

    results: list[_ApiDocument] = Field(default_factory=list[_ApiDocument])
    errors: list[_ApiError] = Field(default_factory=list[_ApiError])


class _ApiDetection(BaseModel):
    """The `/detect` answer; `filename` is checked against the upload."""

    mime_type: str
    filename: str


class _ApiFailure(BaseModel):
    """The error envelope every failing endpoint answers with."""

    error_type: str
    message: str


class _ApiFormat(BaseModel):
    extension: str
    mime_type: str


class _ApiFormats(RootModel[list[_ApiFormat]]):
    pass


def _parsed(body: bytes, model: type[_ModelT], secrets: Sequence[str]) -> _ModelT:
    try:
        return model.model_validate_json(body)
    except ValidationError as e:
        count = e.error_count()
        detail = _clipped(str(e), secrets) if count <= _DETAILED_VALIDATION_ERRORS else f'{count} validation errors'
        raise _refused('unexpected_response', f'The Xberg API server returned an unexpected response: {detail}') from e


def validate_tools(tools: object) -> None:
    """Refuse `tools` unless it is a non-empty list of the tool names this capability serves."""
    try:
        names = _NAMES.validate_python(tools)
    except ValidationError as e:
        raise UserError(f'`tools` must be a list of Xberg tool names, not {_clip(repr(tools))}.') from e
    if not names:
        raise UserError('`tools` must name at least one Xberg tool.')
    unknown = [name for name in names if name not in XBERG_TOOL_NAMES]
    if unknown:
        raise UserError(f'Unknown Xberg tool(s) {", ".join(unknown)}. Xberg serves: {", ".join(XBERG_TOOL_NAMES)}.')


def validate_text(name: str, value: object) -> None:
    """Refuse `value` unless it is text or None."""
    if value is not None and not isinstance(value, str):
        raise UserError(f'`{name}` must be text, not {_clip(repr(value))}.')


def validate_flag(name: str, value: object) -> None:
    """Refuse `value` unless it is a boolean, since a spec's quoted `"false"` would read as on."""
    if not isinstance(value, bool):
        raise UserError(f'`{name}` must be true or false, not {_clip(repr(value))}.')


def validate_output_format(output_format: object) -> None:
    """Refuse `output_format` unless it is one of `OutputFormat`."""
    formats = get_args(OutputFormat)
    if output_format not in formats:
        raise UserError(f'`output_format` must be one of {", ".join(formats)}, not {_clip(repr(output_format))}.')


def validate_redact(redact: object) -> None:
    """Refuse `redact` unless it is a list of strings UTF-8 can encode, without quoting an entry."""
    try:
        entries = _NAMES.validate_python(redact)
    except ValidationError as e:
        raise UserError('`redact` must be a list of strings.') from e
    for entry in entries:
        try:
            entry.encode('utf-8')
        except UnicodeEncodeError as e:
            raise UserError('`redact` entries must be text UTF-8 can encode.') from e


def validate_headers(headers: object) -> None:
    """Refuse `headers` unless it maps names to values the transport sends and the toolset does not set.

    The transport's own refusal would quote the value to the model, so it is checked here first.
    """
    if headers is None:
        return
    if not isinstance(headers, Mapping):
        raise UserError(f'`headers` must map header names to text values, not {type(headers).__name__}.')
    names: list[str] = []
    for name, value in _OPTIONS.validate_python(headers).items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise UserError('`headers` must map header names to text values: a name or value is not text.')
        complaint = _header_complaint(name, value)
        if complaint is not None:
            raise UserError(f'`headers` cannot be sent: {complaint}.')
        names.append(name)
    reserved = sorted(name for name in names if name.lower() in _RESERVED_HEADERS)
    if reserved:
        raise UserError(f'`headers` cannot set {", ".join(reserved)}: the toolset sets them per request.')


def _header_complaint(name: str, value: str) -> str | None:
    """Why the transport would refuse to send `name: value`, naming the header but not the value, or None."""
    if not _HEADER_NAME.fullmatch(name):
        return f'{name!r} is not a valid header name'
    if not _HEADER_VALUE.fullmatch(value):
        return f'the value of {name!r} is not printable ASCII with blanks only between words'
    return None


def validate_url(url: object) -> None:
    """Refuse `url` unless it is an `http` or `https` URL with a host and a port a socket can use.

    httpx accepts a port out of range and fails only at the first request, with an error that is no
    retry. The message never quotes the URL, which may carry a credential.
    """
    if not isinstance(url, str):
        raise UserError(f'`url` must be text, not {_clip(repr(url))}.')
    try:
        parsed = httpx.URL(url)
    except (httpx.InvalidURL, UnicodeEncodeError) as e:
        raise UserError(f'`url` is not a URL httpx can send to: {e}') from e
    if parsed.scheme not in ('http', 'https'):
        raise UserError('`url` must start with http:// or https://.')
    if not parsed.host:
        raise UserError('`url` must name a host.')
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise UserError('`url` port must be between 1 and 65535.')


def validate_root(root: object) -> None:
    """Refuse `root` unless it is text or a path object."""
    if not isinstance(root, (str, os.PathLike)):
        raise UserError(f'`root` must be a path, not {_clip(repr(root))}.')


def validate_config(config: object) -> None:
    """Refuse `config` unless it is a mapping `json` can write with text keys at every depth.

    A YAML spec parses an unquoted date into a value that is not JSON, and an unquoted `1:` into a key
    `json` would write as `"1"` beside a text key of the same spelling.
    """
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise UserError(f'`config` must be a mapping of extraction options, not {_clip(repr(config))}.')
    options = _OPTIONS.validate_python(config)
    try:
        json.dumps(options, allow_nan=False)
    except RecursionError as e:
        raise UserError('`config` is nested too deeply to write as JSON.') from e
    except (TypeError, ValueError) as e:
        raise UserError(f'`config` must hold JSON values only: {e}.') from e
    _require_text_keys(options)


def _require_text_keys(options: Mapping[object, object]) -> None:
    """Refuse a key at any depth that is not text; the JSON check before it has ruled out cycles."""
    pending: list[Mapping[object, object]] = [options]
    while pending:
        for key, value in pending.pop().items():
            if not isinstance(key, str):
                raise UserError(f'`config` keys must be text, not {_clip(repr(key))}.')
            pending.extend(_nested_mappings(value))


def _nested_mappings(value: object) -> list[Mapping[object, object]]:
    """The mappings in `value` or in lists within it, walked without recursion."""
    found: list[Mapping[object, object]] = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, Mapping):
            found.append(_OPTIONS.validate_python(item))
        elif isinstance(item, list):
            pending.extend(_ITEMS.validate_python(item))
    return found


def _snapshot(config: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """A copy of `config` made through its JSON text, since `copy.deepcopy` overflows on deep nesting."""
    try:
        copied: dict[str, JsonValue] = json.loads(json.dumps(dict(config)))
    except RecursionError as e:  # pragma: no cover
        raise UserError('`config` is nested too deeply to read back as JSON.') from e
    return copied


def validate_client(http_client: httpx.AsyncClient | None) -> None:
    """Refuse a supplied client whose default headers frame the body or would be refused by the transport.

    httpx keeps a client's framing header over its own, so the upload limit would measure the wrong body.
    """
    if http_client is None:
        return
    framing = sorted(name for name in http_client.headers if name.lower() in _FRAMING_HEADERS)
    if framing:
        raise UserError(f'`http_client` cannot set {", ".join(framing)}: the toolset frames each request itself.')
    for raw_name, raw_value in http_client.headers.raw:
        complaint = _header_complaint(raw_name.decode('ascii', 'replace'), raw_value.decode('ascii', 'replace'))
        if complaint is not None:
            raise UserError(f'`http_client` sets a header that cannot be sent: {complaint}.')


def validate_limits(
    *,
    max_output_bytes: object,
    max_batch_inputs: object,
    max_upload_bytes: object,
    max_response_bytes: object,
    max_response_values: object,
    timeout: object,
) -> None:
    """Refuse a cap that is not a positive integer or a `timeout` that is not a positive number.

    Both are capped at `_LIMIT_CEILING` and compared as given, so NaN, infinity, and huge integers
    are refused without conversion.
    """
    caps = {
        'max_output_bytes': max_output_bytes,
        'max_batch_inputs': max_batch_inputs,
        'max_upload_bytes': max_upload_bytes,
        'max_response_bytes': max_response_bytes,
        'max_response_values': max_response_values,
    }
    for name, value in caps.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _LIMIT_CEILING:
            raise UserError(f'`{name}` must be a positive integer up to 2**63 - 1, not {_clip(repr(value))}.')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= _LIMIT_CEILING:
        raise UserError(f'`timeout` must be a positive number up to 2**63 - 1, not {_clip(repr(timeout))}.')


def _clip(text: str) -> str:
    """`text` cut to `_ERROR_TEXT_CHARS`."""
    return text if len(text) <= _ERROR_TEXT_CHARS else text[:_ERROR_TEXT_CHARS] + '...'


def _reach(secrets: Sequence[str]) -> int:
    """How far into a text redaction must read to fill the clip, with `secrets` longest first.

    Each marker in the clip may stand for a whole secret, and one more secret may begin at its end.
    """
    longest = len(secrets[0]) if secrets else 0
    return _ERROR_TEXT_CHARS + (_ERROR_TEXT_CHARS // len(_REDACTED) + 1) * longest


def _clipped(text: str, secrets: Sequence[str]) -> str:
    """`text` redacted and clipped, reading only as far as `_reach`."""
    reach = _reach(secrets)
    head = _redacted(text[:reach], secrets)
    if len(text) <= reach and len(head) <= _ERROR_TEXT_CHARS:
        return head
    return head[:_ERROR_TEXT_CHARS] + '...'


def _quoted(body: bytes, secrets: Sequence[str]) -> str:
    """The head of `body` redacted and clipped, decoding at most four bytes per character `_reach` allows."""
    decoded = _reach(secrets) * 4
    head = _redacted(body[:decoded].decode('utf-8', errors='replace'), secrets)
    if len(body) <= decoded and len(head) <= _ERROR_TEXT_CHARS:
        return head
    return head[:_ERROR_TEXT_CHARS] + '...'


def _secrets(
    base: httpx.URL,
    configured: Mapping[str, str],
    extra: Sequence[str],
    written: Collection[tuple[str, str]],
    *sent: httpx.Request,
) -> list[str]:
    """Credentials a gateway may echo, longest first, to redact from any text the model reads.

    Collected from the base URL, `redact`, `headers`, and each request in `sent`: its URL, and every
    header but the host its URL derives and the pairs in `written`. A header value also counts by
    the parts `_components` names, and each secret in the escaped form a transport error renders.
    """
    found = [*_url_secrets(base), *_extra_secrets(extra)]
    carried = list(configured.items())
    for request in sent:
        found.extend(_url_secrets(request.url))
        derived_host = request.url.netloc.decode('ascii')
        for name, value in request.headers.multi_items():
            lowered = name.lower()
            if (lowered, value) in written or (lowered == 'host' and value == derived_host):
                continue
            carried.append((name, value))
    for name, value in carried:
        found.append(value)
        found.extend(_components(name, value))
    secrets = {secret for secret in found if secret}
    escaped = {secret.encode('unicode_escape').decode('ascii') for secret in secrets}
    return sorted(secrets | escaped, key=len, reverse=True)


def _url_secrets(url: httpx.URL) -> list[str]:
    """The userinfo, password, and query of `url`, whole and by value, as written and decoded."""
    userinfo, query = url.userinfo.decode(), url.query.decode()
    found = [userinfo, unquote(userinfo), url.password, query, unquote_plus(query)]
    for pair in query.split('&'):
        value = pair.partition('=')[2]
        found.extend((value, unquote_plus(value)))
    return found


def _components(name: str, value: str) -> list[str]:
    """The parts of a header value a gateway may echo on their own.

    Authorization credentials, the pair and password a Basic credential decodes to, cookie values,
    and the boundary of a replaced `Content-Type`. A username alone is not a secret.
    """
    name = name.lower()
    if name == 'cookie':
        values = [pair.partition('=')[2].strip() for pair in value.split(';')]
        return values + [quoted.strip('"') for quoted in values]
    if name == 'content-type':
        parameters = [parameter.partition('=') for parameter in value.split(';')[1:]]
        values = [found.strip() for key, _, found in parameters if key.strip().lower() == 'boundary']
        return values + [quoted.strip('"') for quoted in values]
    if name not in ('authorization', 'proxy-authorization'):
        return []
    parts = value.split(None, 1)
    if len(parts) < 2:
        return []
    scheme, credentials = parts[0], parts[1].strip()
    if scheme.lower() != 'basic':
        return [credentials]
    try:
        decoded = base64.b64decode(credentials + '=' * (-len(credentials) % 4)).decode('utf-8', errors='replace')
    except ValueError:
        return [credentials]
    return [credentials, decoded, decoded.partition(':')[2]]


def _redacted(text: str, secrets: Sequence[str]) -> str:
    """`text` with each secret replaced wherever it appears, longest first, in one pass.

    Matches need no word boundary, since a gateway may glue a credential to other text, so a short
    secret may also cut an ordinary word.
    """
    if not secrets:
        return text
    return re.sub('|'.join(re.escape(secret) for secret in secrets), _REDACTED, text)


def _extra_secrets(entries: Sequence[str]) -> list[str]:
    """Each `redact` entry whole, by its part after a colon, and as the Basic credential it encodes to.

    For a URL those parts come from its userinfo, which is how httpx sends a proxy's credentials.
    """
    found: list[str] = []
    for entry in entries:
        try:
            url = httpx.URL(entry)
        except httpx.InvalidURL:
            url = None
        if url is not None and url.userinfo:
            found.extend(_url_secrets(url))
            userinfo = unquote(url.userinfo.decode())
        else:
            userinfo = entry
        found.extend((entry, userinfo, userinfo.partition(':')[2], base64.b64encode(userinfo.encode()).decode()))
    return found


class _RecordingAuth(httpx.Auth):
    """`auth` that keeps a copy of each request its flow yields.

    A flow may yield the same request twice with a different credential, so only a copy taken at
    each yield still holds the first.
    """

    def __init__(self, auth: httpx.Auth) -> None:
        self.auth = auth
        self.sent: list[httpx.Request] = []

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        flow = self.auth.async_auth_flow(request)
        try:
            yielded = await flow.__anext__()
            while True:
                self.sent.append(httpx.Request(yielded.method, yielded.url, headers=yielded.headers))
                yielded = await flow.asend((yield yielded))  # codespell:ignore asend
        except StopAsyncIteration:
            return
        finally:
            await flow.aclose()


def _failed_request(error: httpx.HTTPError, sent: httpx.Request) -> httpx.Request:
    """The request `error` names, or `sent` when a hook raised it without one."""
    try:
        return error.request
    except RuntimeError:
        return sent


def _record_exception(span: Span, error: BaseException) -> None:
    """`Span.record_exception` without the chained cause, which may quote a credential."""
    module, qualname = type(error).__module__, type(error).__qualname__
    stacktrace = ''.join(traceback.format_exception(type(error), error, error.__traceback__, chain=False))
    span.add_event(
        'exception',
        {
            'exception.type': f'{module}.{qualname}' if module != 'builtins' else qualname,
            'exception.message': str(error),
            'exception.stacktrace': stacktrace,
            'exception.escaped': 'False',
        },
    )


def _sent_as(uploaded: str) -> tuple[str, str]:
    """The names the server may echo for an upload: as given, and as httpx escapes it in the part."""
    return uploaded, uploaded.translate(_MULTIPART_NAMES)


def _shown(path: str) -> str:
    """`path` quoted and clipped for a message the model reads."""
    return repr(_clip(path))


def _refused(
    reason: str,
    message: str,
    *,
    limit: int | None = None,
    measured: int | None = None,
    status: int | None = None,
) -> ModelRetry:
    """A retry for the model, recording `reason`, and any limit and measurement, on the current span.

    The message names paths and quotes the server, so it stays off the span; `reason` is one of a
    fixed few.
    """
    span = get_current_span()
    if span.is_recording():
        span.set_attribute('xberg.refusal', reason)
        if limit is not None:
            span.set_attribute('xberg.limit', limit)
        if measured is not None:
            span.set_attribute('xberg.measured', measured)
        if status is not None:
            span.set_attribute('xberg.status_code', status)
    return ModelRetry(message)


def _json_size(value: XbergDocument | Sequence[str] | Mapping[str, JsonValue]) -> int:
    return len(to_json(value))


def _truncate_content(document: XbergDocument, max_output_bytes: int) -> tuple[XbergDocument, bool]:
    """Cut `content` to the longest prefix that fits, and say whether the document fits.

    The prefix is found by binary search over serialized size, since a character costs one to six
    bytes. The marker and `truncated` appear only when characters were removed, and nothing is
    searched unless the marker alone fits.
    """
    if len(document.content) <= max_output_bytes and _json_size(document) <= max_output_bytes:
        return document, True
    if not document.content:
        return document, False

    def cut(length: int) -> XbergDocument:
        return document.model_copy(
            update={'content': document.content[:length] + _TRUNCATION_MARKER, 'truncated': True}
        )

    empty = cut(0)
    if _json_size(empty) > max_output_bytes:
        return empty, False
    low, high = 0, min(len(document.content) - 1, max_output_bytes)
    while low < high:
        middle = (low + high + 1) // 2
        if _json_size(cut(middle)) <= max_output_bytes:
            low = middle
        else:
            high = middle - 1
    return cut(low), True


_Part = Literal['tables', 'metadata', 'detected_languages']

_SHED_ORDER: tuple[_Part, ...] = ('tables', 'metadata', 'detected_languages')
"""The parts dropped whole, in order, when no cut of the content fits."""


def _part(document: XbergDocument, part: _Part) -> list[str] | dict[str, JsonValue]:
    if part == 'metadata':
        return document.metadata
    return document.tables if part == 'tables' else document.detected_languages


def _shed(document: XbergDocument, part: _Part) -> XbergDocument:
    """`document` with `part` emptied and named in `omitted`."""
    empty: list[str] | dict[str, JsonValue] = {} if part == 'metadata' else []
    return document.model_copy(update={part: empty, 'omitted': [*document.omitted, part]})


def _fit_document(document: XbergDocument, max_output_bytes: int) -> XbergDocument:
    """Fit `document` into `max_output_bytes`, refusing a cap too small for its path and MIME type.

    A part that alone exceeds the cap goes first. Content is then cut, and cut again after each of
    tables, metadata, and languages is dropped; the marker goes last, since `truncated` says the same.
    """
    for part in _SHED_ORDER:
        if _part(document, part) and _json_size(_part(document, part)) > max_output_bytes:
            document = _shed(document, part)
    fitted, fits = _truncate_content(document, max_output_bytes)
    for part in _SHED_ORDER:
        if fits:
            return fitted
        if _part(document, part):
            document = _shed(document, part)
            fitted, fits = _truncate_content(document, max_output_bytes)
    if fits:
        return fitted
    if fitted.truncated and fitted.content == _TRUNCATION_MARKER:
        fitted = fitted.model_copy(update={'content': ''})
    size = _json_size(fitted)
    if size > max_output_bytes:
        raise _refused(
            'output_limit',
            f'{_shown(document.source)} cannot be returned within max_output_bytes: without its content, tables, '
            f'metadata, and languages it still serializes to {size} bytes, over the {max_output_bytes}-byte cap.',
            limit=max_output_bytes,
            measured=size,
        )
    return fitted


def _merged_config(
    base: Mapping[str, JsonValue] | None,
    *,
    force_ocr: bool | None,
    ocr_language: str | None,
) -> dict[str, JsonValue]:
    """The standing config with the per-call OCR arguments on top, `ocr` merged one level deep."""
    config: dict[str, JsonValue] = dict(base or {})
    if force_ocr is not None:
        config['force_ocr'] = force_ocr
    if ocr_language is not None:
        configured = config.get('ocr')
        ocr: dict[str, JsonValue] = dict(configured) if isinstance(configured, dict) else {}
        ocr['language'] = ocr_language
        config['ocr'] = ocr
    return config


def _failed_inputs(errors: Sequence[_ApiError], inputs: int) -> dict[int, _ApiError] | None:
    """Each error by the input it names, or None when an index is repeated or out of range."""
    failed: dict[int, _ApiError] = {}
    for error in errors:
        if not 0 <= error.index < inputs or error.index in failed:
            return None
        failed[error.index] = error
    return failed


def display_url(url: str, *, with_path: bool) -> str:
    """`url` without userinfo, query, or fragment, and without its path unless `with_path`.

    A gateway may authenticate through any of these, or route by tenant through the path.
    """
    parts = urlsplit(url)
    path = parts.path.rstrip('/') if with_path else ''
    return urlunsplit((parts.scheme, parts.netloc.rpartition('@')[2], path, '', ''))


def _resolve(root: Path, path: str) -> Path:
    """Resolve `path` inside `root`, following symlinks before the check.

    A name UTF-8 cannot encode is refused first, since httpx would raise while building the upload.
    """
    try:
        path.encode('utf-8')
    except UnicodeEncodeError as e:
        raise _refused(
            'invalid_path', f'Path {_shown(path)} is not valid UTF-8, which the upload cannot carry: {e}.'
        ) from e
    try:
        resolved = Path(os.path.realpath(root / path))
    except ValueError as e:
        raise _refused('invalid_path', f'Path {_shown(path)} is not a valid path: {e}.') from e
    if not resolved.is_relative_to(root):
        raise _refused('outside_root', f'Path {_shown(path)} resolves outside the extraction root.')
    return resolved


def _open_walking(resolved: Path) -> int:
    """Open `resolved` one component at a time from the filesystem root, following no symlink.

    A directory swapped for a symlink after the root check then fails the open instead of redirecting it.
    """
    anchor, *components = resolved.parts
    directory = os.open(anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in components[:-1]:
            entered = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = entered
        return os.open(components[-1] if components else '.', os.O_RDONLY | _UPLOAD_OPEN_FLAGS, dir_fd=directory)
    finally:
        os.close(directory)


def _open_directly(resolved: Path) -> int:
    """Open `resolved` where `os.open` takes no `dir_fd` (Windows), refusing it if the open landed elsewhere.

    A symlink or junction swapped in after the root check would redirect the open, so the path the
    platform reports for the open file has to still be the one checked.
    """
    descriptor = os.open(resolved, os.O_RDONLY | _UPLOAD_OPEN_FLAGS)
    try:
        if _comparable(_final_path(descriptor)) != _comparable(str(resolved)):
            raise OSError(errno.ELOOP, 'The file moved after it was checked', str(resolved))
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _comparable(path: str) -> str:
    """`path` case-folded as the platform compares paths, without the extended-length prefix Windows may add."""
    plain = '\\\\' + path[8:] if path.startswith('\\\\?\\UNC\\') else path.removeprefix('\\\\?\\')
    return os.path.normcase(plain)


def _final_path(descriptor: int) -> str:
    """Where `descriptor` was opened, every link followed: from `/proc` on Linux, from the handle on Windows."""
    if sys.platform == 'linux':
        return os.readlink(f'/proc/self/fd/{descriptor}')
    return _handle_final_path(descriptor)  # pragma: no cover


def _handle_final_path(descriptor: int) -> str:  # pragma: no cover
    """`GetFinalPathNameByHandleW` for `descriptor`; a platform with no such report refuses the read."""
    if sys.platform != 'win32':
        raise OSError(errno.ENOTSUP, 'This platform cannot report where an opened file lies')
    import ctypes
    import msvcrt
    from ctypes import wintypes

    get_final_path = ctypes.WinDLL('kernel32', use_last_error=True).GetFinalPathNameByHandleW
    get_final_path.argtypes = (wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD)
    get_final_path.restype = wintypes.DWORD
    handle = msvcrt.get_osfhandle(descriptor)
    size = get_final_path(handle, None, 0, 0)
    buffer = ctypes.create_unicode_buffer(size)
    written = get_final_path(handle, buffer, size, 0) if size else 0
    if not written or written >= size:
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value


def _read_upload(root: Path, path: str, budget: int, max_upload_bytes: int) -> tuple[str, bytes]:
    """Read one upload through a single descriptor, within `budget`, what is left of the upload limit.

    Checking the descriptor rather than the path keeps a file swapped after the root check from
    redirecting the read. The upload keeps the requested name, whose extension the server reads.
    """
    resolved = _resolve(root, path)
    descriptor = -1
    try:
        descriptor = _open_walking(resolved) if os.open in os.supports_dir_fd else _open_directly(resolved)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise _refused('unreadable_path', f'Path {_shown(path)} is not a regular file.')
        if info.st_size > max_upload_bytes:
            raise _refused(
                'upload_limit',
                f'{_shown(path)} is {info.st_size} bytes, over the {max_upload_bytes}-byte upload limit.',
                limit=max_upload_bytes,
                measured=info.st_size,
            )
        if info.st_size > budget:
            raise _refused(
                'upload_limit',
                f'{_shown(path)} takes this request past the {max_upload_bytes}-byte upload limit. Split the batch.',
                limit=max_upload_bytes,
                measured=max_upload_bytes - budget + info.st_size,
            )
        limit = min(max_upload_bytes, budget)
        data = bytearray()
        with os.fdopen(descriptor, 'rb') as source:
            descriptor = -1
            while len(data) <= limit:
                chunk = source.read(min(_READ_CHUNK, limit + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
        if len(data) > limit:
            raise _refused(
                'upload_limit',
                f'{_shown(path)} reads past the {max_upload_bytes}-byte upload limit'
                + ('. Split the batch.' if budget < max_upload_bytes else '.'),
                limit=max_upload_bytes,
                measured=max_upload_bytes - budget + len(data),
            )
        return Path(path).name, bytes(data)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise _refused(
                'unreadable_path', f'Path {_shown(path)} is a symlink loop, or became a symlink after it was checked.'
            ) from e
        raise _refused('unreadable_path', f'Could not read {_shown(path)}: {e.strerror or e}.') from e
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _declared_length(declared: str) -> int | None:
    """`declared` as a length, or None when it is not a number Python converts."""
    if not (declared.isascii() and declared.isdigit()):
        return None
    try:
        return int(declared)
    except ValueError:
        return None


def _over_response_limit(max_response_bytes: int, measured: int) -> ModelRetry:
    return _refused(
        'response_limit',
        f'The Xberg API server answered with a body over the {max_response_bytes}-byte response limit. '
        'Extract fewer files in one call.',
        limit=max_response_bytes,
        measured=measured,
    )


def _values_over(body: bytes, max_response_values: int) -> int | None:
    """The count past `max_response_values` at which scanning `body` stopped, or None.

    Each value is at least a byte, so a body no longer than the limit is not scanned, and a longer one
    only until the limit is passed.
    """
    if len(body) <= max_response_values:
        return None
    for measured, _ in enumerate(_JSON_VALUE.finditer(body), start=1):
        if measured > max_response_values:
            return measured
    return None


def _check_values(body: bytes, max_response_values: int) -> None:
    """Refuse `body` before it is parsed when it holds more JSON values than `max_response_values`."""
    measured = _values_over(body, max_response_values)
    if measured is not None:
        raise _refused(
            'value_limit',
            f'The Xberg API server answered with more than {max_response_values} JSON values, over the '
            'response value limit. Extract fewer files in one call.',
            limit=max_response_values,
            measured=measured,
        )


async def _read_body(response: httpx.Response, max_response_bytes: int, secrets: Sequence[str]) -> bytes:
    """Read a response body as sent, refusing one over `max_response_bytes` before it is held whole.

    An encoded body is refused rather than decoded, since a decoded chunk could pass the cap before it
    is measured. A header value a refusal quotes is redacted first.
    """
    encoding = response.headers.get('content-encoding', 'identity').strip()
    if encoding.lower() != 'identity':
        raise _refused(
            'encoded_response',
            f'The Xberg API server answered with {_clipped(encoding, secrets)!r} content encoding, which this '
            'capability does not decode: it asks for an unencoded body.',
        )
    declared = response.headers.get('content-length')
    if declared is not None:
        length = _declared_length(declared)
        if length is None:
            raise _refused(
                'unexpected_response',
                'The Xberg API server returned an unexpected response: Content-Length '
                f'{_clipped(declared, secrets)!r} is not a length.',
            )
        if length > max_response_bytes:
            raise _over_response_limit(max_response_bytes, length)
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > max_response_bytes:
            raise _over_response_limit(max_response_bytes, len(body) + len(chunk))
        body.extend(chunk)
    return bytes(body)


def _failure(body: bytes, max_response_values: int) -> _ApiFailure | None:
    """Xberg's error envelope, or None for a body that is not one or is too large to parse as one."""
    if len(body) > _ENVELOPE_BYTES or _values_over(body, max_response_values) is not None:
        return None
    try:
        return _ApiFailure.model_validate_json(body)
    except ValidationError:
        return None


def _raise_for_error(status_code: int, body: bytes, secrets: Sequence[str], max_response_values: int) -> NoReturn:
    """Turn an error response into a redacted retry, quoting the body when it is not an envelope."""
    failure = _failure(body, max_response_values)
    if failure is None:
        raise _refused(
            'server_error',
            f'The Xberg API server answered {status_code}: {_quoted(body, secrets)}',
            status=status_code,
        )
    raise _refused(
        'server_error',
        f'Xberg {_clipped(failure.error_type, secrets)} ({status_code}): {_clipped(failure.message, secrets)}',
        status=status_code,
    )


class XbergToolset(FunctionToolset[AgentDepsT]):
    """Xberg's document tools, served by a running `xberg serve` REST API server.

    Each tool uploads the file it was given and returns what a model can use, capped at
    `max_output_bytes` per document. Every tool is sequential, so an agent holds one call's upload
    and answer at a time.

    Use the `Xberg` capability for usage instructions, and this class directly for toolset
    combinators such as `approval_required()`.
    """

    def __init__(
        self,
        *,
        url: str = XBERG_DEFAULT_URL,
        output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT,
        config: Mapping[str, JsonValue] | None = None,
        root: str | Path = DEFAULT_ROOT,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        max_batch_inputs: int = DEFAULT_MAX_BATCH_INPUTS,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_response_values: int = DEFAULT_MAX_RESPONSE_VALUES,
        timeout: float = DEFAULT_TIMEOUT,
        headers: Mapping[str, str] | None = None,
        redact: Sequence[str] = (),
        http_client: httpx.AsyncClient | None = None,
        tools: Sequence[str] = XBERG_TOOL_NAMES,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, sequential=True)
        validate_tools(tools)
        validate_url(url)
        validate_root(root)
        validate_output_format(output_format)
        validate_headers(headers)
        validate_redact(redact)
        validate_config(config)
        validate_client(http_client)
        validate_limits(
            max_output_bytes=max_output_bytes,
            max_batch_inputs=max_batch_inputs,
            max_upload_bytes=max_upload_bytes,
            max_response_bytes=max_response_bytes,
            max_response_values=max_response_values,
            timeout=timeout,
        )
        self._base = httpx.URL(url)
        self._display_url = display_url(url, with_path=True)
        self._origin = display_url(url, with_path=False)
        self._output_format = output_format
        self._config = _snapshot(config) if config is not None else None
        self._root = Path(os.path.realpath(root))
        self._max_output_bytes = max_output_bytes
        self._max_batch_inputs = max_batch_inputs
        self._max_upload_bytes = max_upload_bytes
        self._max_response_bytes = max_response_bytes
        self._max_response_values = max_response_values
        self._timeout = timeout
        self._headers = dict(headers or {})
        self._redact = tuple(redact)
        self._http_client = http_client
        served: dict[str, Callable[..., Awaitable[object]]] = {
            'extract': self.extract,
            'extract_batch': self.extract_batch,
            'detect_mime_type': self.detect_mime_type,
            'list_formats': self.list_formats,
        }
        for name in XBERG_TOOL_NAMES:
            if name in tools:
                self.add_function(served[name], name=name)

    async def extract(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        output_format: OutputFormat | None = None,
        force_ocr: bool | None = None,
        ocr_language: str | None = None,
    ) -> XbergDocument:
        """Extract the text, tables, and metadata of one document.

        Reads PDFs, Office and OpenDocument files, HTML, email, images, audio, archives, and
        source files. Pages with no text layer go through OCR when the server has it enabled.

        Args:
            ctx: The run context (supplied by the agent).
            path: File to extract, inside the extraction root.
            output_format: How to render the content: `plain`, `markdown`, `djot`, `html`,
                `json`, or `doctags`. Defaults to the capability's own setting.
            force_ocr: `true` runs OCR even when the file already has extractable text; `false`
                switches configured forced OCR off for this file.
            ocr_language: OCR language code for this file, such as `eng`.

        Returns:
            The document's content, its tables as markdown, and its metadata. Long content is
            cut to fit: check `truncated` and `omitted` before treating it as the whole file.
        """
        with self._traced(ctx, _EXTRACT_SPAN) as span:
            extraction = await self._extract(
                span,
                ctx,
                [path],
                output_format=output_format,
                force_ocr=force_ocr,
                ocr_language=ocr_language,
            )
            if extraction.errors:
                failure = extraction.errors[0]
                raise _refused(
                    'extraction_error',
                    f'Xberg could not extract {_shown(path)} ({failure.error_type}): {failure.message}',
                )
            return extraction.documents[0]

    async def extract_batch(
        self,
        ctx: RunContext[AgentDepsT],
        paths: Sequence[str],
        output_format: OutputFormat | None = None,
        force_ocr: bool | None = None,
        ocr_language: str | None = None,
    ) -> XbergExtraction:
        """Extract several documents in one request.

        Prefer this over repeated `extract` calls: the server processes the batch with its own
        concurrency, and one tool result costs less than several.

        Args:
            ctx: The run context (supplied by the agent).
            paths: Files to extract, inside the extraction root.
            output_format: How to render the content: `plain`, `markdown`, `djot`, `html`,
                `json`, or `doctags`. Defaults to the capability's own setting.
            force_ocr: `true` runs OCR even when a file already has extractable text; `false`
                switches configured forced OCR off for these files.
            ocr_language: OCR language code for these files, such as `eng`.

        Returns:
            The documents that came back and the inputs that failed, side by side. A failed
            input does not stop the others.
        """
        with self._traced(ctx, _EXTRACT_SPAN) as span:
            return await self._extract(
                span,
                ctx,
                paths,
                output_format=output_format,
                force_ocr=force_ocr,
                ocr_language=ocr_language,
            )

    async def detect_mime_type(self, ctx: RunContext[AgentDepsT], path: str) -> str:
        """Identify a file's MIME type without extracting it.

        Cheap next to `extract`: use it to check what a file is before deciding how to read it.

        Args:
            ctx: The run context (supplied by the agent).
            path: File to identify, inside the extraction root.

        Returns:
            The MIME type Xberg detected, such as `application/pdf`.
        """
        with self._traced(ctx, _DETECT_SPAN) as span:
            if span.is_recording() and ctx.trace_include_content:
                span.set_attribute('xberg.sources', [path])
            uploads = await self._uploads([path])
            body, secrets = await self._send('POST', '/detect', files=uploads)
            detection = _parsed(body, _ApiDetection, secrets)
            uploaded = uploads[0][1][0]
            if detection.filename not in _sent_as(uploaded):
                raise _refused(
                    'unexpected_response',
                    'The Xberg API server returned an unexpected response: it identified '
                    f'{_clipped(detection.filename, secrets)!r}, not the uploaded {_shown(uploaded)}.',
                )
            self._capped(span, detection.mime_type, 'a MIME type')
            return detection.mime_type

    async def list_formats(self, ctx: RunContext[AgentDepsT]) -> dict[str, str]:
        """List the file formats this Xberg server can read.

        Args:
            ctx: The run context (supplied by the agent).

        Returns:
            Each supported extension mapped to the MIME type Xberg reports for it.
        """
        with self._traced(ctx, _FORMATS_SPAN) as span:
            body, secrets = await self._send('GET', '/formats')
            formats = {entry.extension: entry.mime_type for entry in _parsed(body, _ApiFormats, secrets).root}
            self._capped(span, formats, 'a format list')
            return formats

    @contextmanager
    def _traced(self, ctx: RunContext[AgentDepsT], name: str) -> Generator[Span, None, None]:
        """Run one tool call under its span, recording the server, the cap, and a failure's type.

        The server's path prefix and the exception join only when the run traces content, since either
        may name a tenant, a path, or a credential. A cancelled call names its cancellation too.
        """
        with ctx.tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
            if span.is_recording():
                span.set_attribute('xberg.url', self._display_url if ctx.trace_include_content else self._origin)
                span.set_attribute('xberg.max_output_bytes', self._max_output_bytes)
            try:
                yield span
            except anyio.get_cancelled_exc_class() as e:
                if span.is_recording():
                    span.set_attribute('xberg.exception_type', type(e).__name__)
                raise
            except Exception as e:
                if span.is_recording():
                    span.set_attribute('xberg.exception_type', type(e).__name__)
                    if ctx.trace_include_content:
                        _record_exception(span, e)
                raise

    def _capped(self, span: Span, value: str | dict[str, str], what: str) -> None:
        """Refuse a tool result over `max_output_bytes`, recording its size beside the cap on the span."""
        size = len(to_json(value))
        if span.is_recording():
            span.set_attribute('xberg.result_bytes', size)
        if size > self._max_output_bytes:
            raise _refused(
                'output_limit',
                f'The Xberg API server answered with {what} of {size} bytes, '
                f'over the {self._max_output_bytes}-byte cap.',
                limit=self._max_output_bytes,
                measured=size,
            )

    def _endpoint(self, path: str) -> httpx.URL:
        """`path` under the configured base URL, keeping its encoded path and query string as written."""
        base_path, separator, query = self._base.raw_path.partition(b'?')
        return self._base.copy_with(raw_path=base_path.rstrip(b'/') + path.encode('ascii') + separator + query)

    async def _extract(
        self,
        span: Span,
        ctx: RunContext[AgentDepsT],
        paths: Sequence[str],
        *,
        output_format: OutputFormat | None,
        force_ocr: bool | None,
        ocr_language: str | None,
    ) -> XbergExtraction:
        """Upload `paths` and project the answer, under the span the calling tool opened."""
        rendered = output_format if output_format is not None else self._output_format
        config = _merged_config(self._config, force_ocr=force_ocr, ocr_language=ocr_language)
        if span.is_recording():
            span.set_attribute('xberg.output_format', rendered)
            span.set_attribute('xberg.inputs', len(paths))
            if ctx.trace_include_content:
                span.set_attribute('xberg.sources', list(paths))
        if not paths:
            raise _refused(
                'batch_limit', 'Name at least one path to extract.', limit=self._max_batch_inputs, measured=0
            )
        if len(paths) > self._max_batch_inputs:
            raise _refused(
                'batch_limit',
                f'{len(paths)} paths is more than this server accepts in one batch '
                f'({self._max_batch_inputs}). Split the batch.',
                limit=self._max_batch_inputs,
                measured=len(paths),
            )
        files = await self._uploads(paths)
        form = {'output_format': rendered}
        if config:
            form['config'] = json.dumps(config)
        body, secrets = await self._send('POST', '/extract', files=files, data=form)
        extraction = self._projected(_parsed(body, _ApiExtraction, secrets), paths, secrets)
        if span.is_recording():
            span.set_attribute('xberg.documents', len(extraction.documents))
            span.set_attribute('xberg.errors', len(extraction.errors))
            span.set_attribute(
                'xberg.truncated_documents', sum(1 for document in extraction.documents if document.truncated)
            )
        return extraction

    def _projected(self, payload: _ApiExtraction, paths: Sequence[str], secrets: Sequence[str]) -> XbergExtraction:
        """Pair the server's outcomes with the paths they came from, refusing an answer that does not pair up.

        Results arrive in input order without the failed inputs, and each error names its input by index
        and basename, so the counts must add up and each error's index and name must agree.
        """
        failed = _failed_inputs(payload.errors, len(paths))
        if failed is None or len(payload.results) + len(failed) != len(paths):
            uploaded = f'{len(paths)} uploaded file' if len(paths) == 1 else f'{len(paths)} uploaded files'
            raise _refused(
                'unexpected_response',
                f'The Xberg API server returned an unexpected response: {len(payload.results)} results and '
                f'{len(payload.errors)} errors that cannot be paired with the {uploaded}.',
            )
        for index, error in failed.items():
            uploaded = Path(paths[index]).name
            if error.source not in _sent_as(uploaded):
                raise _refused(
                    'unexpected_response',
                    f'The Xberg API server returned an unexpected response: the error for input {index} names '
                    f'{_clipped(error.source, secrets)!r}, not the uploaded {_shown(uploaded)}.',
                )
        surviving = [path for index, path in enumerate(paths) if index not in failed]
        documents = [
            _fit_document(
                XbergDocument(
                    source=source,
                    mime_type=result.mime_type,
                    content=result.content,
                    metadata=result.metadata or {},
                    tables=[table.markdown for table in result.tables or [] if table.markdown],
                    detected_languages=result.detected_languages or [],
                ),
                self._max_output_bytes,
            )
            for source, result in zip(surviving, payload.results)
        ]
        errors = [
            XbergExtractionError(
                source=_clip(paths[index]),
                error_type=_clipped(error.error_type, secrets),
                message=_clipped(error.message, secrets),
            )
            for index, error in failed.items()
        ]
        return XbergExtraction(documents=documents, errors=errors)

    async def _uploads(self, paths: Sequence[str]) -> list[tuple[str, tuple[str, bytes]]]:
        """Read every file of one request within `max_upload_bytes` for the request as a whole.

        Each read runs in a worker thread a cancel abandons, so a stalled filesystem cannot hold the run.
        """
        files: list[tuple[str, tuple[str, bytes]]] = []
        budget = self._max_upload_bytes
        for path in paths:
            upload = await anyio.to_thread.run_sync(
                _read_upload, self._root, path, budget, self._max_upload_bytes, abandon_on_cancel=True
            )
            budget -= len(upload[1])
            files.append(('files', upload))
        return files

    async def _send(
        self,
        method: str,
        path: str,
        *,
        files: Sequence[tuple[str, tuple[str, bytes]]] | None = None,
        data: Mapping[str, str] | None = None,
    ) -> tuple[bytes, list[str]]:
        """Send one request through the supplied client, or through one opened for the call.

        A client per call keeps the toolset stateless, so it composes with durability wrappers. It ignores
        proxy variables, so a local file cannot leave through a proxy nobody configured, and `timeout`
        bounds the whole request. A supplied client's headers are checked again, since its owner may
        change them.
        """
        if self._http_client is not None:
            validate_client(self._http_client)
            return await self._request(self._http_client, method, path, files=files, data=data)
        client = httpx.AsyncClient(timeout=self._timeout, trust_env=False)
        try:
            with anyio.fail_after(self._timeout):
                return await self._request(client, method, path, files=files, data=data)
        except TimeoutError as e:
            raise _refused(
                'timeout',
                f'The Xberg API server did not finish answering within {self._timeout} seconds. '
                'Extract fewer files in one call.',
            ) from e
        finally:
            with anyio.CancelScope(shield=True), anyio.move_on_after(_CLOSE_TIMEOUT):
                await client.aclose()

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        files: Sequence[tuple[str, tuple[str, bytes]]] | None,
        data: Mapping[str, str] | None,
    ) -> tuple[bytes, list[str]]:
        """Send the request httpx builds, once its framed size fits the upload limit.

        Returns the body with the secrets of every request that went out: the one built, each an `auth`
        flow yielded, each redirect hop, and the one answered, since each may carry a credential.
        """
        headers = httpx.Headers(self._headers)
        headers['accept-encoding'] = 'identity'
        request = client.build_request(method, self._endpoint(path), files=files, data=data, headers=headers)
        written = _written(request)
        recording = _RecordingAuth(client.auth) if client.auth is not None else None
        yielded: list[httpx.Request] = [] if recording is None else recording.sent
        framed = int(request.headers.get('content-length', '0'))
        if framed > self._max_upload_bytes:
            raise _refused(
                'upload_limit',
                f'This request is {framed} bytes with its multipart framing, over the {self._max_upload_bytes}-byte '
                'upload limit. Extract fewer files in one call, or a smaller file.',
                limit=self._max_upload_bytes,
                measured=framed,
            )
        try:
            response = await client.send(
                request, stream=True, auth=httpx.USE_CLIENT_DEFAULT if recording is None else recording
            )
            hops = [hop.request for hop in response.history]
            secrets = _secrets(
                self._base, self._headers, self._redact, written, request, *yielded, *hops, response.request
            )
            try:
                body = await _read_body(response, self._max_response_bytes, secrets)
            finally:
                with anyio.CancelScope(shield=True), anyio.move_on_after(_CLOSE_TIMEOUT):
                    await response.aclose()
        except httpx.TimeoutException as e:
            raise _refused(
                'timeout',
                f'The Xberg API server did not finish answering within the client timeout ({type(e).__name__}). '
                'Extract fewer files in one call.',
            ) from e
        except httpx.HTTPError as e:
            secrets = _secrets(
                self._base, self._headers, self._redact, written, request, *yielded, _failed_request(e, request)
            )
            raise _refused(
                'unreachable',
                f'Could not reach the Xberg API server at {_clip(self._display_url)}: {_clipped(str(e), secrets)}. '
                'Start one with `xberg serve` (or point the capability at a running server).',
            ) from e
        if response.is_error:
            _raise_for_error(response.status_code, body, secrets, self._max_response_values)
        _check_values(body, self._max_response_values)
        return body, secrets
