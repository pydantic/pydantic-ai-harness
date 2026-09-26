"""Xberg capability: document extraction for an agent, from a Xberg API server."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, fields
from pathlib import Path

import httpx
from pydantic import JsonValue
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.xberg._toolset import (
    DEFAULT_MAX_BATCH_INPUTS,
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_MAX_RESPONSE_VALUES,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_ROOT,
    DEFAULT_TIMEOUT,
    XBERG_DEFAULT_URL,
    XBERG_TOOL_NAMES,
    OutputFormat,
    XbergToolset,
    display_url,
    validate_client,
    validate_config,
    validate_flag,
    validate_headers,
    validate_limits,
    validate_output_format,
    validate_redact,
    validate_root,
    validate_text,
    validate_tools,
    validate_url,
)

_DEFAULT_DESCRIPTION = 'Read documents: PDFs, Office files, images, audio, archives, and source files.'

_INTRODUCTION = (
    'Xberg extracts text, tables, and metadata from documents: PDFs, Office and OpenDocument '
    'files, HTML, email, images, audio, archives, and source files. Use it for any file you '
    'cannot read as text, and for scanned pages, which it puts through OCR.'
)

_TOOL_GUIDANCE = {
    'extract': 'Call `extract` with one path to read a document.',
    'extract_batch': (
        'Call `extract_batch` with several paths to read them in one call, which costs less than one call per file.'
    ),
    'detect_mime_type': '`detect_mime_type` identifies a file without extracting it.',
    'list_formats': '`list_formats` says which formats the server can read.',
}

_TRUNCATION_GUIDANCE = (
    'A document whose content did not fit comes back with `truncated` set, and `omitted` names '
    'any part dropped whole, so check both before treating content as the whole file.'
)


@dataclass
class Xberg(AbstractCapability[AgentDepsT]):
    """Document extraction through a Xberg API server.

    Gives an agent tools that turn a file into text, tables, and metadata: PDFs, Office and
    OpenDocument files, HTML, email, images (with OCR), audio (with transcription), archives,
    and source files.

    ```python
    from pydantic_ai import Agent

    from pydantic_ai_harness import Xberg

    agent = Agent('anthropic:claude-fable-5', capabilities=[Xberg()])
    ```

    Extraction runs in a server started with `xberg serve`, so the tools upload each file they read.
    """

    _: KW_ONLY

    description: str | None = _DEFAULT_DESCRIPTION
    """Routing description used when the capability is loaded on demand."""

    url: str = XBERG_DEFAULT_URL
    """Base URL of the API server, with any path prefix and query string a gateway needs.

    Credentials in it go out with every request and are kept out of spans, the repr, and the
    messages the model reads. The server authenticates nothing itself; see the README before
    using it on a shared host.
    """

    output_format: OutputFormat = DEFAULT_OUTPUT_FORMAT
    """How content is rendered when a call does not ask otherwise.

    Markdown rather than Xberg's `plain` default, since a model can use its structure.
    """

    config: Mapping[str, JsonValue] | None = None
    """Extraction config sent with every request, such as `{'chunking': {'max_characters': 2000}}`.

    Per-call arguments take precedence, and `ocr` merges one level deep, so a call that names an OCR
    language keeps the backend configured here. The toolset keeps its own copy.
    """

    root: str | Path = DEFAULT_ROOT
    """Directory the tools may read files from, the working directory by default.

    Paths are resolved through symlinks before the check and opened without following them, so a
    swap after the check is refused. On Windows, which opens by path, the open file's final path is
    checked instead.
    """

    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    """Cap on the serialized size of each document a tool returns.

    Content is cut first, then tables, metadata, and languages are dropped, and the document records
    what it lost.
    """

    max_batch_inputs: int = DEFAULT_MAX_BATCH_INPUTS
    """How many paths one `extract_batch` call may name.

    A result can approach `max_batch_inputs * max_output_bytes`, so size the two together.
    """

    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    """Cap on what one request uploads, all files of an `extract_batch` call together.

    Matches the server's 100 MiB defaults; raise it together with `XBERG_MAX_REQUEST_BODY_BYTES` and
    `XBERG_MAX_MULTIPART_FIELD_BYTES`.
    """

    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    """Cap on one answer from the server, refused as it arrives rather than held whole."""

    max_response_values: int = DEFAULT_MAX_RESPONSE_VALUES
    """Cap on the JSON values one answer may hold, counted before it is parsed.

    Parsing costs far more per value than the bytes read. Raise it for answers that hold more, such
    as large spreadsheets.
    """

    timeout: float = DEFAULT_TIMEOUT
    """Seconds one request may take in all, on the client the capability opens.

    A supplied `http_client` keeps its own timeouts, which is also the way to run without a deadline.
    """

    headers: Mapping[str, str] | None = field(default=None, repr=False)
    """Headers sent with every request, for a proxy or gateway in front of the server.

    The body-framing headers and `Accept-Encoding` belong to the toolset. Left out of the repr.
    """

    redact: Sequence[str] = field(default=(), repr=False)
    """More text to redact from the messages the model reads, such as a proxy's credentials.

    An entry counts as written and as the Basic credential it encodes to. Left out of the repr.
    """

    http_client: httpx.AsyncClient | None = field(default=None, repr=False)
    """Client to send requests with, instead of one opened and closed per call.

    Its retry, proxy, timeout, and `verify` settings then apply, and what it reads on its own, such
    as a response hook that reads the body, is not bounded by the caps. Runtime-only.
    """

    tools: Sequence[str] = XBERG_TOOL_NAMES
    """Which of `XBERG_TOOL_NAMES` to register. Narrow it to shrink what the model reads; the
    instructions follow."""

    include_instructions: bool = True
    """Inject Xberg usage instructions into the system prompt.

    They name the tools as registered here, so switch this off when `PrefixTools` renames them.
    """

    def __post_init__(self) -> None:
        """Reject, where written, any option a spec could have mistyped, from `id` to the limits."""
        validate_text('id', self.id)
        validate_text('description', self.description)
        validate_flag('defer_loading', self.defer_loading)
        validate_tools(self.tools)
        validate_url(self.url)
        validate_root(self.root)
        validate_output_format(self.output_format)
        validate_flag('include_instructions', self.include_instructions)
        validate_headers(self.headers)
        validate_redact(self.redact)
        validate_config(self.config)
        validate_client(self.http_client)
        validate_limits(
            max_output_bytes=self.max_output_bytes,
            max_batch_inputs=self.max_batch_inputs,
            max_upload_bytes=self.max_upload_bytes,
            max_response_bytes=self.max_response_bytes,
            max_response_values=self.max_response_values,
            timeout=self.timeout,
        )

    def __repr__(self) -> str:
        """The repr with `url` stripped of credentials; `headers` and `redact`, which carry them, are left out."""
        shown: dict[str, object] = {f.name: getattr(self, f.name) for f in fields(self) if f.repr}
        shown['url'] = display_url(self.url, with_path=True)
        parts = ', '.join(f'{name}={value!r}' for name, value in shown.items())
        return f'{type(self).__name__}({parts})'

    @classmethod
    def from_spec(
        cls,
        *,
        id: str | None = None,
        description: str | None = _DEFAULT_DESCRIPTION,
        defer_loading: bool = False,
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
        tools: Sequence[str] = XBERG_TOOL_NAMES,
        include_instructions: bool = True,
    ) -> Xberg[AgentDepsT]:
        """Construct from serializable options, leaving out the runtime-only `http_client`."""
        return cls(
            id=id,
            description=description,
            defer_loading=defer_loading,
            url=url,
            output_format=output_format,
            config=config,
            root=root,
            max_output_bytes=max_output_bytes,
            max_batch_inputs=max_batch_inputs,
            max_upload_bytes=max_upload_bytes,
            max_response_bytes=max_response_bytes,
            max_response_values=max_response_values,
            timeout=timeout,
            headers=headers,
            redact=redact,
            tools=tools,
            include_instructions=include_instructions,
        )

    def get_toolset(self) -> XbergToolset[AgentDepsT]:
        """Build the toolset that talks to the API server."""
        return XbergToolset[AgentDepsT](
            url=self.url,
            output_format=self.output_format,
            config=self.config,
            root=self.root,
            max_output_bytes=self.max_output_bytes,
            max_batch_inputs=self.max_batch_inputs,
            max_upload_bytes=self.max_upload_bytes,
            max_response_bytes=self.max_response_bytes,
            max_response_values=self.max_response_values,
            timeout=self.timeout,
            headers=self.headers,
            redact=self.redact,
            http_client=self.http_client,
            tools=self.tools,
            id=self.id if self.id is not None else 'xberg',
        )

    def get_instructions(self) -> str | None:
        """Xberg usage guidance for the system prompt, naming only the tools this capability serves."""
        if not self.include_instructions:
            return None
        served = [name for name in XBERG_TOOL_NAMES if name in self.tools]
        sentences = [_INTRODUCTION, *(_TOOL_GUIDANCE[name] for name in served)]
        if 'extract' in served or 'extract_batch' in served:
            sentences.append(_TRUNCATION_GUIDANCE)
        return ' '.join(sentences)
