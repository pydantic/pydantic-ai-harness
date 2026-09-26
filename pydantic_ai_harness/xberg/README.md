# Xberg

Use `Xberg` when an agent has to read files it cannot read as text: PDFs, Office and OpenDocument documents, HTML,
email, scanned images, audio, archives, and source files. [Xberg](https://xberg.io) is a document intelligence engine
that turns each of those into text, tables, and metadata, and it runs as a server that your agent talks to.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/xberg/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

## Before you start

Install the Xberg CLI (`brew install xberg-io/tap/xberg`, the install script, or the `ghcr.io/xberg-io/xberg` image)
and start its REST API server:

```bash
xberg serve
```

That listens on `http://127.0.0.1:8000` by default; `-H` and `-p` change host and port, and `--config xberg.toml`
points it at a configuration file. Extraction runs where the server runs, so this capability reads each file locally
and uploads it.

You also need an API key for the model your agent uses.

## Installation

No extra: the server is reached over HTTP through the `httpx` dependency this package already declares.

uv:

```bash
uv add pydantic-ai-harness "pydantic-ai-slim[anthropic]"
```

pip:

```bash
pip install pydantic-ai-harness "pydantic-ai-slim[anthropic]"
```

Install the provider extra for a different model provider instead.

## Run your first agent

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Xberg

agent = Agent('anthropic:claude-fable-5', capabilities=[Xberg()])
result = agent.run_sync('What does reports/q3-audit.pdf conclude about deferred revenue?')
print(result.output)
#> ...
```

The model gets four tools: `extract` for one file, `extract_batch` for several in one request, `detect_mime_type` to
identify a file without extracting it, and `list_formats` to ask what the server can read.

Point it at a server somewhere else with `url`:

```python
from pydantic_ai_harness import Xberg

Xberg(url='http://xberg.internal:8000', output_format='plain')
```

## Keep documents inside the context window

Extracted content is unbounded: a scanned report can outweigh the context window it is being read into. Each document
is capped at `max_output_bytes` as serialized JSON. A part that alone serializes past the cap, tables or metadata
larger than the whole allowance, is dropped before anything else is touched, since no cut of the content could make
room for it. Then content is truncated; when not even the truncation marker fits beside the rest, tables, metadata,
and detected languages are dropped whole in that order, and content is cut again after each, so the room a dropped
part leaves goes to content before the next part is touched. The document records what it lost in `truncated` and
`omitted`; the truncation marker itself goes last, since `truncated` says the same. A cap too small for a document's
identity (its path and MIME type) is refused as a retry rather than exceeded:

```python
from pydantic_ai_harness import Xberg

Xberg(max_output_bytes=50_000, max_batch_inputs=5)
```

`max_batch_inputs` bounds one `extract_batch` call. Each document is fitted to `max_output_bytes` on its own, so one
result can approach `max_batch_inputs * max_output_bytes` of document JSON, plus the envelope and any clipped
errors; size the two together for the context window at hand. The answers of `detect_mime_type` and
`list_formats` are held to the same cap and refused when over it, and a malformed answer's diagnostics are
clipped too.

`max_response_bytes` (100 MiB by default) bounds the server's answer itself, which arrives before `max_output_bytes`
applies to each document: the text of an archive can outweigh the archive many times over, and an answer over the
cap is refused as it arrives rather than held whole. The cap bounds what the capability reads; a supplied
`http_client` reads on its own beneath it (a response hook that reads the body, an `auth` flow that asks for
response bodies, and the body of each redirect it follows, which httpx reads whole before the next hop), so keep
such reads off a client given to the capability, or bound them in the client.

`max_response_values` (1,000,000 by default) bounds how many JSON values one answer may hold, counted before it is
parsed: parsing costs a few hundred bytes for each value however small the value, so a body of tiny values under
`max_response_bytes` could otherwise cost many times its size. A value is a string (a key included), a number, a
literal (`true`, `false`, `null`, and the `NaN` and `Infinity` the parser admits), an array, or an object; the cells
of a large spreadsheet are values too, so raise the limit for a server whose answers legitimately hold more. An answer
over it is refused unparsed (an error body over it is quoted by its
head rather than parsed as the envelope, as one over 1 MiB is), and one no longer than the limit in bytes is not
scanned, since each value takes at least one.

The tools are registered as sequential, a barrier in Pydantic AI's terms: an agent runs each Xberg call alone, not
overlapping with any other tool call, so a response that asks for many extractions costs the memory of one call at a
time rather than of all of them at once. That is a bound on concurrency, not a ceiling in bytes: one call holds its
upload (up to `max_upload_bytes`), its answer as it arrives (up to `max_response_bytes`), the values parsed from it
(up to `max_response_values`, at a few hundred bytes each), and the documents fitted from those, so size a worker
for a few times the caps. To read several files in one bounded request, use `extract_batch`.

## Restrict which files the model can read

The model names the file, and the capability reads it. `root` decides how far that reaches: paths are resolved through
symlinks and one that lands outside the root is refused. It defaults to the working directory, so widen it
deliberately:

```python
from pydantic_ai_harness import Xberg

Xberg(root='/srv/incoming')  # only files under /srv/incoming
Xberg(root='/')              # anything this process can read
```

Files are opened one path component at a time from the filesystem root without following symlinks, and checked
through the descriptor they are read from, so a directory or file swapped for a symlink after the root check is
refused rather than followed. A symlink that resolves inside the root is read through its target, and uploaded
under the name the model asked for rather than the target's, so the server reads the format from the extension it
was given. On Windows, where the standard library cannot open relative to a directory handle, the file is opened
by path and refused unless the path Windows reports for the open handle is still the one checked, so a symlink or
junction swapped in after the check is refused there too.
`max_upload_bytes` (100 MiB by default) bounds what one request uploads: the file `extract` names, or every file of
an `extract_batch` call together. It matches the server's own request body limit and is checked three times: on
the size a file reports, before it is read, so a file that reports more than what is left of the limit is refused
unread; on the read itself, which goes in chunks and stops at the first byte past what is left, so a file that
reports less than it holds, or none, as files under `/proc` do, is held up to that byte and no further, with the
files of a batch read before it still held; and on the request httpx framed, multipart boundaries included. The
server also caps each file (`XBERG_MAX_MULTIPART_FIELD_BYTES`) as well as the request body
(`XBERG_MAX_REQUEST_BODY_BYTES`), both at 100 MiB by default, so raise all three together for larger files.

Every file the model names inside `root` is uploaded to whatever answers at `url`, and Xberg's server has no
authentication of its own. Loopback, the default, keeps remote hosts out but does not authenticate a local process:
on a host shared with untrusted users, a process that binds the port before the server does receives the uploads.
Where that matters, start the server before the agent and authenticate it, which a credential in `headers` or `url`
cannot do (it identifies the client to the server, and a squatter would receive it too): an `https://` URL makes
httpx verify the server's certificate on the client the capability opens (a supplied client keeps its own `verify`
setting, so one built with `verify=False` verifies nothing), a private certificate authority is pinned with
`http_client=httpx.AsyncClient(verify=ssl.create_default_context(cafile='ca.pem'))`, and where the server can
listen on a Unix socket, reaching it through `httpx.AsyncHTTPTransport(uds='xberg.sock')` on a supplied client
leaves the socket's file permissions to decide who can answer.

## Configure extraction

`config` is sent with every request, in the form the Xberg API documents for its `config` field, so OCR backends,
chunking, and embeddings are configured where they always are:

```python
from pydantic_ai_harness import Xberg

Xberg(config={'ocr': {'backend': 'paddleocr'}, 'chunking': {'max_characters': 2000}})
```

The model can override per call: `extract` and `extract_batch` each take `output_format`, `force_ocr`, and
`ocr_language`. An `ocr_language` argument merges into the `ocr` config rather than replacing it, so the backend
above survives, and `force_ocr=False` on a call switches configured forced OCR off for that file, or for every file
of that batch.

## Choose the tools

`tools` names which of Xberg's tools the model sees. Narrow it to shrink what the model has to read:

```python
from pydantic_ai_harness import Xberg

Xberg(tools=['extract'])                      # extraction only
Xberg(tools=['extract', 'detect_mime_type'])  # and the cheap identity check
```

The instructions the capability injects name only the tools it serves.

The server also exposes its version, its model cache, and an async job queue. Those are operator actions rather than
agent ones, so they are not tools: a run should not be deciding to delete a cache other runs share.

## Failure modes

Everything that goes wrong with one request reaches the model as a `ModelRetry`, so an agent can report the problem or
try something else instead of ending the run with a traceback:

- an unreachable server names the URL (without any credentials it carries, and clipped to 500 characters) and how to
  start one
- a Xberg error response carries its `error_type` and `message` (`ValidationError`, `ParsingError`, `OcrError`)
- a missing or invalid path (one with a NUL in it, or a name that is not valid UTF-8, which a multipart upload cannot
  carry), a path outside `root`, a request over `max_upload_bytes`, an answer over `max_response_bytes` or
  `max_response_values` or with a content encoding, a request past `timeout` or past a supplied client's own
  timeouts, or a batch wider than `max_batch_inputs` says so, quoting the path the model gave clipped to 500
  characters
- in a batch, a failed input appears in `errors` with the path that failed, clipped to 500 characters; the other
  documents still come back
- an answer that does not account for every uploaded file, names a failed input by an index that is repeated or
  out of range or by a source that disagrees with it, leaves a result or an error without the fields the contract
  promises, carries a metadata number that is not finite (JSON cannot carry one, and the tool result would write
  it as null), or identifies a file other than the one uploaded (by its name as given, or as httpx wrote it into
  the multipart part, where a quote, a backslash, and a control character are escaped), is refused rather than
  paired with paths by guesswork
- the server's own error text is clipped to 500 characters, since it is not under `max_output_bytes`, and only as
  much of it as the clip and the credentials in play can reach is read for redaction; an error body over 1 MiB or
  over `max_response_values` is quoted by its head rather than parsed as Xberg's envelope, so beyond receiving it,
  an error the size of the response limit costs no more than a short one
- a credential the request carried (the userinfo, password, and query values of `url` and of the URL a redirect led
  to, and every header it was sent with, on the request as built, on each hop of a redirect chain a supplied client
  followed, and on the one finally answered: `headers`, a supplied client's defaults, a cookie a redirect set, and
  what its `auth` or event hooks added, under any name, the host only when it is the one the URL derives, and
  with any value but the one httpx or the capability wrote for the request as built under that name, so a hook
  that replaced the framing of the body, its boundary or length included, or the encoding asked for makes its
  value a secret) is redacted from any message the server or transport sends about the request that the model
  reads (an error envelope or page, a diagnostic, a quoted header), since a gateway's error page may echo the
  request; a credential is redacted wherever it appears, whatever its length, even inside a longer word, so a
  short value, such as a cookie a redirect set, may cut an ordinary word that holds it, the price of letting none
  through; a header is redacted as a whole and by the parts a gateway may echo on their own, the credentials
  after an `Authorization` scheme, the pair and the password a Basic credential decodes to, each cookie's value,
  and the boundary of a `Content-Type` a hook replaced, and every credential also in the escaped form a transport
  error renders a control character in, while a username on its own is not treated as a secret; every entry of
  `redact` is redacted too, as written and as the Basic credential it encodes to (a proxy URL by its userinfo), for
  a credential the request carries where the capability cannot see it, such as a proxy's, which httpx sends as
  `Proxy-Authorization` below the request it builds
- a response header a refusal quotes, a content encoding or a `Content-Length` that is not a length, is redacted the
  same way, since a gateway may echo a credential there
- a document's content, tables, and metadata come back as the server extracted them, never redacted: the model asked
  to read that file, and altering its text would corrupt the extraction, so a file that itself holds a credential
  hands it to the model as any file-reading tool would

The agent's own `retries` setting bounds how many times the model may try again.

## Require approval

Approval is not enabled automatically. Use the public toolset with Pydantic AI's
[tool approval](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/#requiring-tool-approval) when extraction should
be confirmed (a paid OCR backend, say):

```python
from pydantic_ai import Agent
from pydantic_ai_harness.xberg import XbergToolset

agent = Agent('anthropic:claude-fable-5', toolsets=[XbergToolset().approval_required()])
```

`XbergToolset` is usable on its own with
[`Agent(toolsets=[...])`](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/) and other toolset wrappers. The
capability adds the usage instructions on top.

## Telemetry

Each extraction emits one `xberg_extract` span through `ctx.tracer`, carrying `xberg.url` (the server's scheme,
host, and port), `xberg.max_output_bytes`, `xberg.output_format`, `xberg.inputs`, `xberg.documents`,
`xberg.errors`, and `xberg.truncated_documents`: what the run asked of the server, and how much of the answer the
model actually saw. `detect_mime_type` and `list_formats` emit `xberg_detect` and `xberg_formats` spans carrying
`xberg.url`, `xberg.max_output_bytes`, and `xberg.result_bytes`, so an answer refused for its size shows the measured
and the configured size side by side. The paths the model named are content, so they are attached as
`xberg.sources` only when `trace_include_content` is enabled; the path prefix of `url`, through which a gateway may
route by tenant, joins `xberg.url` only then, and its userinfo and query string never do. A failed call leaves
`xberg.exception_type` on its span, as does a cancelled one, naming the cancellation, and a call the capability
refused says why in `xberg.refusal`: one of
`invalid_path`, `outside_root`, `unreadable_path`, `upload_limit`, `batch_limit`, `unreachable`, `timeout`,
`encoded_response`, `response_limit`, `value_limit`, `unexpected_response`, `server_error`,
`extraction_error`, or
`output_limit`, with `xberg.limit` and `xberg.measured` beside a size refusal (bytes, inputs for `batch_limit`,
or values for `value_limit`) and `xberg.status_code` beside a server error. The exception's message names the
path the model gave and can quote the server, so the exception itself is recorded only under
`trace_include_content`, and then without the exceptions it was raised from, since the cause of a redacted refusal
is the raw transport or validation error, which may quote a credential. That
covers `extract` turning its one failed input into a retry, recorded beside `xberg.errors`, and a batch refused
before upload, whose span carries what was asked and no answer.

## Sizing the connection

Retries, proxies, and connection limits belong to an `httpx.AsyncClient` you pass as `http_client`; the client the
capability opens itself ignores `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY`, since a document read from
the local filesystem must not leave the host through a proxy nobody configured for this capability. A proxy's
credentials travel in a `Proxy-Authorization` header httpx adds below the request it builds, where the capability
cannot see them, so list the proxy URL or its `user:password` in `redact` to keep a proxy error that echoes them from
the model. `timeout` (120
seconds by default, because OCR and transcription are slow) is a deadline for the whole request, from connecting to
the last byte of the answer, and applies only to that client:

```python
import httpx

from pydantic_ai_harness import Xberg

client = httpx.AsyncClient(timeout=600.0, transport=httpx.AsyncHTTPTransport(retries=3))
Xberg(url='http://xberg.internal:8000', http_client=client)
```

`http_client` is runtime-only: an agent spec loaded with `Agent.from_spec` or `Agent.from_file` can set every other
option, and one that names `http_client` is rejected.

`timeout` and the five caps (`max_output_bytes`, `max_batch_inputs`, `max_upload_bytes`, `max_response_bytes`,
`max_response_values`) must be positive, the caps integers, and none larger than 2**63 - 1; any other value raises
`UserError` where it is written, since a cap of zero refuses every call, an infinite `timeout` is no deadline, and a
spec hands over
whatever the file said; `config` has to be a mapping of JSON values under text keys, nested no more deeply than
`json` can write, for the same reason (a YAML spec parses an unquoted date into a value that is not JSON, and an
unquoted `1:` into a key `json` would write as `"1"`), `tools` and `redact` lists of strings (a mapping written for
`tools` would register its keys, and a `redact` entry has to be text UTF-8 can encode), `output_format` one of the
six names, `include_instructions` and `defer_loading` booleans, `id` and `description` text, `url` text with an
`http://` or `https://` scheme, a host, and a port between 1 and 65535 when it names one, and `root` a path. A
supplied `http_client` is the way to run without a deadline.

`headers` is for a proxy or gateway in front of the server, which documents no authentication of its own; a gateway
credential can also travel in `url`, as userinfo, which httpx sends as HTTP Basic authentication in place of any
`Authorization` header in `headers`, or as a query string. Either stays out of spans, out of the messages the server
or transport sends about the request that the model reads, and out of the capability's repr, which shows `url`
without them and leaves `headers` out; a document's own content comes back as extracted, so a file that holds the
credential returns it (see the failure modes). The headers that frame the request body
(`Content-Type`, `Content-Length`, `Transfer-Encoding`, `Accept-Encoding`) belong to the capability, and naming one
in `headers`, like `headers` that is not a mapping, a name or value that is not text, a name that is not an HTTP
token, or a value with a control
character or a leading or trailing blank (which the transport would refuse to send, quoting the value), raises
`UserError` where it is written. A supplied `http_client` is held to the first three and to the same header text as
well, where it is written and again before each request: httpx keeps a
client default over the framing its multipart encoder computes, so the upload limit would otherwise be checked
against the default rather than the body sent.
Every request asks for an unencoded answer (`Accept-Encoding: identity`) and reads it as sent; a gateway that
compresses regardless gets its answer refused, since decoding could expand one chunk past `max_response_bytes`
before it is measured.

The client the capability opens for a call, and every streamed answer, gets a shielded and bounded close when the
call is cancelled: the cancellation cannot interrupt the close, and the close is abandoned after a few seconds if
the transport does not finish, so a cancelled extraction releases its connection in all but that case.

## Composing with other capabilities

Set `root` to the directory your other file tools resolve from (`FileSystem.cwd`, `Shell.cwd`), so a path the model
learned from one of them works in `extract`. Absolute paths inside the root are accepted too.

The per-document cap keeps each result small. To hand the model a whole document it can page through instead, raise
`max_output_bytes` past a [Tool Output Limits](../tool_output_limits/README.md) band: an oversized return then spills
to a store the model reads with `read_tool_result`, rather than being cut.

Xberg also ships an MCP server (`xberg mcp`). Pydantic AI's own [`MCP`](https://pydantic.dev/docs/ai/capabilities/mcp/)
capability connects to it, so use one or the other, not both. This capability speaks the REST server because owning
the HTTP call is what lets it cap each document, keep the model's paths inside `root`, and turn a server error into a
`ModelRetry` the agent can act on.

The toolset registers `extract` and its siblings under the names Xberg gives them, so two `Xberg` capabilities on one
agent collide on the tool names. Namespace them with core's
[`PrefixTools`](https://pydantic.dev/docs/ai/capabilities/overview/):

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness import Xberg

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        PrefixTools(Xberg(url='http://fast.internal:8000'), prefix='fast'),
        PrefixTools(Xberg(url='http://ocr.internal:8000'), prefix='ocr'),
    ],
)
```

The prefix applies to the tool names, not to the instructions the capability injects, which would still say
`extract`. Pass `include_instructions=False` and write your own when you prefix.

To keep the tools out of the model's context until they are needed, give the capability an `id` and pass
`defer_loading=True`:

```python
from pydantic_ai_harness import Xberg

Xberg(id='xberg', defer_loading=True)
```
