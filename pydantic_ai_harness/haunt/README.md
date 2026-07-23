# Haunt Extraction

Give an agent web extraction tools backed by the
[Haunt](https://hauntapi.com/?utm_source=pydantic-ai-harness&utm_medium=integration&utm_campaign=sweep-2026-07)
API: `read_page` returns any public page as clean Markdown, and `extract_data`
returns specific fields, described in plain English, as structured JSON. When a
page cannot be read, the tools say so honestly instead of returning fabricated
content.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/haunt/)

## Installation

No extra is needed: the capability uses `httpx`, which Pydantic AI already
depends on.

Set the `HAUNT_API_KEY` environment variable (free key at
<https://hauntapi.com/#signup?utm_source=pydantic-ai-harness&utm_medium=integration&utm_campaign=sweep-2026-07>),
or pass a configured client (see [Custom client](#custom-client)).

## The problem

Web tools fail dishonestly. A blocked page comes back as an empty string, a
bot-wall interstitial gets summarized as if it were the article, and a
login-walled shop page yields a confidently wrong price. The agent cannot tell
a bad page from a bad extraction, so it hallucinates around the gap and the
error surfaces much later, in the agent's answer, where it is hardest to trace.

`HauntExtract` makes failure a first-class result. When the page itself is
unavailable, the tool returns an error code and a plain-words reason:

| Code | Meaning |
|---|---|
| `access_denied` | The site refused automated access (bot wall) |
| `login_required` | The content is behind a login |
| `captcha_required` | A captcha guards the page |
| `not_found` | The page does not exist |

The model sees the code in the tool text (and is instructed to treat it as
final for that URL), and applications can branch on
`ToolReturnPart.metadata['error_code']` without parsing the text.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.haunt import HauntExtract

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[HauntExtract()])

result = agent.run_sync(
    'What does https://hauntapi.com say the free tier includes?'
)
print(result.output)
```

The agent gets two tools:

- **`read_page(url)`** -- the page as clean Markdown, for reading articles,
  docs, or product pages in full.
- **`extract_data(url, prompt)`** -- specific fields as structured JSON, with
  the fields described in plain English, e.g. `'the product name, price, and
  stock status'`.

Both cap their output at `max_text_chars` characters (default 50,000),
truncated head-first with a marker.

## Configuration

```python
HauntExtract(
    max_text_chars=20_000,  # cap tool text per call
    guidance='',            # '' disables the default system-prompt guidance
)
```

## Custom client

Authentication defaults to the `HAUNT_API_KEY` environment variable. Pass a
client to configure it explicitly, point at a different base URL, or
substitute a fake in tests -- anything satisfying the `HauntClient` protocol
(a single async `extract` method) works:

```python
from pydantic_ai_harness.haunt import HauntExtract, HttpxHauntClient

capability = HauntExtract(client=HttpxHauntClient(api_key='haunt_...'))
```

## Error handling

- **Honest page failures** (`access_denied`, `login_required`,
  `captcha_required`, `not_found`) are returned as tool text, never raised:
  the page will not change within a run, so raising would either burn retries
  or abort the run. The model branches instead.
- **Transient failures** (network errors, timeouts, rate limits) raise
  `ModelRetry`, so the model can retry or adjust.
- **Auth failures** (bad or missing API key) raise `UserError`: configuration
  the model cannot correct.
