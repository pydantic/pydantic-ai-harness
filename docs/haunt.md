---
title: Haunt Extraction
description: Give a Pydantic AI agent tools for reading supported web pages as Markdown and extracting specific fields as JSON through the Haunt API.
---

# Haunt Extraction

> [!NOTE]
> Import this capability from its submodule. There is no top-level
> `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.haunt import HauntExtract
> ```
>
> The API may change between releases. Where practical, breaking changes ship
> with a deprecation warning.

Give an agent tools for reading supported web pages as Markdown and extracting
specific fields as JSON through the [Haunt](https://hauntapi.com/) API.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/haunt/)

## Installation

No extra is needed. Set `HAUNT_API_KEY` to a key from the
[Haunt dashboard](https://hauntapi.com/?utm_source=pydantic-ai-harness&utm_medium=integration&utm_campaign=haunt-capability#signup):

```bash
export HAUNT_API_KEY='haunt_...'
```

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.haunt import HauntExtract

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[HauntExtract()])

result = agent.run_sync(
    'Read https://hauntapi.com/#pricing and return the plan names and prices.'
)
print(result.output)
```

`HauntExtract` contributes two tools:

| Tool | Purpose |
| --- | --- |
| `read_page(url)` | Read a page as Markdown. |
| `extract_data(url, prompt)` | Extract fields described in plain English and return them as JSON. |

For successful results, both tools keep up to `max_text_chars` source-content
characters, which defaults to 50,000. Longer results keep the beginning and
then append a truncation marker. Short page-level failure reports are not
subject to this content limit.

## Page-level failures

Some pages cannot be read during a run. Haunt returns typed failure codes for
common terminal cases:

| Code | Meaning |
| --- | --- |
| `access_denied` | The site refused the request. |
| `login_required` | The content requires an authenticated session. |
| `captcha_required` | The page requires human verification. |
| `not_found` | The page was not found. |

These failures are returned to the model as tool text instead of triggering an
automatic retry of the same URL. The code is also available at
`ToolReturnPart.metadata['error_code']`, so an application can branch without
parsing the message.

Network errors, timeouts, rate limits, and other non-success HTTP responses
raise [`ModelRetry`](/ai/tools-toolsets/tools-advanced/#tool-retries).
Authentication failures raise `UserError`.

## Configuration

```python
from pydantic_ai_harness.haunt import HauntExtract


HauntExtract(
    max_text_chars=20_000,
    guidance='Use extraction only for the product pages the user names.',
)
```

Set `guidance=''` to disable the default agent instructions.

## Custom client

Pass `HttpxHauntClient` to set the API key, base URL, timeout, or HTTPX
transport explicitly. A client passed by the caller remains caller-owned and
should be closed by the caller:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.haunt import HauntExtract, HttpxHauntClient


async def run() -> str:
    async with HttpxHauntClient(api_key='haunt_...') as client:
        agent = Agent(
            'anthropic:claude-sonnet-4-6',
            capabilities=[HauntExtract(client=client)],
        )
        result = await agent.run('Read https://example.com and summarize it.')
        return result.output
```

Any object that satisfies the `HauntClient` protocol can be used instead,
including a test double or a client with custom transport policy.
