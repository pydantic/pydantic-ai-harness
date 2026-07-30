---
title: Browser Use
description: Your agent can log into websites, click buttons, fill out forms, extract data, and complete tasks for you.
---

# Browser Use

Give an agent a real web browser powered by the [Browser Use](https://browser-use.com) Python SDK: delegate whole tasks to an autonomous browser agent that clicks, types, extracts data, fills out forms, and completes them -- in a headless browser on your machine, a browser you already use with your logins saved, or one in the cloud.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/browser_use/)

## Installation

```bash
uv add "pydantic-ai-harness[browser-use]"
```

That is the only required step. The extra needs Python 3.11+.

The browser is driven directly over CDP: by default, headless Chromium is launched on your machine per session, downloaded on first run if none is found. For cloud browsers, see [Cloud browsers](#cloud-browsers).

## The problem

An agent with search and fetch tools can read text on public websites, but it cannot use the web. Most of what people do involves actions: booking a flight, paying a bill, cancelling a subscription, finding invoices for an expense report, rescheduling a delivery, filling in the same details across a dozen job applications.

## The solution

`BrowserUse` gives the agent one tool that hands whole tasks to a real browser. An autonomous [browser-use](https://github.com/browser-use/browser-use) agent looks at the page, plans, clicks, types, scrapes data, and reports back when the task is done.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.browser_use import BrowserUse

agent = Agent('openai:gpt-5.5', capabilities=[BrowserUse(llm='openai:gpt-5.5')])

result = agent.run_sync('Open news.ycombinator.com, and summarize the top 5 stories.')
print(result.output)
```

## What agents do with it

These are what people most often ask a browser agent to do, in order of how common they are:

- **Research and compare across sites.** Answer a question that takes several sources, rank options, or check a claim -- the top repositories on GitHub, which fund recently bought a stock, what an organization is affiliated with.
- **Pull repeated records into a table.** Scrape products, listings, prices, reviews, or videos into structured output: every offer on an Amazon product page, or every venue on Google Maps in one city.
- **Reach a page and act on it.** Open a site, get through to the right screen, and do one thing there -- often waiting for you to log in first, then carrying on.
- **Test a website end to end.** Walk a real user flow, audit the UX, hunt for bugs, or re-check a page after a change.
- **Build and run browser automation.** Write UI tests for a page, script a repeatable crawl, or reproduce an interaction step by step.
- **Fill in forms and account flows.** Registrations, surveys, course modules, assessments, and other multi-page sequences.
- **Shop and book.** Compare products, prices, sellers, and coupons, search listings against your criteria, and go through checkout or a booking.
- **Apply for jobs and update systems.** Work through applications, or log into an admin tool and update records, listings, and settings.

## Tools

`BrowserUse` contributes one tool to the agent:

| Tool | Purpose |
|---|---|
| `browse_web` | Hand one web task, stated in natural language, to the browser agent, and return its result. |

Each call runs the browser agent's loop to completion, up to `max_steps` steps. The [prompting guide](https://docs.browser-use.com/open-source/customize/agent/prompting-guide) covers how to phrase tasks that browser agents do well.

Set `progress` (e.g. `progress=print`) to watch the task and each step live while your browser agent runs.

## The browser agent's model

Pass `llm` the same model string or `Model` your host agent uses. It carries over all the configuration from `PydanticAIChatModel`, including your API key and structured output. Browser Use's own [chat models](https://docs.browser-use.com/open-source/supported-models) (`ChatOpenAI`, `ChatAnthropic`, ...) pass through as-is. `llm=None` uses Browser Use's hosted model, billed to your `BROWSER_USE_API_KEY`.

## Choosing a browser

| Configuration | What it uses | Reach for it when |
|---|---|---|
| (default) | Headless Chromium launched per session on a clean profile | You are on a server or CI, or several runs go at once and must not share tabs |
| `cdp_url` | A browser you run yourself, via its DevTools endpoint | You want the sites you are already signed into |
| `use_cloud=True` | A [Browser Use Cloud](https://cloud.browser-use.com) browser | You want to bypass CAPTCHAs, or you want the run fully off your machine to run many browsers in parallel |

`headless=False` shows the browser window on local launches. `cdp_url` connects to [your own Chrome](https://docs.browser-use.com/open-source/customize/browser/real-browser) or a [remote browser](https://docs.browser-use.com/open-source/customize/browser/remote). For everything else -- [staying logged in](https://docs.browser-use.com/open-source/customize/browser/authentication) with a persistent `user_data_dir`, proxy, cookies, viewport -- pass a [`BrowserProfile`](https://docs.browser-use.com/open-source/customize/browser/all-parameters) as `browser_profile`; it covers every browser option.

## Cloud browsers

Cloud browsers need an account. Set the key from <https://cloud.browser-use.com>:

```bash
export BROWSER_USE_API_KEY=your-key
```

To import all the websites you're logged into on your computer to a cloud browser, so it can complete tasks while signed into your accounts, sync your Chrome profile:

```bash
export BROWSER_USE_API_KEY=your-key && curl -fsSL https://browser-use.com/profile.sh | sh
```

A cloud browser is provisioned when a session starts, released when it stops, and bills while it runs. It automatically bypasses CAPTCHAs if necessary.

## Structured output

Set `output_schema` to a Pydantic model class, and the browser agent produces its [final result in that schema](https://docs.browser-use.com/open-source/customize/agent/output-format). The tool returns the validated result as JSON:

```python
from pydantic import BaseModel
from pydantic_ai_harness.browser_use import BrowserUse


class Product(BaseModel):
    name: str
    price_usd: float


BrowserUse(output_schema=Product)
```

## Secrets & Security

[`sensitive_data`](https://docs.browser-use.com/open-source/examples/templates/sensitive-data) lets the browser agent type credentials without its model ever seeing the values. You can also scope your secrets to a specific domain, so the values cannot be typed elsewhere, and prevent your agent from accessing other domains.

```python
from pydantic_ai_harness.browser_use import BrowserUse

BrowserUse(
    allowed_domains=['travel.example.com'],
    sensitive_data={'https://travel.example.com': {'x_user': 'me@example.com', 'x_pass': '...'}},
)
```

## Configuration

Every field of `BrowserUse` with its default:

```python
from pydantic_ai_harness.browser_use import BrowserUse

BrowserUse(
    llm=None,                    # browser agent's model: Pydantic AI model/string; None = Browser Use's hosted model
    browser_profile=None,        # full browser-use BrowserProfile: proxy, user_data_dir, cookies, ...
    allowed_domains=None,        # navigation allowlist, e.g. ['*.example.com']; None = unrestricted
    headless=None,               # False shows the browser window on local launches
    max_steps=50,                # cap on browser-agent steps per call (one LLM call each)
    use_vision=True,             # send page screenshots to the browser agent's model
    output_schema=None,          # Pydantic model class for a structured, validated result
    sensitive_data=None,         # secrets typed by the browser, never shown to the model
    extend_system_message=None,  # extra standing instructions for the browser agent
    agent_settings=None,         # BrowserAgentSettings: every remaining browser_use.Agent option
    session_scope='call',        # 'agent' shares one browser session across runs (chat loops)
    cdp_url=None,                # attach to a browser you run yourself, over CDP
    use_cloud=None,              # True = a Browser Use Cloud browser (needs BROWSER_USE_API_KEY)
    guidance=None,               # None = default instructions, '' = none, str = custom
    progress=None,               # narrate the browser agent's steps live, e.g. progress=print
    browser_agent=None,          # factory for building the browser agent; the seam for tests
)
```

`session_scope='agent'` keeps one browser session across runs -- tabs, logins, and page state carry over -- until the capability is closed with `aclose()` or `async with`:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.browser_use import BrowserUse


async def main():
    async with BrowserUse(llm='openai:gpt-5.5', session_scope='agent') as browser:
        agent = Agent('openai:gpt-5.5', capabilities=[browser])
        first = await agent.run('Log in to app.example.com with the stored credentials.')
        await agent.run('Now open the latest report.', message_history=first.all_messages())
```

A run that fails in `'agent'` scope kills the shared session, and the next call starts fresh. The default `'call'` scope gives every call its own browser, so concurrent calls run in parallel.

`agent_settings` takes a `BrowserAgentSettings` covering [the rest of `browser_use.Agent`'s options](https://docs.browser-use.com/open-source/customize/agent/all-parameters) (judge, planning, flash mode, timeouts, GIF recording, custom [tools](https://docs.browser-use.com/open-source/customize/tools/basics), and so on). Options you have to build in code -- callbacks, injected state, a custom skill service -- go through a `browser_agent` factory instead, which also substitutes a fake in tests:

```python
from browser_use import Agent as BrowserUseAgent
from pydantic_ai_harness.browser_use import BrowserAgent, BrowserTask, BrowserUse


def factory(request: BrowserTask) -> BrowserAgent:
    return BrowserUseAgent(
        task=request.task,
        llm=request.llm,
        browser_session=request.browser_session,
        enable_signal_handler=False,
        skill_ids=['*'],
    )


BrowserUse(browser_agent=factory)
```

## Agent spec (YAML/JSON)

`BrowserUse` works with Pydantic AI's [agent spec](/ai/core-concepts/agent-spec/), so you can declare it in a config file instead of Python:

```yaml
# agent.yaml
model: openai:gpt-5.5
capabilities:
  - BrowserUse:
      use_cloud: true
      max_steps: 30
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.browser_use import BrowserUse

agent = Agent.from_file('agent.yaml', custom_capability_types=[BrowserUse])
```

Pass `custom_capability_types`, so the spec loader knows how to build `BrowserUse`. Loading YAML needs the `spec` extra: `uv add 'pydantic-ai-slim[spec]'`. The `llm`, `browser_profile`, `output_schema`, `agent_settings`, `progress`, and `browser_agent` fields are not spec-serializable; spec-loaded instances use Browser Use's defaults for them.

## Further reading

- [Browser Use documentation](https://docs.browser-use.com/open-source/introduction)
- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Toolsets](/ai/tools-toolsets/toolsets/)

## API reference

::: pydantic_ai_harness.browser_use.BrowserUse

::: pydantic_ai_harness.browser_use.BrowserAgentSettings

::: pydantic_ai_harness.browser_use.PydanticAIChatModel

::: pydantic_ai_harness.browser_use.BrowserUseToolset

::: pydantic_ai_harness.browser_use.BrowserTask

::: pydantic_ai_harness.browser_use.BrowserAgentFactory

::: pydantic_ai_harness.browser_use.BrowserAgent

::: pydantic_ai_harness.browser_use.BrowserAgentHistory
