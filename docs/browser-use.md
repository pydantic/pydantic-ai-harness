---
title: Browser Use
description: Your agent can log into websites, click buttons, fill out forms, extract data, and complete tasks for you.
---

# Browser Use

Give an agent a real web browser powered by the [Browser Use](https://browser-use.com) CLI: automate your local browser, or spin up one in the cloud, to click, type, extract data, fill out forms, and complete tasks. Connecting to your local browser needs no API key, and it works with your websites already logged in.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/browser_use/)

## Installation

```bash
uv add pydantic-ai-harness
```

That is the only required step. The browser is controlled by the `browser-use` CLI.

Installing the CLI outright avoids the first `uvx` run's download and pins the version you get:

```bash
uv tool install browser-use
```

Check that the CLI connects to your browser:

```bash
browser-use --doctor
```

The CLI attaches to the Google Chrome or Chromium browser on your computer. For cloud browsers, see [Cloud browsers](#cloud-browsers).

## The problem

An agent with search and fetch tools can read text on public websites, but it cannot use the web. Most of what people do involves actions: booking a flight, paying a bill, cancelling a subscription, finding invoices for an expense report, rescheduling a delivery, filling in the same details across a dozen job applications.

## The solution

`BrowserUse` gives the agent one tool that controls a real browser: your local Chrome, already signed into the websites you use, or a browser in the cloud. The agent writes Python in the browser that navigates, clicks, types, and reads the page.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.browser_use import BrowserUse

agent = Agent('openai:gpt-5.5', capabilities=[BrowserUse()])

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
| `browser_exec` | Run Python in the browser and return whatever it prints. Optional `session` picks a named cloud browser; optional `timeout_seconds` bounds one call. |

The agent writes Python to be executed within the browser and has a set of helpers -- `new_tab`, `goto_url`, `page_info`, `js`, `fill_input`, `press_key`, `scroll`, `wait_for_element`, `capture_screenshot`, tab management, and more.

The browser stays open between calls, so the agent signs in once and keeps working for the rest of the run. Its calls share one persistent Python session, so variables carry over too -- open handles included -- letting it gather results across several calls and use them at the end.

When the agent's code raises, the traceback and exit code come back to it in the tool output, along with anything it printed first. The run continues and the model can fix its code and try again. When the CLI is missing or a call times out, the tool raises a [`ModelRetry`](/ai/tools-toolsets/tools-advanced/#tool-retries) rather than a hard error.

Files the code produces flow through two channels. Screenshots come back attached as images, so the model can look at the page it captured. Every other file stays on disk, and each path the code prints is listed on the tool return's [`metadata`](/ai/tools-toolsets/tools-advanced/#advanced-tool-returns) under `files` (path, media type, size). Your application reads it from the run's `ToolReturnPart`:

```python
from pydantic_ai.messages import ModelRequest, ToolReturnPart

for message in result.all_messages():
    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.metadata is not None:
                for file in part.metadata.get('files', []):
                    print(file['path'], file['media_type'], file['bytes'])
```

If your agent, for example, scrapes data and saves it as a CSV, you can then access the file.

## Choosing a browser

| `browser` | What it uses | Reach for it when |
|---|---|---|
| `'local'` (default) | Chrome or Chromium already running on your machine | You want the sites you are already signed into |
| `'headless'` | It starts and stops headless Chromium for the run on a clean profile | You are on a server or CI, or several runs go at once and must not share tabs |
| `'cloud'` | A [Browser Use Cloud](https://cloud.browser-use.com) browser | A site blocks automated traffic, or you want the run fully off your machine to run many browsers in parallel |

```python
from pydantic_ai_harness.browser_use import BrowserUse

BrowserUse(browser='headless')
```

`'local'` shares one browser with everything else on the machine, so you can only run one task at a time. `'headless'` and `'cloud'` give each run its own browser, and both start signed out. You can import your logins to a cloud browser and run as many tasks as you want, and it automatically bypasses CAPTCHAs if necessary.

To point at a browser you are running yourself, set the `BU_CDP_URL` environment variable to its DevTools endpoint.

## Cloud browsers

Cloud browsers need an account. Sign in once, and the CLI remembers it:

```bash
browser-use auth login
browser-use auth status   # confirm it worked
```

Or set the key yourself, from <https://cloud.browser-use.com>:

```bash
export BROWSER_USE_API_KEY=your-key
```

To import all the websites you're logged into on your computer to a cloud browser, so it can complete tasks while signed into your accounts, sync your Chrome profile:

```bash
export BROWSER_USE_API_KEY=your-key && curl -fsSL https://browser-use.com/profile.sh | sh
```

## Configuration

Every field of `BrowserUse` with its default:

```python
from pydantic_ai_harness.browser_use import BrowserUse

BrowserUse(
    browser='local',         # 'local', 'headless', or 'cloud'
    scope='run',             # 'agent' shares one session across runs (chat loops)
    default_timeout=300.0,   # seconds per call when the agent gives no timeout
    progress=None,           # narrate steps live, e.g. progress=print
    guidance=None,           # None = default instructions, '' = none, str = custom
)
```

On a timeout, the CLI is killed, and the agent is told it can retry.

## Agent spec (YAML/JSON)

`BrowserUse` works with Pydantic AI's [agent spec](/ai/core-concepts/agent-spec/), so you can declare it in a config file instead of Python:

```yaml
# agent.yaml
model: openai:gpt-5.5
capabilities:
  - BrowserUse:
      browser: cloud
      default_timeout: 600
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.browser_use import BrowserUse

agent = Agent.from_file('agent.yaml', custom_capability_types=[BrowserUse])
```

Pass `custom_capability_types`, so the spec loader knows how to build `BrowserUse`. Loading YAML needs the `spec` extra: `uv add 'pydantic-ai-slim[spec]'`.

## Further reading

- [Browser Use documentation](https://docs.browser-use.com)
- [Pydantic AI capabilities](/ai/capabilities/overview/)
- [Toolsets](/ai/tools-toolsets/toolsets/)

## API reference

::: pydantic_ai_harness.browser_use.BrowserUse

::: pydantic_ai_harness.browser_use.BrowserUseToolset
