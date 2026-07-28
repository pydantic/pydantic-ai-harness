# Browser Use

Give an agent a real web browser powered by the [Browser Use](https://browser-use.com) CLI: automate your local browser, or spin up one in the cloud, to click, type, extract data, fill out forms, and complete tasks. Connecting to your local browser needs no API key, and it works with your websites already logged in.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/browser_use/)

## Installation

```bash
uv add pydantic-ai-harness
```

That is the only required step. The browser is driven by the `browser-use` CLI, which runs as a separate program rather than a Python import, so it is not a dependency of your project. When the CLI is not on your PATH, the tool runs `uvx browser-use` instead. When neither is available, it hands the agent the install command rather than failing the run.

Installing the CLI outright avoids the first `uvx` run's download and pins the version you get:

```bash
uv tool install browser-use
```

Local browsing needs no account or API key. Check that the CLI reaches your browser:

```bash
browser-use --doctor
```

It attaches to a running Chrome or Chromium with remote debugging on, and walks you through turning it on if needed. For cloud browsers, see [Cloud browsers](#cloud-browsers).

## The problem

An agent with search and fetch tools can read text on public websites, but it cannot use the web. Most of what people do involves actions: booking a flight, paying a bill, cancelling a subscription buried four clicks deep in account settings, pulling last year's invoices for an expense report, rescheduling a delivery, filling in the same details across a dozen job applications.

None of that is reachable by reading. Fetch one of those pages and you get a login wall, a cookie banner, or an empty shell that fills in only once JavaScript runs. The work is behind a button, and a fetch tool has no way to press it.

The same limit applies to an agent checking its own work. To confirm a fix really shipped, it has to sign in and click the button a user would click.

## The solution

`BrowserUse` gives the agent one tool that drives a real browser: your local Chrome, already signed in to the sites you use, or a fresh browser in the cloud. The agent writes short Python that navigates, clicks, types, reads the page, and takes screenshots, and gets back whatever that code prints.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.browser_use import BrowserUse

agent = Agent('openai:gpt-5.5', capabilities=[BrowserUse()])

result = agent.run_sync('Open news.ycombinator.com and summarize the top 5 stories.')
print(result.output)
```

## What agents do with it

- **Read what sits behind a login.** Pull the numbers off a dashboard, an invoice out of a billing portal, a thread out of a support inbox -- through the session already open in your browser, with no API to wire up first.
- **Fill in and submit.** Complete forms, place an order, file a ticket, upload a document, work through a checkout.
- **Handle sites that only render in a browser.** Single-page apps, infinite scroll, filters and pagination that never change the URL.
- **Test your own app.** Sign up, click through the flow, screenshot the result, check that a fix reached staging.
- **Repeat the same admin work.** Copy a record from one internal tool into another, every morning, without an integration between them.

## How it works

`BrowserUse` adds a single `browser_exec` tool that pipes the model's Python to the `browser-use` CLI on stdin. The CLI runs that code with browser helpers already imported (`new_tab`, `goto_url`, `page_info`, `js`, `fill_input`, `capture_screenshot`, and more) against a browser it keeps alive in a background daemon.

Two things follow from running the CLI as a subprocess. The harness pulls in no new Python packages, so nothing collides with the versions your project pins. And because the daemon holds the browser open, tabs, cookies, and logins survive from one call to the next: the agent signs in once and keeps working for the rest of the run.

One call also carries a whole step -- navigate, wait, extract, act -- rather than spending a model round trip per click.

## The tool

| Tool | Purpose |
|---|---|
| `browser_exec` | Execute Python in the persistent browser session and return what it prints. Accepts optional `session` (named cloud browser) and `timeout_seconds`. |

The browser and the model's Python variables both persist across calls, so the agent can gather results over several calls and use them later. Code that raises comes back to the model with the traceback and exit code, so it can correct itself and try again.

One persistent Python session serves the whole run, so everything the code assigns survives to the next call -- including open handles that a snapshot could never carry. A call that times out restarts the session, while the browser itself is unaffected, since it lives in the CLI's daemon. Each agent run gets its own session, discarded when the run ends, so concurrent runs never see each other's variables; `scope='agent'` instead shares one session across every run inside `async with agent:`, which is what a chat loop wants.

## Instructions

The model is told about the browser twice, and the two sources divide the work.

The system prompt gets a short note, set with `guidance` (or `''` for none). It covers only what the CLI's own documentation cannot: that variables persist between calls here, unlike at the shell, and which values do not survive. Batching a whole step into one call is still worth doing, since it costs a round trip rather than a line of code.

The model also gets the CLI's own reference documentation, fetched once with `browser-use skill show` and appended to the `browser_exec` description. The model writes the Python this tool runs, so it needs that reference to write it correctly: full helper signatures, and the workflow the CLI recommends -- drive from the accessibility tree rather than screenshots, verify after each click, wait for load after navigating, stop and ask at a login wall, and prefer a plain HTTP fetch when no browser is needed. It is the same text the CLI hands to any other agent, and it tracks whichever CLI version is installed.

When the CLI is missing or slow to answer, the tool keeps its short built-in description; a setup problem then surfaces when the model calls the tool, rather than breaking agent construction.

## Cloud browsers

Cloud browsers are optional. Reach for one when the agent runs on a headless server, when several agents browse at once (they would otherwise fight over the tabs of one local Chrome), or when a site blocks automated traffic -- Browser Use runs cloud browsers with managed IPs and stealth settings.

Sign in once and the CLI stores the credentials:

```bash
browser-use auth login    # add --device-code over SSH
browser-use auth status   # confirm it worked
```

Or set the key yourself. Create one at <https://cloud.browser-use.com>, then export it:

```bash
export BROWSER_USE_API_KEY=your-key
```

The environment variable wins when both are set.

To run in a named cloud browser, the agent starts one with `start_remote_daemon('<name>')` and passes that name as the tool's `session` argument. To carry your logins into it, first sync your local browser profile (interactive; prints the profile to use, and re-sync when a site's login expires):

```bash
export BROWSER_USE_API_KEY=your-key && curl -fsSL https://browser-use.com/profile.sh | sh
```

then attach it when the browser is created: `start_remote_daemon('<name>', profileName='<profile>')`. A cloud browser without a profile starts logged out.

## Options

Every field of `BrowserUse` with its default:

```python
from pydantic_ai_harness.browser_use import BrowserUse

BrowserUse(
    browser='local',         # 'local', 'headless', or 'cloud'
    scope='run',             # 'agent' shares one session across runs (chat loops)
    command='browser-use',   # CLI name or path
    default_timeout=300.0,   # seconds per call when the agent gives no timeout
    progress=None,           # narrate steps live, e.g. progress=print
    guidance=None,           # None = default instructions, '' = none, str = custom
)
```

On timeout the CLI's process group is killed and the model is told it can retry; the browser daemon keeps running, so the session is not lost. To attach to a browser you run yourself, set the `BU_CDP_URL` environment variable to its DevTools endpoint; `BH_CHROME_PATH` picks the binary for headless mode.

## Security

The agent's code runs on your machine with no sandbox, and it reaches every site your browser is signed in to. Treat `BrowserUse` the way you treat `Shell`: give it to agents whose prompts you control. LLM provider API keys (`OPENAI_*`, `ANTHROPIC_*`, `GOOGLE_*`, ...) are scrubbed from the browser subprocess automatically; your Browser Use cloud credentials pass through, since the CLI needs them.

For stronger isolation, point the agent at a cloud browser rather than your local Chrome: it starts clean, sees none of your logged-in sessions, and carries only the cookies you sync to it.

## Agent spec (YAML/JSON)

`BrowserUse` works with Pydantic AI's [agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: openai:gpt-5.5
capabilities:
  - BrowserUse:
      workspace: /tmp/agent-workspace
      default_timeout: 600
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.browser_use import BrowserUse

agent = Agent.from_file('agent.yaml', custom_capability_types=[BrowserUse])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate `BrowserUse`.

## Further reading

- [Browser Use documentation](https://docs.browser-use.com)
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
