# The `pydantic-ai` engine for GitHub Agentic Workflows

`pydantic.md` in this directory is the engine definition that
[GitHub Agentic Workflows](https://github.com/github/gh-aw) (gh-aw) uses for its
`pydantic-ai` engine. This directory is the source; the `gh-aw-engine` branch of
this repository is what ships, and workflows import the file from that branch.

A workflow has to write the `imports:` line itself. gh-aw's engine catalog maps
the `pydantic-ai` id to this path, but only to suggest it: naming the engine
without the import fails to compile with a tip carrying the line to add.

The engine runs the [Pydantic AI](https://ai.pydantic.dev) CLI (`pai`) over an agent
composed from this package's `Coder` capability: filesystem, shell, planning,
repository context and an explorer sub-agent, plus one toolset per MCP server the
gh-aw gateway exposes.

## Quick start

```yaml
---
on:
  issues:
    types: [opened]
permissions:
  contents: read
imports:
  - pydantic/pydantic-ai-harness/gh-aw/pydantic.md@gh-aw-engine
engine:
  id: pydantic-ai
  model: copilot/claude-sonnet-4-5
---

# Triage

Read the issue and summarize what changed.
```

`model` is required and must be `provider/model`. gh-aw accepts `copilot`,
`anthropic`, `openai` and `codex` as the provider segment; it selects which backend
of the AWF api-proxy serves the request. The proxy holds the credentials, so a
workflow supplies none.

## What actually runs

`pai -a` takes one target and its agent-spec format cannot name harness
capabilities, so the engine writes the composition as a Python module at
`.pydantic-ai/gh_aw_agent.py` and always passes `-a gh_aw_agent:agent`. The CLI and
its dependencies are installed before the agent starts, with
`pip install --user "pydantic-ai-harness[cli]==<engine version>"
"pydantic-ai-slim[openai,mcp]"`. The pinned version is `engine.version` in
`pydantic.md`, and it always names a published release: CI refuses to move the
`gh-aw-engine` branch to a commit pinning a version that is not on PyPI.

MCP servers arrive as `.pydantic-ai/mcp.json` in the `mcpServers` shape Claude
Desktop and Cursor use. Tools carry their server name as a prefix, so safe outputs
are reachable as `safeoutputs_create_issue` and so on. HTTP servers are carried
over; CLI-mounted servers remain on the agent's `PATH` as executables.

## Pointing the engine at your own endpoint

`PAI_BASE_URL` in `engine.env` sends requests to any endpoint that speaks the
OpenAI **Chat Completions** API, instead of the AWF api-proxy. The engine skips
`/reflect` discovery and uses the URL verbatim.

```yaml
engine:
  id: pydantic-ai
  model: openai/<model-id>
  env:
    PAI_BASE_URL: https://your-endpoint.example.com/v1
network:
  allowed:
    - your-endpoint.example.com
```

Three things to get right:

- **The URL is used verbatim.** The client posts to `<PAI_BASE_URL>/chat/completions`,
  so include whatever path prefix the endpoint expects, usually `/v1`.
- **Add the host to `network.allowed`.** The agent runs behind an egress firewall
  that denies everything else.
- **Keep the `provider/` prefix on `model`.** gh-aw requires it and only accepts its
  four known providers, but the engine strips it before calling the endpoint. Write
  `openai/<model-id>`, and `<model-id>` is what goes upstream. `openai` here means
  "OpenAI-compatible", not OpenAI the company.

`PAI_BASE_URL` is a variable of this engine's own rather than `OPENAI_BASE_URL`,
because gh-aw sets `OPENAI_BASE_URL` itself, pointing at the proxy. It is always
present, so it cannot express a choice.

### Credentials, and why there is no `PAI_API_KEY`

gh-aw keeps repository secrets out of the agent sandbox. Any `engine.env` value
containing `${{ secrets.* }}` is stripped from the agent's environment
(`awf --exclude-env`), and the compiler rejects the workflow rather than letting you
believe otherwise. That is deliberate: for the proxy-backed providers, credentials
live in the proxy, outside the agent, and the agent sends a placeholder bearer token
it cannot leak.

The same rule applies here, so this engine has no key setting. Two consequences:

**Endpoints that need no credential from the agent work directly.** A self-hosted
OpenAI-compatible server, a local model runner, or an internal gateway that
authenticates by network position rather than by token:

```yaml
engine:
  id: pydantic-ai
  model: openai/qwen3-coder
  env:
    PAI_BASE_URL: http://models.internal.example.com/v1
network:
  allowed:
    - models.internal.example.com
```

**Keyed providers are reached through a gateway you run.** For a commercial
endpoint that requires an API key (MiniMax, Together, Fireworks and the like), point
`PAI_BASE_URL` at a service you control that holds the key and forwards upstream:

```
agent (placeholder bearer)  ->  your gateway (adds the real key)  ->  provider
```

The gateway is yours to deploy and is out of scope for this repository. What matters
here is the shape: the credential lives on the far side of the sandbox boundary, the
same place gh-aw already keeps them. There is no configuration in this engine that
puts a provider key in the agent's hands, and adding one is not possible without a
change to gh-aw.

## Troubleshooting

Only mechanisms that have been checked against a compiled workflow are listed.

**The request never leaves, or fails to connect.** The host is missing from
`network.allowed`. The agent's egress is default-deny, and `PAI_BASE_URL` does not
open a hole on its own.

**404 from the endpoint.** The base path is wrong. The client appends
`/chat/completions`, so `https://host/v1` produces `https://host/v1/chat/completions`.
An endpoint documented as `https://host/v1/chat/completions` should be given as
`https://host/v1`.

**401 or 403 from the endpoint.** It wanted a credential. The agent sent the
placeholder bearer token, which is all it has. Put a gateway in front that adds the
real key, per the section above.

**`engine.model is required ... must use provider/model format` at compile time.**
The `provider/` prefix is missing. gh-aw validates it before the engine ever runs,
even when `PAI_BASE_URL` makes the provider irrelevant.

**`strict mode: secrets detected in 'engine.env'` at compile time.** A secret was
put in `engine.env`. It would have been removed from the agent's environment anyway;
see the credentials section.

## Changing this file

`pydantic.md` is the published artifact. Editing it here does not ship it: CI moves
the `gh-aw-engine` branch, either after a release whose tag matches the pinned
`engine.version`, or through the `gh-aw engine ref` workflow for changes that do not
need a release. Both refuse to move the branch backwards.
