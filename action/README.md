# Run a Pydantic AI agent

This composite action installs Pydantic AI and `pydantic-ai-harness`, then runs an agent directly on a GitHub Actions
runner. It uses `pydantic_ai_harness.coder:coder_agent` by default. Set `agent` to use an existing Python agent or a
Pydantic AI agent spec instead.

## Security posture

There is no sandbox. The agent runs as the workflow user directly on the runner.

The default `Coder` agent has filesystem and shell access. Using the default agent therefore gives the model shell
access to the runner and everything available on it. Anything in the job environment is reachable by the agent,
including `GITHUB_TOKEN` if the workflow puts it there. Do not pass `GITHUB_TOKEN` to the step, and minimize the
workflow's `permissions:`.

Use this action only when the workflow trigger and prompt are trusted. For untrusted triggers, use the
[gh-aw engine](https://pydantic.dev/docs/ai/harness/gh-aw/) instead. It runs the agent behind an egress firewall, keeps
credentials outside the agent sandbox, and writes changes back through safe outputs.

Provider credentials belong in `env:` on the calling step, where the workflow controls secret handling. This action
does not define an API key input, so a key does not pass through an action input that could be logged on error. OIDC or
workload identity federation is the better path. The official
[Anthropic action](https://github.com/anthropics/claude-code-action) and
[OpenAI action](https://github.com/openai/codex-action) support that approach. This action does not implement the
credential exchange yet.

## Basic usage

The job needs a checkout when the agent should inspect or change repository files. `persist-credentials: false` keeps
the checkout from storing a GitHub credential where the agent can read it.

```yaml
name: Ask the coder

on:
  workflow_dispatch:
    inputs:
      prompt:
        description: Task for the agent
        required: true

permissions:
  contents: read

jobs:
  run-agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          persist-credentials: false

      - id: agent
        uses: pydantic/pydantic-ai-harness/action@main
        with:
          model: openai:gpt-5.6-sol
          prompt: ${{ inputs.prompt }}
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Use the result
        env:
          AGENT_RESULT: ${{ steps.agent.outputs.result }}
        run: printf '%s\n' "$AGENT_RESULT"
```

Pin this action to a full commit SHA in production workflows.

Exactly one of `prompt` and `prompt-file` must be set. A prompt file is read relative to `working-directory`:

```yaml
- uses: pydantic/pydantic-ai-harness/action@main
  with:
    model: anthropic:claude-sonnet-4-5
    prompt-file: .github/prompts/review.md
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Use your own agent

For a Python agent, pass the same `module:variable` shape accepted by the gh-aw engine's `PAI_AGENT`. Install the
package that provides it with `pip-install`. For example, this installs the checked-out project before importing
`my_project.agents:reviewer`:

```yaml
- uses: pydantic/pydantic-ai-harness/action@main
  with:
    agent: my_project.agents:reviewer
    model: openai:gpt-5.6-sol
    prompt: Review the current checkout.
    pip-install: .
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

An agent spec target ends in `.yml`, `.yaml`, or `.json`. The path is resolved from `working-directory`:

```yaml
- uses: pydantic/pydantic-ai-harness/action@main
  with:
    agent: .github/agents/reviewer.yml
    model: openai:gpt-5.6-sol
    prompt: Review the current checkout.
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The action installs `pydantic-ai-slim` with the OpenAI, Anthropic, and agent-spec extras. Use `pip-install` for other
providers and for dependencies imported by a custom agent. Its value is a space-separated package list. Set
`harness-version` to install an exact `pydantic-ai-harness` release instead of the latest release.

The agent's text output is printed to stdout, exposed as the `result` action output, and appended to the job's step
summary.
