"""Chat with a Pydantic AI agent that controls your browser via the BrowserUse capability.

uv run examples/browser_use.py                       # interactive chat (needs an LLM key)
uv run examples/browser_use.py "YOUR TASK"           # same, with a task you specify
uv run examples/browser_use.py --cloud "YOUR TASK"   # run in a Browser Use cloud browser (Browser Use API key needed)
uv run examples/browser_use.py --verbose "YOUR TASK"  # show each step's code and output, not just its label
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from pydantic_ai import Agent

from pydantic_ai_harness.browser_use import BrowserUse

_PROVIDERS = (
    ('OPENAI_API_KEY', 'openai:gpt-5.5'),
    ('ANTHROPIC_API_KEY', 'anthropic:claude-sonnet-5'),
    ('GEMINI_API_KEY', 'google-gla:gemini-2.0-flash'),
)


def _load_dotenv() -> None:
    env_file = Path(__file__).parent.parent / '.env'
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        name, _, value = line.partition('=')
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def _pick_model() -> str | None:
    if override := os.environ.get('MODEL'):
        return override
    return next((model for env_var, model in _PROVIDERS if os.environ.get(env_var)), None)


async def chat(model: str, cloud: bool, verbose: bool, opening_task: str | None) -> None:  # noqa: D103
    capability = BrowserUse(
        browser='cloud' if cloud else 'local',
        scope='agent',
        progress=print,
        progress_detail='code' if verbose else 'steps',
        default_timeout=180.0,
    )
    agent = Agent(model, capabilities=[capability])
    where = 'a cloud browser (started on first use, stops on /exit)' if cloud else 'your local Chrome'
    print(f'--- model: {model}\n--- browser: {where}\n--- /exit to quit\n')

    async with agent:
        history = None
        if opening_task is not None:
            result = await agent.run(opening_task)
            print(f'\n{result.output}\n')
            history = result.all_messages()
        await agent.to_cli(prog_name='browser-use', message_history=history)


def main() -> int:
    """Parse the flags and launch the chat, returning a process exit code."""
    _load_dotenv()
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    args = [a for a in sys.argv[1:] if not a.startswith('--')]

    model = _pick_model()
    if model is None:
        keys = ', '.join(env_var for env_var, _ in _PROVIDERS)
        print(
            'No LLM API key found, so the agent cannot run.\n'
            f'Set one of: {keys} (or MODEL=<provider:name> for something else).',
            file=sys.stderr,
        )
        return 1

    asyncio.run(chat(model, '--cloud' in flags, '--verbose' in flags, ' '.join(args) or None))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
