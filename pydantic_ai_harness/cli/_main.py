"""The CLI agent, argument parsing, and the entry point for both the session and one-shot modes."""

import argparse
import asyncio
import sys
from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.cli._bridge import CliBridge
from pydantic_ai_harness.cli._config import Config
from pydantic_ai_harness.cli._repl import Lines, Repl
from pydantic_ai_harness.coder import Coder

cli_agent: Agent[None, str] = Agent(
    name='harness',
    instructions='You are a coding agent built on Pydantic AI.',
    capabilities=[Coder(), CliBridge()],
)
"""Model-less agent the CLI drives: `Coder` rooted at the current directory, rendered by `CliBridge`.

The model comes from the config file or `--model` at run time, so tests swap it with
`cli_agent.override(model=...)`. The bridge reads the same config file when a run starts.
Later plan items insert the CLI's own capabilities before the bridge, which stays last so it
observes every other capability's events.
"""


async def _session(*, model: str, prompt: str | None) -> None:
    if prompt is None:
        await Repl(agent=cli_agent, model=model, lines=Lines.from_stdin(), output=sys.stdout).run()
    else:
        await Repl(agent=cli_agent, model=model, output=sys.stdout).run_once(prompt)


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point. `argv` defaults to `sys.argv[1:]`."""
    parser = argparse.ArgumentParser(prog='harness', description='Pydantic AI coding agent for the terminal.')
    parser.add_argument(
        '-p', '--prompt', help='Run one prompt, print the response, and exit instead of starting a session.'
    )
    parser.add_argument(
        '--model',
        help=f'Model in Pydantic AI `provider:name` form. Overrides the config file at {Config.default_path()}.',
    )
    args = parser.parse_args(argv)
    try:
        model = args.model or Config.load().model
        asyncio.run(_session(model=model, prompt=args.prompt))
    except UserError as exc:
        parser.error(str(exc))
