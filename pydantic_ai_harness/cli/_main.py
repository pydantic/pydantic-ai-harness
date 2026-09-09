"""The CLI agent, argument parsing, and the one-shot prompt mode."""

import argparse
import asyncio
from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.cli._bridge import CliBridge
from pydantic_ai_harness.coder import Coder

DEFAULT_MODEL = 'anthropic:claude-fable-5'

cli_agent: Agent[None, str] = Agent(
    name='harness',
    instructions='You are a coding agent built on Pydantic AI.',
    capabilities=[Coder(), CliBridge()],
)
"""Model-less agent the CLI drives: `Coder` rooted at the current directory, rendered by `CliBridge`.

The model comes from `--model` at run time, so tests swap it with `cli_agent.override(model=...)`.
Later plan items insert the CLI's own capabilities before the bridge, which stays last so it
observes every other capability's events.
"""


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point. `argv` defaults to `sys.argv[1:]`."""
    parser = argparse.ArgumentParser(prog='harness', description='Pydantic AI coding agent for the terminal.')
    parser.add_argument('-p', '--prompt', required=True, help='Run one prompt, print the response, and exit.')
    parser.add_argument(
        '--model', default=DEFAULT_MODEL, help=f'Model in Pydantic AI `provider:name` form (default: {DEFAULT_MODEL}).'
    )
    args = parser.parse_args(argv)
    try:
        asyncio.run(cli_agent.run(args.prompt, model=args.model))
    except UserError as exc:
        parser.error(str(exc))
