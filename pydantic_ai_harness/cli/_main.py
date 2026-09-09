"""The CLI agent, argument parsing, and the entry point for both the session and one-shot modes."""

import argparse
import asyncio
import sys
from collections.abc import Sequence

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.cli._approve import Approver, CliDeps, DeclineAll, allow_all
from pydantic_ai_harness.cli._bridge import CliBridge
from pydantic_ai_harness.cli._config import Config
from pydantic_ai_harness.cli._repl import Lines, Repl
from pydantic_ai_harness.coder import Coder

cli_agent: Agent[CliDeps, str] = Agent(
    name='harness',
    deps_type=CliDeps,
    instructions='You are a coding agent built on Pydantic AI.',
    capabilities=[Coder(), CliBridge()],
)
"""Model-less agent the CLI drives: `Coder` rooted at the current directory, rendered by `CliBridge`.

The model comes from the config file or `--model` at run time, so tests swap it with
`cli_agent.override(model=...)`. The bridge reads the same config file when a run starts and
takes the approver from the run's `CliDeps`. Later plan items insert the CLI's own capabilities
before the bridge, which stays last so it observes every other capability's events.
"""

NO_TERMINAL = DeclineAll(reason='one-shot mode has no terminal to ask; run with --yolo to allow')


async def _session(*, model: str, prompt: str | None, yolo: bool) -> None:
    if prompt is None:
        approver: Approver | None = allow_all if yolo else None
        await Repl(agent=cli_agent, model=model, lines=Lines.from_stdin(), output=sys.stdout, approver=approver).run()
    else:
        approver = allow_all if yolo else NO_TERMINAL
        await Repl(agent=cli_agent, model=model, output=sys.stdout, approver=approver).run_once(prompt)


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
    parser.add_argument(
        '--yolo',
        action='store_true',
        help='Approve every shell command and file change without asking. Overrides `yolo` in the config file.',
    )
    args = parser.parse_args(argv)
    try:
        config = Config.load()
        model = args.model or config.model
        asyncio.run(_session(model=model, prompt=args.prompt, yolo=args.yolo or config.yolo))
    except UserError as exc:
        parser.error(str(exc))
