"""Run a configured Pydantic AI agent for the composite action."""

from __future__ import annotations

import importlib
import os
import secrets
import sys
from pathlib import Path
from typing import TypeGuard

from pydantic_ai import Agent


def _is_agent(value: object) -> TypeGuard[Agent[object, object]]:
    return isinstance(value, Agent)


def _resolve_agent(target: str) -> Agent[object, object]:
    if Path(target).suffix.lower() in {'.yml', '.yaml', '.json'}:
        agent: Agent[object, object] = Agent.from_file(target)
        return agent

    module_name, separator, variable_name = target.partition(':')
    if not separator or not module_name or not variable_name:
        raise ValueError('expected a module:variable target or a .yml, .yaml, or .json spec file')

    module = importlib.import_module(module_name)
    value: object = getattr(module, variable_name)
    if not _is_agent(value):
        value_type = f'{type(value).__module__}.{type(value).__qualname__}'
        raise TypeError(f'{target!r} resolved to {value_type}, expected pydantic_ai.Agent')
    return value


def _read_prompt() -> str:
    prompt = os.environ.get('PAI_PROMPT', '')
    prompt_file = os.environ.get('PAI_PROMPT_FILE', '')
    if bool(prompt) == bool(prompt_file):
        raise ValueError('exactly one of prompt and prompt-file must be provided')

    if prompt_file:
        try:
            prompt = Path(prompt_file).read_text(encoding='utf-8')
        except OSError as error:
            raise ValueError(f'could not read prompt file {prompt_file!r}: {error}') from error
        if not prompt:
            raise ValueError(f'prompt file {prompt_file!r} is empty')

    return prompt


def _append_github_output(path: str, output: str) -> None:
    output_lines = set(output.splitlines())
    delimiter = f'EOF_{secrets.token_hex(16)}'
    while delimiter in output_lines:
        delimiter = f'EOF_{secrets.token_hex(16)}'

    with Path(path).open('a', encoding='utf-8') as output_file:
        output_file.write(f'result<<{delimiter}\n{output}\n{delimiter}\n')


def _append_step_summary(path: str, output: str) -> None:
    with Path(path).open('a', encoding='utf-8') as summary_file:
        summary_file.write(output)
        if not output.endswith('\n'):
            summary_file.write('\n')


def main() -> int:
    """Resolve the action inputs, run the agent, and publish its output."""
    model = os.environ.get('PAI_MODEL', '').strip()
    if not model:
        print('error: model is required', file=sys.stderr)
        return 2

    try:
        prompt = _read_prompt()
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return 2

    target = os.environ.get('PAI_AGENT', '').strip()
    if not target:
        print('error: agent target is required', file=sys.stderr)
        return 2

    try:
        agent = _resolve_agent(target)
    except Exception as error:
        print(f'error: could not resolve agent target {target!r}: {error}', file=sys.stderr)
        return 2

    try:
        output = str(agent.run_sync(prompt, model=model).output)
    except Exception as error:
        print(f'error: agent run failed: {error}', file=sys.stderr)
        return 1

    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        try:
            _append_github_output(github_output, output)
        except OSError as error:
            print(f'error: could not write GITHUB_OUTPUT: {error}', file=sys.stderr)
            return 1

    step_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if step_summary:
        try:
            _append_step_summary(step_summary, output)
        except OSError as error:
            print(f'error: could not write GITHUB_STEP_SUMMARY: {error}', file=sys.stderr)
            return 1

    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
