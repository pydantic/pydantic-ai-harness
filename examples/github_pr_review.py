"""Review a GitHub pull request through GitHub's official remote MCP server.

Set `GITHUB_TOKEN`, `GITHUB_REPOSITORY`, and your model provider credential, then run:

    uv run python examples/github_pr_review.py 123
"""

import os
import sys

from pydantic_ai import Agent
from pydantic_ai.models import Model

from pydantic_ai_harness.github import GitHub

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'openai:gpt-5.6-sol')


def build_agent(model: Model | str = DEFAULT_MODEL, *, github: GitHub[None] | None = None) -> Agent[None, str]:
    """Build a read-only pull request reviewer for one repository."""
    if github is None:
        github = GitHub[None](
            repository=os.environ['GITHUB_REPOSITORY'],
            auth=os.environ['GITHUB_TOKEN'],
            toolsets=('repos', 'pull_requests'),
        )
    return Agent(
        model,
        deps_type=type(None),
        instructions='Review the requested pull request. Cite file paths and line numbers for every finding.',
        capabilities=[github],
    )


def main() -> None:
    """Review the pull request number passed on the command line."""
    if len(sys.argv) != 2:
        raise SystemExit('usage: python examples/github_pr_review.py PR_NUMBER')
    result = build_agent().run_sync(f'Review pull request #{sys.argv[1]}')
    print(result.output)


if __name__ == '__main__':
    main()
