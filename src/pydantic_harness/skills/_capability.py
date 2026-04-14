"""Skills capability -- progressive skill discovery and loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_harness.skills._toolset import Skill, load_skills_from_directory


@dataclass
class Skills(AbstractCapability[AgentDepsT]):
    """Capability for progressive skill discovery and loading.

    Provides ``search_skills``, ``load_skill``, and ``unload_skill``
    meta-tools.  Tools belonging to registered skills are hidden until
    the agent explicitly loads the skill that owns them.

    Per-run state (which skills are loaded) is isolated via
    :meth:`for_run`.

    Example::

        from pydantic_ai import Agent
        from pydantic_harness.skills import Skill, Skills

        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        math_skill = Skill(
            name='math',
            description='Basic arithmetic operations',
            tools=[add],
        )
        agent = Agent('openai:gpt-4o', capabilities=[Skills(skills=[math_skill])])
    """

    skills: list[Skill] = field(default_factory=lambda: list[Skill]())
    """Registered skills."""

    _loaded_skill_names: set[str] = field(default_factory=lambda: set[str](), init=False, repr=False)
    """Names of skills that have been loaded in the current run (per-run state)."""

    @classmethod
    def get_serialization_name(cls) -> str | None:  # noqa: D102
        return 'Skills'

    @classmethod
    def from_spec(cls, *args: Any, **kwargs: Any) -> Skills[Any]:
        """Create from spec arguments.

        Accepts ``dirs`` (list of directory paths) to load markdown skills.
        """
        dirs: list[str] = kwargs.pop('dirs', []) or list(args)
        all_skills: list[Skill] = []
        for d in dirs:
            all_skills.extend(load_skills_from_directory(d))
        return cls(skills=all_skills, **kwargs)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Skills[AgentDepsT]:
        """Return a fresh copy with empty loaded-skills state."""
        clone: Skills[AgentDepsT] = Skills(skills=self.skills)
        return clone

    def get_instructions(self) -> str | None:
        """Provide baseline instructions for skill discovery."""
        if not self.skills:
            return None
        return (
            'You have access to a skill catalog. '
            'Use `search_skills` to find relevant skills by keyword, '
            'then `load_skill` to activate a skill and make its tools available. '
            'Use `unload_skill` when you no longer need a skill, to free context. '
            "Only loaded skills' tools appear in your tool list."
        )

    def get_toolset(self) -> FunctionToolset[AgentDepsT] | None:
        """Build the toolset containing meta-tools and all skill tools."""
        if not self.skills:
            return None

        toolset: FunctionToolset[AgentDepsT] = FunctionToolset()

        # Register meta-tools
        toolset.add_function(self._search_skills, takes_ctx=False, name='search_skills')
        toolset.add_function(self._load_skill, takes_ctx=False, name='load_skill')
        toolset.add_function(self._unload_skill, takes_ctx=False, name='unload_skill')

        # Register each skill's tools (they will be hidden until loaded)
        for skill in self.skills:
            if isinstance(skill.tools, FunctionToolset):
                for tool in skill.tools.tools.values():
                    toolset.add_tool(tool)
            else:
                for fn in skill.tools:
                    toolset.add_function(fn, takes_ctx=False)

        return toolset

    async def prepare_tools(
        self,
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        """Hide tools belonging to skills that have not been loaded yet."""
        # Build set of tool names that should be hidden
        hidden: set[str] = set()
        for skill in self.skills:
            if skill.name not in self._loaded_skill_names:
                hidden.update(skill.tool_names())

        # Always keep meta-tools visible
        meta_tools = {'search_skills', 'load_skill', 'unload_skill'}

        return [td for td in tool_defs if td.name in meta_tools or td.name not in hidden]

    # -- Meta-tool implementations --

    def _search_skills(self, query: str) -> list[dict[str, str]]:
        """Search available skills by keyword.

        Returns a list of matching skills with their name, description,
        and whether they are currently loaded, ranked by relevance.

        The query is split into words and each word is matched
        case-insensitively against the skill name and description.
        Skills matching at least one word are returned, ordered by the
        number of matching words (most relevant first).

        Args:
            query: A keyword or phrase to search for in skill names and descriptions.
        """
        words = query.lower().split()
        if not words:
            return []

        scored: list[tuple[int, Skill]] = []
        for skill in self.skills:
            haystack = f'{skill.name} {skill.description}'.lower()
            matches = sum(1 for w in words if w in haystack)
            if matches:
                scored.append((matches, skill))

        # Sort by match count descending, then by name for stability
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))

        return [
            {
                'name': skill.name,
                'description': skill.description,
                'loaded': 'yes' if skill.name in self._loaded_skill_names else 'no',
            }
            for _, skill in scored
        ]

    def _load_skill(self, name: str) -> str:
        """Load a skill by name, making its tools available.

        After loading, the skill's tools will appear in subsequent tool
        lists and any associated instructions will be included.

        Args:
            name: The exact name of the skill to load (as returned by ``search_skills``).
        """
        skill = self._find_skill(name)
        if skill is None:
            available = ', '.join(s.name for s in self.skills)
            return f'Skill {name!r} not found. Available skills: {available}'

        self._loaded_skill_names.add(name)

        parts = [f'Skill {name!r} loaded.']
        tool_names = skill.tool_names()
        if tool_names:
            parts.append(f'Available tools: {", ".join(tool_names)}')
        if skill.instructions:
            parts.append(f'Instructions:\n{skill.instructions}')
        return '\n'.join(parts)

    def _unload_skill(self, name: str) -> str:
        """Unload a skill by name, removing its tools from the context.

        Use this when you no longer need a skill's tools, to free up
        space in the context window.

        Args:
            name: The exact name of the skill to unload.
        """
        skill = self._find_skill(name)
        if skill is None:
            available = ', '.join(s.name for s in self.skills)
            return f'Skill {name!r} not found. Available skills: {available}'

        if name not in self._loaded_skill_names:
            return f'Skill {name!r} is not currently loaded.'

        self._loaded_skill_names.discard(name)
        return f'Skill {name!r} unloaded. Its tools are no longer available.'

    # -- Helpers --

    def _find_skill(self, name: str) -> Skill | None:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None
