"""Tests for the Skills capability."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_harness.skills import Skill, Skills, load_skills_from_directory
from pydantic_harness.skills._toolset import _parse_skill_markdown

# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------


class TestSkill:
    def test_valid_name(self) -> None:
        s = Skill(name='my-skill', description='A skill')
        assert s.name == 'my-skill'

    def test_valid_single_char_name(self) -> None:
        s = Skill(name='a', description='A skill')
        assert s.name == 'a'

    def test_valid_numeric_name(self) -> None:
        s = Skill(name='s3', description='A skill')
        assert s.name == 's3'

    def test_invalid_name_uppercase(self) -> None:
        with pytest.raises(ValueError, match='lowercase alphanumeric'):
            Skill(name='MySkill', description='bad')

    def test_invalid_name_underscore(self) -> None:
        with pytest.raises(ValueError, match='lowercase alphanumeric'):
            Skill(name='my_skill', description='bad')

    def test_invalid_name_leading_hyphen(self) -> None:
        with pytest.raises(ValueError, match='lowercase alphanumeric'):
            Skill(name='-bad', description='bad')

    def test_invalid_name_trailing_hyphen(self) -> None:
        with pytest.raises(ValueError, match='lowercase alphanumeric'):
            Skill(name='bad-', description='bad')

    def test_invalid_name_empty(self) -> None:
        with pytest.raises(ValueError, match='lowercase alphanumeric'):
            Skill(name='', description='bad')

    def test_tool_names_from_callables(self) -> None:
        def add(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Add two numbers."""
            return a + b

        def subtract(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Subtract two numbers."""
            return a - b

        s = Skill(name='math', description='Math', tools=[add, subtract])
        assert s.tool_names() == ['add', 'subtract']

    def test_tool_names_from_empty(self) -> None:
        s = Skill(name='empty', description='No tools')
        assert s.tool_names() == []

    def test_instructions_default_none(self) -> None:
        s = Skill(name='plain', description='No instructions')
        assert s.instructions is None

    def test_instructions_set(self) -> None:
        s = Skill(name='guided', description='Has instructions', instructions='Do the thing.')
        assert s.instructions == 'Do the thing.'


# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------


class TestParseSkillMarkdown:
    def test_basic(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: my-skill
            description: A useful skill
            ---
            Some instructions here.
        """)
        skill = _parse_skill_markdown(md)
        assert skill.name == 'my-skill'
        assert skill.description == 'A useful skill'
        assert skill.instructions == 'Some instructions here.'

    def test_no_body(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: bare
            description: No body
            ---
        """)
        skill = _parse_skill_markdown(md)
        assert skill.name == 'bare'
        assert skill.instructions is None

    def test_missing_frontmatter(self) -> None:
        with pytest.raises(ValueError, match='Missing YAML frontmatter'):
            _parse_skill_markdown('No frontmatter here')

    def test_unclosed_frontmatter(self) -> None:
        with pytest.raises(ValueError, match='Unclosed YAML frontmatter'):
            _parse_skill_markdown('---\nname: oops\n')

    def test_missing_name(self) -> None:
        md = '---\ndescription: No name\n---\n'
        with pytest.raises(ValueError, match='missing required "name"'):
            _parse_skill_markdown(md)

    def test_missing_description(self) -> None:
        md = '---\nname: no-desc\n---\n'
        with pytest.raises(ValueError, match='missing required "description"'):
            _parse_skill_markdown(md)

    def test_multiline_body(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: multi
            description: Multi-line body
            ---
            Line one.

            Line two.
        """)
        skill = _parse_skill_markdown(md)
        assert 'Line one.' in skill.instructions  # type: ignore[operator]
        assert 'Line two.' in skill.instructions  # type: ignore[operator]

    def test_comment_lines_in_frontmatter(self) -> None:
        md = textwrap.dedent("""\
            ---
            # This is a comment
            name: commented
            description: Has comments
            ---
        """)
        skill = _parse_skill_markdown(md)
        assert skill.name == 'commented'

    def test_lines_without_colon_ignored(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: tolerant
            description: Ignores bad lines
            this line has no colon
            ---
        """)
        skill = _parse_skill_markdown(md)
        assert skill.name == 'tolerant'

    def test_unknown_frontmatter_keys_ignored(self) -> None:
        """agentskills.io compatibility: extra fields like tools, dependencies, etc. are silently ignored."""
        md = textwrap.dedent("""\
            ---
            name: external-skill
            description: From agentskills.io
            tools: [fetch, parse]
            dependencies: [httpx]
            author: someone
            version: 1.0.0
            ---
            Instructions for the skill.
        """)
        skill = _parse_skill_markdown(md)
        assert skill.name == 'external-skill'
        assert skill.description == 'From agentskills.io'
        assert skill.instructions == 'Instructions for the skill.'


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------


class TestLoadSkillsFromDirectory:
    def test_loads_md_files(self, tmp_path: Path) -> None:
        (tmp_path / 'alpha.md').write_text('---\nname: alpha\ndescription: First\n---\nAlpha body')
        (tmp_path / 'beta.md').write_text('---\nname: beta\ndescription: Second\n---\n')

        skills = load_skills_from_directory(tmp_path)
        assert len(skills) == 2
        assert skills[0].name == 'alpha'
        assert skills[1].name == 'beta'

    def test_ignores_non_md(self, tmp_path: Path) -> None:
        (tmp_path / 'readme.txt').write_text('not a skill')
        (tmp_path / 'ok.md').write_text('---\nname: ok\ndescription: Valid\n---\n')

        skills = load_skills_from_directory(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == 'ok'

    def test_empty_directory(self, tmp_path: Path) -> None:
        skills = load_skills_from_directory(tmp_path)
        assert skills == []

    def test_string_path(self, tmp_path: Path) -> None:
        (tmp_path / 'one.md').write_text('---\nname: one\ndescription: Test\n---\n')
        skills = load_skills_from_directory(str(tmp_path))
        assert len(skills) == 1


# ---------------------------------------------------------------------------
# Skills capability
# ---------------------------------------------------------------------------


def _make_tool_def(name: str) -> ToolDefinition:
    return ToolDefinition(name=name, description=f'Tool {name}')


class TestSkillsCapability:
    def test_get_instructions_with_skills(self) -> None:
        cap = Skills(skills=[Skill(name='s1', description='Skill one')])
        instructions = cap.get_instructions()
        assert instructions is not None
        assert 'search_skills' in instructions
        assert 'load_skill' in instructions
        assert 'unload_skill' in instructions

    def test_get_instructions_no_skills(self) -> None:
        cap: Skills[None] = Skills()
        assert cap.get_instructions() is None

    def test_get_toolset_with_tools(self) -> None:
        def add(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Add."""
            return a + b

        cap = Skills(skills=[Skill(name='math', description='Math', tools=[add])])
        toolset = cap.get_toolset()
        assert toolset is not None
        # Should contain meta-tools + skill tools
        assert 'search_skills' in toolset.tools
        assert 'load_skill' in toolset.tools
        assert 'unload_skill' in toolset.tools
        assert 'add' in toolset.tools

    def test_get_toolset_with_function_toolset(self) -> None:
        toolset_in: FunctionToolset[None] = FunctionToolset()

        def multiply(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Multiply."""
            return a * b

        toolset_in.add_function(multiply, takes_ctx=False)
        cap = Skills(skills=[Skill(name='calc', description='Calculator', tools=toolset_in)])
        toolset = cap.get_toolset()
        assert toolset is not None
        assert 'multiply' in toolset.tools

    def test_get_toolset_no_skills(self) -> None:
        cap: Skills[None] = Skills()
        assert cap.get_toolset() is None

    def test_serialization_name(self) -> None:
        assert Skills.get_serialization_name() == 'Skills'


class TestSkillsMetaTools:
    def test_search_skills_matching(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic operations'),
                Skill(name='web', description='Web scraping'),
            ]
        )
        results = cap._search_skills('math')
        assert len(results) == 1
        assert results[0]['name'] == 'math'
        assert results[0]['loaded'] == 'no'

    def test_search_skills_description_match(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='calc', description='Arithmetic operations'),
            ]
        )
        results = cap._search_skills('arithmetic')
        assert len(results) == 1
        assert results[0]['name'] == 'calc'

    def test_search_skills_case_insensitive(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        results = cap._search_skills('MATH')
        assert len(results) == 1

    def test_search_skills_no_match(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        results = cap._search_skills('cooking')
        assert results == []

    def test_search_skills_shows_loaded_status(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        cap._loaded_skill_names.add('math')
        results = cap._search_skills('math')
        assert results[0]['loaded'] == 'yes'

    def test_load_skill_success(self) -> None:
        def add(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Add."""
            return a + b

        cap = Skills(
            skills=[
                Skill(
                    name='math',
                    description='Arithmetic',
                    tools=[add],
                    instructions='Use add for addition.',
                ),
            ]
        )
        result = cap._load_skill('math')
        assert 'math' in cap._loaded_skill_names
        assert "Skill 'math' loaded." in result
        assert 'add' in result
        assert 'Use add for addition.' in result

    def test_load_skill_not_found(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        result = cap._load_skill('nonexistent')
        assert 'not found' in result
        assert 'math' in result  # suggests available skills

    def test_load_skill_no_tools(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='knowledge', description='Just instructions', instructions='Be helpful.'),
            ]
        )
        result = cap._load_skill('knowledge')
        assert "Skill 'knowledge' loaded." in result
        assert 'Be helpful.' in result

    def test_load_skill_no_instructions(self) -> None:
        def add(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Add."""
            return a + b

        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic', tools=[add]),
            ]
        )
        result = cap._load_skill('math')
        assert "Skill 'math' loaded." in result
        assert 'Instructions' not in result

    def test_search_skills_multi_word_ranks_by_match_count(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='web-scraper', description='Scrape web pages'),
                Skill(name='web-api', description='HTTP API client'),
                Skill(name='math', description='Arithmetic operations'),
            ]
        )
        results = cap._search_skills('web scrape')
        assert len(results) == 2
        # web-scraper matches both words, web-api matches only "web"
        assert results[0]['name'] == 'web-scraper'
        assert results[1]['name'] == 'web-api'

    def test_search_skills_empty_query(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        results = cap._search_skills('')
        assert results == []

    def test_search_skills_whitespace_query(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        results = cap._search_skills('   ')
        assert results == []

    def test_search_skills_partial_word_match(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic operations'),
            ]
        )
        # "arith" is a substring of "arithmetic"
        results = cap._search_skills('arith')
        assert len(results) == 1
        assert results[0]['name'] == 'math'

    def test_unload_skill_success(self) -> None:
        def add(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Add."""
            return a + b

        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic', tools=[add]),
            ]
        )
        cap._loaded_skill_names.add('math')
        result = cap._unload_skill('math')
        assert 'math' not in cap._loaded_skill_names
        assert "Skill 'math' unloaded." in result

    def test_unload_skill_not_found(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        result = cap._unload_skill('nonexistent')
        assert 'not found' in result
        assert 'math' in result  # suggests available skills

    def test_unload_skill_not_loaded(self) -> None:
        cap = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        result = cap._unload_skill('math')
        assert 'not currently loaded' in result

    def test_unload_skill_hides_tools_again(self) -> None:
        def add(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Add."""
            return a + b

        cap: Skills[None] = Skills(
            skills=[
                Skill(name='math', description='Arithmetic', tools=[add]),
            ]
        )
        cap._loaded_skill_names.add('math')
        tool_defs = [
            _make_tool_def('search_skills'),
            _make_tool_def('load_skill'),
            _make_tool_def('unload_skill'),
            _make_tool_def('add'),
        ]
        # Tools visible while loaded
        result = asyncio.run(cap.prepare_tools(None, tool_defs))  # type: ignore[arg-type]
        assert 'add' in [td.name for td in result]

        # Unload and verify tools are hidden
        cap._unload_skill('math')
        result = asyncio.run(cap.prepare_tools(None, tool_defs))  # type: ignore[arg-type]
        names = [td.name for td in result]
        assert 'add' not in names
        assert 'unload_skill' in names


class TestSkillsPrepareTools:
    def test_hides_unloaded_skill_tools(self) -> None:
        def add(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Add."""
            return a + b

        cap: Skills[None] = Skills(
            skills=[
                Skill(name='math', description='Arithmetic', tools=[add]),
            ]
        )
        tool_defs = [
            _make_tool_def('search_skills'),
            _make_tool_def('load_skill'),
            _make_tool_def('add'),
        ]
        # Pretend we have a RunContext -- prepare_tools only uses self state
        result = asyncio.run(cap.prepare_tools(None, tool_defs))  # type: ignore[arg-type]
        names = [td.name for td in result]
        assert 'search_skills' in names
        assert 'load_skill' in names
        assert 'add' not in names

    def test_shows_loaded_skill_tools(self) -> None:
        def add(a: int, b: int) -> int:  # pragma: no cover – tool stub
            """Add."""
            return a + b

        cap: Skills[None] = Skills(
            skills=[
                Skill(name='math', description='Arithmetic', tools=[add]),
            ]
        )
        cap._loaded_skill_names.add('math')
        tool_defs = [
            _make_tool_def('search_skills'),
            _make_tool_def('load_skill'),
            _make_tool_def('add'),
        ]
        result = asyncio.run(cap.prepare_tools(None, tool_defs))  # type: ignore[arg-type]
        names = [td.name for td in result]
        assert 'add' in names

    def test_non_skill_tools_always_visible(self) -> None:
        cap: Skills[None] = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        tool_defs = [
            _make_tool_def('search_skills'),
            _make_tool_def('load_skill'),
            _make_tool_def('other_agent_tool'),
        ]
        result = asyncio.run(cap.prepare_tools(None, tool_defs))  # type: ignore[arg-type]
        names = [td.name for td in result]
        assert 'other_agent_tool' in names


class TestSkillsForRun:
    def test_for_run_isolates_state(self) -> None:
        cap: Skills[None] = Skills(
            skills=[
                Skill(name='math', description='Arithmetic'),
            ]
        )
        cap._loaded_skill_names.add('math')

        run_cap = asyncio.run(cap.for_run(None))  # type: ignore[arg-type]
        assert isinstance(run_cap, Skills)
        # New instance should have empty loaded set
        assert len(run_cap._loaded_skill_names) == 0
        # Original should be unchanged
        assert 'math' in cap._loaded_skill_names

    def test_for_run_preserves_skills(self) -> None:
        skill = Skill(name='math', description='Arithmetic')
        cap: Skills[None] = Skills(skills=[skill])
        run_cap = asyncio.run(cap.for_run(None))  # type: ignore[arg-type]
        assert run_cap.skills is cap.skills


class TestSkillsFromSpec:
    def test_from_spec_with_dirs(self, tmp_path: Path) -> None:
        (tmp_path / 'test.md').write_text('---\nname: test\ndescription: Test skill\n---\n')
        cap = Skills.from_spec(dirs=[str(tmp_path)])
        assert len(cap.skills) == 1
        assert cap.skills[0].name == 'test'

    def test_from_spec_with_positional_dirs(self, tmp_path: Path) -> None:
        (tmp_path / 'test.md').write_text('---\nname: test\ndescription: Test skill\n---\n')
        cap = Skills.from_spec(str(tmp_path))
        assert len(cap.skills) == 1

    def test_from_spec_empty(self) -> None:
        cap = Skills.from_spec()
        assert cap.skills == []


class TestSkillToolNamesFromToolset:
    def test_tool_names_from_function_toolset(self) -> None:
        toolset: FunctionToolset[None] = FunctionToolset()

        def greet(name: str) -> str:  # pragma: no cover – tool stub
            """Say hello."""
            return f'Hello, {name}!'

        toolset.add_function(greet, takes_ctx=False)
        s = Skill(name='greeting', description='Greetings', tools=toolset)
        assert s.tool_names() == ['greet']
