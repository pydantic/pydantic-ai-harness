import pytest

from pydantic_ai_harness._workspace_provider import absolute_path


def test_absolute_path_passes_none_and_absolute_paths_through() -> None:
    assert absolute_path('workdir', None) is None
    assert absolute_path('workdir', '/home/user/../project') == '/home/user/../project'


def test_absolute_path_rejects_relative_paths() -> None:
    with pytest.raises(ValueError, match="workdir must be an absolute workspace path or None, got 'project'."):
        absolute_path('workdir', 'project')
