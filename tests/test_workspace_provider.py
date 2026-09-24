import pytest
from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceBackend,
    WrapperWorkspace,
)

from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness._warn import warn_argument_renamed
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import absolute_path, check_integer, check_working_dir, command_argv


def test_absolute_path_passes_none_and_absolute_paths_through() -> None:
    assert absolute_path('workdir', None) is None
    assert absolute_path('workdir', '/home/user/../project') == '/home/user/../project'


def test_absolute_path_rejects_relative_paths() -> None:
    with pytest.raises(ValueError, match="workdir must be an absolute workspace path or None, got 'project'."):
        absolute_path('workdir', 'project')


@pytest.mark.parametrize(
    ('command', 'shell', 'argv'),
    [('echo "$HOME"', True, ['/bin/sh', '-c', 'echo "$HOME"']), (('echo', '$HOME'), False, ['echo', '$HOME'])],
)
def test_command_argv(command: str | tuple[str, ...], shell: bool, argv: list[str]) -> None:
    assert command_argv(command, shell) == argv


@pytest.mark.parametrize(
    ('command', 'shell', 'message'),
    [
        ('echo hi', False, 'a string command requires shell=True; pass an argv sequence otherwise'),
        (['echo', 'hi'], True, 'an argv sequence cannot be combined with shell=True; pass a single command string'),
    ],
)
def test_command_argv_rejects_a_mismatched_shell_flag(command: str | list[str], shell: bool, message: str) -> None:
    with pytest.raises(TypeError) as error:
        command_argv(command, shell)
    assert str(error.value) == message


class DurableLikeWorkspace(WrapperWorkspace):
    """A wrapper that refuses `backend`, as a durable workspace does in workflow code."""

    @property
    def backend(self) -> WorkspaceBackend:
        raise AssertionError('`backend` was read on a wrapper')  # pragma: no cover


def test_innermost_backend_unwraps_facades_and_wrappers_through_wrapped() -> None:
    backend = LocalWorkspaceBackend('.')
    assert innermost_backend(Workspace(backend)) is backend
    assert innermost_backend(DurableLikeWorkspace(ReadOnlyWorkspace(Workspace(backend)))) is backend


@pytest.mark.parametrize('value', [None, '/', '/home/user'])
def test_check_working_dir_accepts_absolute_paths_and_none(value: str | None) -> None:
    check_working_dir(value)


def test_check_working_dir_rejects_relative_paths() -> None:
    with pytest.raises(UserError, match=r"^working_dir must be an absolute POSIX path or None, got 'project'\.$"):
        check_working_dir('project')


@pytest.mark.parametrize(
    ('value', 'minimum', 'optional'),
    [(1, 1, False), (0, 0, False), (None, 1, True)],
)
def test_check_integer_accepts(value: int | None, minimum: int, optional: bool) -> None:
    check_integer('timeout', value, minimum=minimum, optional=optional)


@pytest.mark.parametrize(
    ('value', 'optional', 'message'),
    [
        (0, False, 'timeout must be an integer of at least 1, got 0.'),
        (None, False, 'timeout must be an integer of at least 1, got None.'),
        (True, False, 'timeout must be an integer of at least 1, got True.'),
        (1.5, True, 'timeout must be an integer of at least 1 or None, got 1.5.'),
    ],
)
def test_check_integer_rejects(value: object, optional: bool, message: str) -> None:
    with pytest.raises(UserError) as error:
        check_integer('timeout', value, optional=optional)  # pyright: ignore[reportArgumentType]
    assert str(error.value) == message


def test_warn_argument_renamed_points_at_the_new_name() -> None:
    with pytest.warns(HarnessDeprecationWarning) as record:
        warn_argument_renamed('Sandbox', 'workdir', 'working_dir', stacklevel=2)
    assert str(record[0].message) == (
        '`Sandbox(workdir=...)` has been renamed to `Sandbox(working_dir=...)`. '
        'Update the call; this deprecated alias will be removed in a future release.'
    )
    assert record[0].filename == __file__
