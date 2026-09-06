import pytest

from pydantic_ai_harness.slack import SlackContext, SlackFile


def context() -> SlackContext:
    return SlackContext(team_id='T1', channel_id='C1', thread_ts='1.1', message_ts='1.2', user_id='U1')


def test_context_is_an_immutable_typed_snapshot() -> None:
    value = context()
    assert value.enterprise_id is None
    assert value.files == ()
    with pytest.raises(AttributeError):
        value.user_id = 'U2'  # type: ignore[misc]


def test_file_is_an_immutable_typed_snapshot() -> None:
    value = SlackFile(file_id='F1', name='a.txt', mimetype='text/plain')
    assert value == SlackFile(file_id='F1', name='a.txt', mimetype='text/plain')
    with pytest.raises(AttributeError):
        value.file_id = 'F2'  # type: ignore[misc]
