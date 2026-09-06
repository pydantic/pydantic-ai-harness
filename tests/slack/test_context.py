import pytest

from pydantic_ai_harness.slack import SlackContext, SlackFile, current_slack_context


def context() -> SlackContext:
    return SlackContext(team_id='T1', channel_id='C1', thread_ts='1.1', message_ts='1.2', user_id='U1')


def test_context_defaults_and_conversation_id() -> None:
    value = context()
    assert value.enterprise_id is None
    assert value.files == ()
    assert value.conversation_id == 'T1:C1:1.1'
    assert current_slack_context() is None


def test_file_and_context_are_frozen() -> None:
    value = SlackFile(file_id='F1', name='a.txt', mimetype='text/plain')
    assert value == SlackFile(file_id='F1', name='a.txt', mimetype='text/plain')
    with pytest.raises(AttributeError):
        value.file_id = 'F2'  # type: ignore[misc]


@pytest.mark.parametrize('field', ('team_id', 'channel_id', 'thread_ts', 'message_ts', 'user_id'))
def test_required_ids_must_be_non_empty_strings(field: str) -> None:
    values = {'team_id': 'T1', 'channel_id': 'C1', 'thread_ts': '1.1', 'message_ts': '1.2', 'user_id': 'U1'}
    values[field] = ''
    with pytest.raises(ValueError):
        SlackContext(**values)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('value', ('', 1, None))
def test_file_id_must_be_non_empty_string(value: object) -> None:
    with pytest.raises(ValueError):
        SlackFile(file_id=value)  # type: ignore[arg-type]


@pytest.mark.parametrize('field', ('name', 'mimetype'))
def test_file_optional_metadata_must_be_strings(field: str) -> None:
    with pytest.raises(ValueError):
        SlackFile(file_id='F1', **{field: 1})  # type: ignore[arg-type]


def test_enterprise_id_must_be_string() -> None:
    with pytest.raises(ValueError):
        SlackContext(  # pyright: ignore[reportArgumentType]
            team_id='T1',
            channel_id='C1',
            thread_ts='1.1',
            message_ts='1.2',
            user_id='U1',
            enterprise_id=1,  # pyright: ignore[reportArgumentType]
        )


def test_files_must_be_tuple_of_slack_files() -> None:
    with pytest.raises(ValueError):
        SlackContext(  # pyright: ignore[reportArgumentType]
            team_id='T1',
            channel_id='C1',
            thread_ts='1.1',
            message_ts='1.2',
            user_id='U1',
            files=[SlackFile(file_id='F1')],  # pyright: ignore[reportArgumentType]
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SlackContext(  # pyright: ignore[reportArgumentType]
            team_id='T1',
            channel_id='C1',
            thread_ts='1.1',
            message_ts='1.2',
            user_id='U1',
            files=('not-a-file',),  # pyright: ignore[reportArgumentType]
        )  # type: ignore[arg-type]
