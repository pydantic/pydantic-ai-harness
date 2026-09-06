import subprocess
import sys
import textwrap
from dataclasses import fields

import pytest

import pydantic_ai_harness.slack as slack
from pydantic_ai_harness.slack import SlackContext, SlackFile


def context() -> SlackContext:
    return SlackContext(team_id='T1', channel_id='C1', thread_ts='1.1', message_ts='1.2', user_id='U1')


def test_context_is_an_immutable_typed_snapshot() -> None:
    value = context()
    assert tuple(field.name for field in fields(SlackContext)) == (
        'team_id',
        'channel_id',
        'thread_ts',
        'message_ts',
        'user_id',
        'enterprise_id',
        'files',
    )
    assert value.enterprise_id is None
    assert value.files == ()
    with pytest.raises(AttributeError):
        value.user_id = 'U2'  # type: ignore[misc]


def test_file_is_an_immutable_typed_snapshot() -> None:
    value = SlackFile(file_id='F1', name='a.txt', mimetype='text/plain')
    assert tuple(field.name for field in fields(SlackFile)) == ('file_id', 'name', 'mimetype')
    assert value == SlackFile(file_id='F1', name='a.txt', mimetype='text/plain')
    with pytest.raises(AttributeError):
        value.file_id = 'F2'  # type: ignore[misc]


def test_context_has_no_derived_public_api() -> None:
    value = context()
    assert not hasattr(value, 'thread_id')
    assert not hasattr(value, 'conversation_id')


def test_register_slack_is_an_optional_lazy_export() -> None:
    script = textwrap.dedent(
        """\
        import importlib.abc
        import sys

        class BlockSlackBolt(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'slack_bolt' or fullname.startswith('slack_bolt.'):
                    raise ModuleNotFoundError('blocked slack_bolt')
                return None

        sys.meta_path.insert(0, BlockSlackBolt())
        import pydantic_ai_harness.slack as slack
        from pydantic_ai_harness.slack import Slack, SlackContext, SlackFile

        assert Slack is not None
        assert SlackContext is not None
        assert SlackFile is not None
        try:
            slack.register_slack
        except ImportError as exc:
            assert str(exc) == (
                'slack-bolt is required for register_slack. '
                'Install it with: pip install "pydantic-ai-harness[slack]"'
            )
        else:
            raise AssertionError('register_slack unexpectedly imported without slack-bolt')
        """
    )
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


def test_unknown_names_fail() -> None:
    with pytest.raises(AttributeError):
        _ = slack.not_an_export  # type: ignore[attr-defined]
