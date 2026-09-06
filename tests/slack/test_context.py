import subprocess
import sys
import textwrap

import pytest

import pydantic_ai_harness.slack as slack
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


def test_bolt_is_required_only_for_hosting() -> None:
    script = textwrap.dedent(
        """\
        import sys

        sys.modules['slack_bolt'] = None
        from pydantic_ai_harness.slack import Slack

        Slack(token='test-user-token').get_toolset()
        try:
            from pydantic_ai_harness.slack import register_slack
        except ImportError as exc:
            assert 'slack-bolt is required for register_slack' in str(exc)
        else:
            raise AssertionError('hosting imported without Bolt')
        """
    )
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_unknown_export_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(slack, 'not_an_export')
