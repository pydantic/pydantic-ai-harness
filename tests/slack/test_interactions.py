from __future__ import annotations

import anyio
import pytest

from pydantic_ai_harness.slack import PROMPT_ACTION_PREFIX, SlackInteractions, SlackPromptError, SlackThread

from .conftest import FakeSlackClient, prompt_block_id, prompt_buttons

pytestmark = pytest.mark.anyio


async def _answer(interactions: SlackInteractions, client: FakeSlackClient, value: str, user_id: str) -> bool:
    # The prompt is only registered once the message has been posted.
    while not client.method_calls('chat_postMessage'):
        await anyio.sleep(0)
    return interactions.resolve(block_id=prompt_block_id(client), value=value, user_id=user_id)


class TestAsk:
    async def test_returns_the_clicked_option(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions()
        answers: list[str | None] = []

        async with anyio.create_task_group() as tg:

            async def ask() -> None:
                answers.append(await interactions.ask(thread, 'Ship it?', ['Yes', 'No']))

            tg.start_soon(ask)
            assert await _answer(interactions, slack_client, 'Yes', 'U0ASKER')

        assert answers == ['Yes']

    async def test_posts_one_button_per_option_into_the_thread(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: interactions.ask(thread, 'Pick', ['A', 'B', 'C']))
            await _answer(interactions, slack_client, 'B', 'U0ASKER')

        post = slack_client.method_calls('chat_postMessage')[0]
        assert post.kwargs['thread_ts'] == thread.thread_ts
        buttons = prompt_buttons(slack_client)
        assert [button['value'] for button in buttons] == ['A', 'B', 'C']
        assert all(str(button['action_id']).startswith(PROMPT_ACTION_PREFIX) for button in buttons)

    async def test_edits_the_message_to_record_the_answer(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: interactions.ask(thread, 'Ship it?', ['Yes', 'No']))
            await _answer(interactions, slack_client, 'No', 'U0ASKER')

        update = slack_client.method_calls('chat_update')[0]
        assert update.kwargs['blocks'] == []
        assert 'chose *No*' in str(update.kwargs['text'])

    async def test_returns_none_and_says_so_when_nobody_answers(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions(timeout_seconds=0.01)
        assert await interactions.ask(thread, 'Ship it?', ['Yes', 'No']) is None
        assert 'expired' in str(slack_client.method_calls('chat_update')[0].kwargs['text'])

    async def test_raises_when_slack_returns_no_timestamp(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        slack_client.recorder.post_response = {'ok': True}
        with pytest.raises(SlackPromptError, match='did not return a timestamp'):
            await SlackInteractions(timeout_seconds=0.01).ask(thread, 'Ship it?', ['Yes'])

    async def test_an_answer_survives_a_failure_to_tidy_the_buttons_away(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        slack_client.recorder.update_error = RuntimeError('slack is having a moment')
        interactions = SlackInteractions()
        answers: list[str | None] = []

        async with anyio.create_task_group() as tg:

            async def ask() -> None:
                answers.append(await interactions.ask(thread, 'Ship it?', ['Yes', 'No']))

            tg.start_soon(ask)
            assert await _answer(interactions, slack_client, 'Yes', 'U0ASKER')

        assert answers == ['Yes']

    async def test_each_prompt_gets_its_own_unguessable_id(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        # A counter would restart with the process, so a button left over from a
        # previous run could answer the first prompt of the next one.
        interactions = SlackInteractions(timeout_seconds=0.01)
        await interactions.ask(thread, 'First?', ['Yes'])
        await interactions.ask(thread, 'Second?', ['Yes'])
        first, second = prompt_block_id(slack_client, 0), prompt_block_id(slack_client, 1)
        assert first != second
        assert thread.key not in first

    async def test_second_prompt_waits_for_the_first(self, thread: SlackThread, slack_client: FakeSlackClient) -> None:
        interactions = SlackInteractions(timeout_seconds=0.01)
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: interactions.ask(thread, 'First?', ['Yes']))
            tg.start_soon(lambda: interactions.ask(thread, 'Second?', ['Yes']))

        posted = [str(call.kwargs['text']) for call in slack_client.method_calls('chat_postMessage')]
        assert posted == ['First?', 'Second?']


class TestAskValidation:
    @pytest.mark.parametrize(
        'options,message',
        [
            ([], 'at least one option'),
            (['A'] * 26, 'at most 25 buttons'),
            (['A', 'A'], 'must be unique'),
            ([''], 'between 1 and 75'),
            (['x' * 76], 'between 1 and 75'),
        ],
    )
    async def test_rejects_unusable_options(self, thread: SlackThread, options: list[str], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            await SlackInteractions().ask(thread, 'Pick', options)

    async def test_rejects_an_allowlist_passed_as_a_string(self, thread: SlackThread) -> None:
        with pytest.raises(ValueError, match='not a string'):
            await SlackInteractions().ask(thread, 'Pick', ['A'], allowed_user_ids='U0REVIEWER')

    def test_rejects_a_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match='timeout_seconds must be positive'):
            SlackInteractions(timeout_seconds=0)


class TestResolve:
    async def test_ignores_an_unknown_prompt(self) -> None:
        assert SlackInteractions().resolve(block_id='nope', value='Yes', user_id='U0ASKER') is False

    @pytest.mark.parametrize(
        'value,user_id',
        [('Maybe', 'U0ASKER'), ('Yes', 'U0OTHER')],
    )
    async def test_ignores_an_unusable_click(
        self, thread: SlackThread, slack_client: FakeSlackClient, value: str, user_id: str
    ) -> None:
        interactions = SlackInteractions(timeout_seconds=0.05)
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: interactions.ask(thread, 'Ship it?', ['Yes', 'No']))
            assert await _answer(interactions, slack_client, value, user_id) is False

    async def test_ignores_a_second_click_on_an_answered_prompt(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        async with anyio.create_task_group() as tg:
            tg.start_soon(lambda: interactions.ask(thread, 'Ship it?', ['Yes', 'No']))
            while not slack_client.method_calls('chat_postMessage'):
                await anyio.sleep(0)
            block_id = prompt_block_id(slack_client)
            assert interactions.resolve(block_id=block_id, value='Yes', user_id='U0ASKER') is True
            assert interactions.resolve(block_id=block_id, value='No', user_id='U0ASKER') is False

    async def test_a_named_group_can_answer_instead_of_the_asker(
        self, thread: SlackThread, slack_client: FakeSlackClient
    ) -> None:
        interactions = SlackInteractions()
        answers: list[str | None] = []

        async with anyio.create_task_group() as tg:

            async def ask() -> None:
                answers.append(await interactions.ask(thread, 'Ship it?', ['Yes'], allowed_user_ids=['U0REVIEWER']))

            tg.start_soon(ask)
            assert await _answer(interactions, slack_client, 'Yes', 'U0ASKER') is False
            assert await _answer(interactions, slack_client, 'Yes', 'U0REVIEWER') is True

        assert answers == ['Yes']
