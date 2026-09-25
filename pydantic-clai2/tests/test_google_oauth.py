"""Sign in with Google: a real loopback callback, a mocked token endpoint, and refresh on later turns."""

import asyncio
import base64
import hashlib
import inspect
import io
from urllib.parse import parse_qs, urlparse

import httpx
import keyring
import pytest
from menu_script import pick, typed
from pydantic import JsonValue, ValidationError
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.google_workspace import GoogleWorkspace
from rich.console import Console
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import api_keys, google_workspace
from pydantic_clai2.api_keys import KeyReference
from pydantic_clai2.credential_store import load_codex_credentials
from pydantic_clai2.field_menu import Runners
from pydantic_clai2.google_oauth import GoogleOAuth, SignedIn, scopes_for
from pydantic_clai2.plugins import PluginHost

CLIENT_ID = '123-abc.apps.googleusercontent.com'
DEFAULT_SCOPES = scopes_for(['gmail', 'calendar', 'drive'])


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Google:
    """A browser that completes consent on the loopback callback, and a scripted token endpoint."""

    def __init__(self, responses: list[httpx.Response], *, callback: str = 'code=auth-code') -> None:
        self.responses = iter(responses)
        self.callback = callback
        self.urls: list[str] = []
        self.forms: list[dict[str, str]] = []
        self.early: list[int] = []

    def browser(self, url: str) -> bool:
        self.urls.append(url)
        params = parse_qs(urlparse(url).query)
        redirect, state = params['redirect_uri'][0], params['state'][0]
        # Forged requests without this login's state are answered but do not end the login.
        self.early.append(httpx.get(f'{redirect}?code=forged&state=other').status_code)
        self.early.append(httpx.get(f'{redirect}?error=access_denied').status_code)
        httpx.get(f'{redirect}?{self.callback}&state={state}')
        return True

    def token(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == 'https://oauth2.googleapis.com/token'
        self.forms.append({name: values[0] for name, values in parse_qs(request.content.decode()).items()})
        return next(self.responses)


async def never_pasted(message: str) -> str:
    await asyncio.Future[None]()
    raise AssertionError('unreachable')


class Prompt:
    def __init__(self, *values: str) -> None:
        self.values = iter(values)

    async def prompt_async(self, label: str, *, is_password: bool = False) -> str:
        assert is_password
        return next(self.values)


def tokens(access: str, *, refresh: str | None = 'refresh-1', scopes: list[str] = DEFAULT_SCOPES) -> httpx.Response:
    body: dict[str, JsonValue] = {'access_token': access, 'expires_in': 3600, 'scope': ' '.join(scopes)}
    if refresh is not None:
        body['refresh_token'] = refresh
    return httpx.Response(200, json=body)


def plugin_with(
    monkeypatch: pytest.MonkeyPatch,
    google: Google,
    settings: dict[str, JsonValue] | None = None,
    *,
    secrets: tuple[str, ...] = ('client-secret',),
) -> tuple[PluginHost[None], GoogleOAuth, Clock]:
    clock = Clock()
    oauth = GoogleOAuth(
        console=Console(file=io.StringIO()),
        read_line=never_pasted,
        open_browser=google.browser,
        transport=httpx.MockTransport(google.token),
        clock=clock,
    )

    def shared(*, console: Console) -> GoogleOAuth:
        return oauth

    monkeypatch.setattr(google_workspace, 'GoogleOAuth', shared)
    prompt = Prompt(*secrets)
    monkeypatch.setattr(google_workspace, 'PromptSession', lambda: prompt)
    plugin = PluginHost[None](name='google_workspace', console=Console(file=io.StringIO()), settings=settings or {})
    google_workspace.activate(plugin)
    return plugin, oauth, clock


async def menu(plugin: PluginHost[None], oauth: GoogleOAuth, *rows: str, text: str | None = None) -> str:
    lists = iter([*(pick(row) for row in rows), MenuResult(cancelled=True)])
    texts = iter([] if text is None else [typed(text)])
    runners = Runners(run_list=lambda _: next(lists), run_choice=lambda _: next(lists), run_text=lambda _: next(texts))
    return await google_workspace.configure(plugin, [], oauth=oauth, runners=runners)


async def token_for(plugin: PluginHost[None]) -> str | None:
    [factory] = plugin.capabilities
    assert not isinstance(factory, AbstractCapability)
    context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    built = factory(context)
    capability = await built if inspect.isawaitable(built) else built
    assert isinstance(capability, GoogleWorkspace) and callable(capability.auth)
    return capability.auth(context)


def test_scopes_follow_googles_list_per_product() -> None:
    assert scopes_for(['docs', 'drive']) == [
        'https://www.googleapis.com/auth/documents',
        'https://www.googleapis.com/auth/documents.readonly',
        'https://www.googleapis.com/auth/drive.file',
        'https://www.googleapis.com/auth/drive.readonly',
    ]


async def test_sign_in_keeps_only_names_outside_keys_and_refreshes_later_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    google = Google([tokens('access-1'), tokens('access-2', refresh=None), tokens('access-3', refresh='refresh-2')])
    plugin, oauth, clock = plugin_with(monkeypatch, google)
    result = await menu(plugin, oauth, 'client_id', 'sign_in', text=CLIENT_ID)
    assert result.splitlines() == [
        'Saved OAuth client ID.',
        'Signed in with Google. The refresh token is saved in /keys as GOOGLE_REFRESH_TOKEN.',
    ]
    assert google.early == [400, 400]
    [url] = google.urls
    params = parse_qs(urlparse(url).query)
    assert params['client_id'] == [CLIENT_ID]
    assert params['scope'] == [' '.join(DEFAULT_SCOPES)]
    assert (params['access_type'], params['prompt'], params['code_challenge_method']) == (
        ['offline'],
        ['consent'],
        ['S256'],
    )
    [exchange] = google.forms
    challenge = base64.urlsafe_b64encode(hashlib.sha256(exchange['code_verifier'].encode()).digest()).rstrip(b'=')
    assert params['code_challenge'] == [challenge.decode()]
    assert exchange['code'] == 'auth-code'
    assert exchange['redirect_uri'] == params['redirect_uri'][0]
    assert exchange['client_secret'] == 'client-secret'

    keys = api_keys.load_keys()
    assert keys['GOOGLE_CLIENT_SECRET'].get_secret_value() == 'client-secret'
    assert keys['GOOGLE_REFRESH_TOKEN'].get_secret_value() == 'refresh-1'
    stored = load_codex_credentials(account='google-workspace')
    assert stored is not None
    assert not any(secret in stored for secret in ('client-secret', 'refresh-1', 'access-1'))
    assert plugin.settings(google_workspace.GoogleWorkspaceSettings).client_id == CLIENT_ID
    source = google_workspace.SettingsSource(plugin)
    rows = {row.key: row for row in source.rows()}
    assert source.current(rows['sign_in']) == 'signed in'
    assert source.current(rows['token']) == '(signed in with Google)'
    for name in ('GOOGLE_CLIENT_SECRET', 'GOOGLE_REFRESH_TOKEN'):
        with pytest.raises(ValueError, match='used by google-workspace'):
            api_keys.rename_key(name=name, new_name='OTHER')

    assert await token_for(plugin) == 'access-1'
    assert len(google.forms) == 1
    clock.now += 3541
    assert await token_for(plugin) == 'access-2'
    assert google.forms[1] == {
        'grant_type': 'refresh_token',
        'refresh_token': 'refresh-1',
        'client_id': CLIENT_ID,
        'client_secret': 'client-secret',
    }
    assert await token_for(plugin) == 'access-2'
    clock.now += 3600
    assert await token_for(plugin) == 'access-3'
    assert api_keys.load_keys()['GOOGLE_REFRESH_TOKEN'].get_secret_value() == 'refresh-2'


async def test_changed_settings_and_missing_keys_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    google = Google([tokens('access-1')])
    plugin, oauth, _ = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID})
    await menu(plugin, oauth, 'sign_in')
    source = google_workspace.SettingsSource(plugin)
    rows = {row.key: row for row in source.rows()}

    source.apply(rows['client_id'], 'other.apps.googleusercontent.com')
    with pytest.raises(UserError, match='client ID changed'):
        await token_for(plugin)
    source.reset(rows['client_id'])
    source.apply(rows['client_id'], CLIENT_ID)
    plugin.save_settings(google_workspace.GoogleWorkspaceSettings(client_id=CLIENT_ID, services=['chat']))
    with pytest.raises(UserError, match='need Google permissions'):
        await token_for(plugin)
    output = io.StringIO()
    settings: dict[str, JsonValue] = {'client_id': CLIENT_ID, 'services': ['chat']}
    google_workspace.activate(
        PluginHost[None](name='google_workspace', console=Console(file=output), settings=settings)
    )
    assert 'sign in with Google again' in output.getvalue()

    plugin.save_settings(google_workspace.GoogleWorkspaceSettings(client_id=CLIENT_ID))
    api_keys.delete_key(name='GOOGLE_REFRESH_TOKEN')
    with pytest.raises(UserError, match='GOOGLE_REFRESH_TOKEN is missing'):
        await token_for(plugin)


@pytest.mark.parametrize(
    'response,message',
    [
        (httpx.Response(400, json={'error': 'invalid_grant'}), 'expired or been revoked'),
        (httpx.Response(500), 'Could not refresh'),
        (httpx.Response(200, json={'access_token': ''}), 'Could not refresh'),
        (httpx.Response(200, json={'access_token': 'a', 'expires_in': 60, 'refresh_token': ''}), 'Could not refresh'),
    ],
)
async def test_refresh_failures_name_the_fix_without_secrets(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response, message: str
) -> None:
    google = Google([tokens('access-1'), response])
    plugin, oauth, clock = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID})
    await menu(plugin, oauth, 'sign_in')
    clock.now += 7200
    with pytest.raises(UserError, match=message) as error:
        await token_for(plugin)
    assert 'refresh-1' not in str(error.value)


async def test_refresh_network_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    google = Google([tokens('access-1')])
    plugin, oauth, clock = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID})
    await menu(plugin, oauth, 'sign_in')

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('refresh-1', request=request)

    oauth.transport = httpx.MockTransport(unreachable)
    clock.now += 7200
    with pytest.raises(UserError, match='Could not refresh') as error:
        await token_for(plugin)
    assert 'refresh-1' not in str(error.value)


@pytest.mark.parametrize(
    'google,message',
    [
        (Google([tokens('a', scopes=DEFAULT_SCOPES[1:])]), 'did not grant 1 of the requested permissions'),
        (Google([tokens('a', refresh=None)]), 'did not return a refresh token'),
        (Google([httpx.Response(401)]), 'Google sign-in failed'),
        (Google([], callback='error=access_denied'), 'Google authorization was denied'),
    ],
)
async def test_a_failed_sign_in_saves_nothing_and_reopens_the_menu(
    monkeypatch: pytest.MonkeyPatch, google: Google, message: str
) -> None:
    plugin, oauth, _ = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID})
    result = await menu(plugin, oauth, 'sign_in')
    assert message in result
    assert 'GOOGLE_REFRESH_TOKEN' not in api_keys.load_keys()
    assert load_codex_credentials(account='google-workspace') is None


async def test_sign_in_needs_a_client_id_and_can_be_cancelled_or_signed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    google = Google([tokens('access-1')])
    plugin, oauth, _ = plugin_with(monkeypatch, google)
    assert await menu(plugin, oauth, 'sign_in') == 'Set the OAuth client ID first, then sign in with Google.'
    source = google_workspace.SettingsSource(plugin)
    rows = {row.key: row for row in source.rows()}
    assert source.current(rows['client_id']) == '(not set)'
    assert (
        source.problem(rows['client_id'], 'not-a-client') == 'Google client IDs end with .apps.googleusercontent.com.'
    )
    assert source.problem(rows['client_id'], CLIENT_ID) is None

    plugin.save_settings(google_workspace.GoogleWorkspaceSettings(client_id=CLIENT_ID))

    async def cancelled(*, prompt: object, label: str) -> None:
        return None

    monkeypatch.setattr(google_workspace, 'prompt_api_key', cancelled)
    assert await menu(plugin, oauth, 'sign_in') == 'Google sign-in cancelled.'
    assert google.urls == []


async def test_sign_out_returns_to_the_access_token_key(monkeypatch: pytest.MonkeyPatch) -> None:
    entries: dict[tuple[str, str], str] = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        del entries[service, account]

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)
    google = Google([tokens('access-1')])
    plugin, oauth, _ = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID})
    await menu(plugin, oauth, 'sign_in')
    source = google_workspace.SettingsSource(plugin)
    rows = {row.key: row for row in source.rows()}
    assert source.reset(rows['sign_in']) == 'Google Workspace uses the saved key GOOGLE_ACCESS_TOKEN again.'
    assert source.current(rows['sign_in']) == 'not signed in'
    with pytest.raises(UserError, match='GOOGLE_ACCESS_TOKEN'):
        await token_for(plugin)


async def test_a_failed_sign_in_again_keeps_the_earlier_client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    google = Google([tokens('access-1'), tokens('access-2', refresh=None)])
    plugin, oauth, clock = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID}, secrets=('client-secret',))
    await menu(plugin, oauth, 'sign_in')
    google.callback = 'error=access_denied'
    pressed = iter(['down', 'down', 'enter'])
    monkeypatch.setattr(api_keys, 'menu_key', lambda: next(pressed))
    replacement = Prompt('replacement-secret')
    monkeypatch.setattr(google_workspace, 'PromptSession', lambda: replacement)
    assert 'denied' in await menu(plugin, oauth, 'sign_in')
    assert next(replacement.values, None) is None
    assert len(google.urls) == 2
    assert api_keys.load_keys()['GOOGLE_CLIENT_SECRET'].get_secret_value() == 'client-secret'
    clock.now += 7200
    assert await token_for(plugin) == 'access-2'
    assert google.forms[-1]['client_secret'] == 'client-secret'


async def test_the_refresh_token_key_cannot_hold_the_client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    google = Google([tokens('access-1')])
    plugin, oauth, _ = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID})

    async def refresh_key(*, prompt: object, label: str) -> KeyReference:
        return KeyReference(name='GOOGLE_REFRESH_TOKEN')

    monkeypatch.setattr(google_workspace, 'prompt_api_key', refresh_key)
    assert 'Choose another key for the client secret' in await menu(plugin, oauth, 'sign_in')
    assert google.urls == []
    same = KeyReference(name='SHARED')
    with pytest.raises(ValidationError, match='different /keys entries'):
        SignedIn(client_id=CLIENT_ID, client_secret=same, refresh_token=same, scopes=[])


async def test_a_blank_client_secret_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    google = Google([])
    plugin, oauth, _ = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID}, secrets=(' ',))
    assert await menu(plugin, oauth, 'sign_in') == 'A client secret is required.'
    assert google.urls == []


async def test_a_saved_client_secret_is_referenced_not_copied(monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='WORK_SECRET', value='work-secret')
    google = Google([tokens('access-1')])
    plugin, oauth, _ = plugin_with(monkeypatch, google, {'client_id': CLIENT_ID})

    async def saved(*, prompt: object, label: str) -> KeyReference:
        return KeyReference(name='WORK_SECRET')

    monkeypatch.setattr(google_workspace, 'prompt_api_key', saved)
    await menu(plugin, oauth, 'sign_in')
    assert google.forms[0]['client_secret'] == 'work-secret'
    assert 'GOOGLE_CLIENT_SECRET' not in api_keys.load_keys()
    with pytest.raises(ValueError, match='used by google-workspace'):
        api_keys.rename_key(name='WORK_SECRET', new_name='OTHER')
