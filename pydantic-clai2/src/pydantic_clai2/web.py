"""Local browser chat using Pydantic AI's UI and run loop."""

from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from inspect import isawaitable

from anyio import CancelScope
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability
from rich.console import Console

from ._app import DEFAULT_PLUGINS
from .auth import CodexAuth
from .commands import Commands
from .config import PluginSettings, Settings
from .model_resolution import resolve_model
from .model_settings import model_settings_from_json
from .plugin_loader import PluginLoader
from .plugins import SessionEndReason, SessionStart
from .project_settings import ProjectSettings
from .settings_store import SettingsStore

try:
    import uvicorn
    from starlette.applications import Starlette
except ImportError as exc:
    raise ImportError('Install pydantic-clai2[web] to use --web.') from exc


@asynccontextmanager
async def web_app(
    *,
    settings: Settings,
    store: SettingsStore,
    project: ProjectSettings,
    builtin_plugins: Sequence[PluginSettings] = DEFAULT_PLUGINS,
    console: Console | None = None,
) -> AsyncGenerator[Starlette]:
    """Own the agent and plugins until the server has drained its requests.

    Shell turn hooks fail registration rather than losing guards. Core hooks run normally.
    The caller owns serving the returned app, on the same event loop as this context.
    """
    if settings.model is None:
        raise ValueError('Choose a model with --model or clai2 config set model before using --web.')
    console = console or Console()

    def no_terminal() -> AbstractAsyncContextManager[None]:
        raise ValueError('Terminal widgets are unavailable with --web.')

    async def capabilities(ctx: RunContext[None]) -> CombinedCapability[None]:
        bound: list[AbstractCapability[None]] = []
        for cap in loader.capabilities():
            if isinstance(cap, AbstractCapability):
                bound.append(cap)  # pyright: ignore[reportUnknownArgumentType]
            else:
                resolved = cap(ctx)
                if isawaitable(resolved):
                    resolved = await resolved
                if resolved is not None:
                    bound.append(resolved)
        return CombinedCapability(bound)

    agent = Agent(
        await resolve_model(settings.model, auth=CodexAuth(console)),
        name='clai2',
        deps_type=type(None),
        model_settings=model_settings_from_json(store.model_settings(settings.model)).to_model_settings(),
        retries={'tools': settings.tool_retries},
        max_concurrency=1,
        capabilities=[capabilities],
    )
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=console,
        commands=Commands(),
        session_start=lambda: SessionStart(agent=agent, settings=settings),
        builtin=builtin_plugins,
        project=project.plugins,
        full_screen=no_terminal,
        host_hooks=frozenset({'session_start', 'session_end'}),
    )
    stack = AsyncExitStack()
    reason: SessionEndReason = 'error'
    try:
        await stack.enter_async_context(agent)
        for entry in loader.entries():
            if not entry.declaration.enabled:
                continue
            if entry.path is None and entry.declaration.factory.partition(':')[0] in {
                'pydantic_clai2.ask_user_menu',
                'pydantic_clai2.sessions',
                'pydantic_clai2.notifications',
                'pydantic_clai2.updates',
            }:
                console.print(f'Omitting terminal plugin {entry.name!r} for --web.', markup=False)
                continue
            await loader.load(entry.name)
        yield agent.to_web()
        reason = 'exit'
    finally:
        with CancelScope(shield=True):
            try:
                await loader.close(reason)
            finally:
                await stack.aclose()


async def serve_web(*, settings: Settings, store: SettingsStore, project: ProjectSettings, port: int) -> None:
    """Bind only to IPv4 loopback; Uvicorn and plugin cleanup share the caller's loop."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None]:
        async with web_app(settings=settings, store=store, project=project) as chat:
            app.mount('/', chat)
            async with chat.router.lifespan_context(chat):
                yield
                # Uvicorn requests normal lifespan shutdown after a socket bind failure.
                if not server.started:
                    raise RuntimeError('Web server stopped before binding its socket.')

    app = Starlette(lifespan=lifespan)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port))
    await server.serve()
