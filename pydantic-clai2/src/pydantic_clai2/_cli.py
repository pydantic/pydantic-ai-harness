"""CLI settings resolution and interactive application startup."""

import argparse
import asyncio
import os
from pathlib import Path

from pydantic_ai.usage import UsageLimits

from ._app import DEFAULT_PLUGINS, chat, create_agent
from .commands import config_command, plugins_command
from .config import resolve_settings
from .plugin_loader import PluginError
from .project_settings import load_project_settings
from .settings_store import SettingsStore


def run() -> None:
    """Parse explicit overrides without replacing persisted preferences."""
    parser = argparse.ArgumentParser(description='CLAI 2.0: streaming Pydantic AI terminal')
    parser.add_argument(
        '--resume', nargs='?', const='', metavar='SESSION-ID', help='Restore a saved session; no ID opens the browser'
    )
    parser.add_argument('--web', action='store_true', help='Serve the browser chat UI on 127.0.0.1')
    parser.add_argument('--port', type=int, help='Web port (default: 7932); requires --web')
    parser.add_argument('--model', help='Provider-qualified model name')
    parser.add_argument('--request-limit', type=int)
    parser.add_argument('--database', type=Path, help='Settings database location')
    parser.add_argument('command', nargs='?', choices=('config', 'plugins'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.port is not None and (not args.web or not 1 <= args.port <= 65535):
        parser.error('--port requires --web and a value between 1 and 65535')
    if args.web and (args.command or args.resume is not None):
        parser.error('--web cannot be combined with --resume, config, or plugins')
    try:
        store = SettingsStore(args.database)
        if args.command and args.resume is not None:
            parser.error('--resume cannot be combined with config or plugins')
        if args.command:
            handler = config_command if args.command == 'config' else plugins_command
            print(handler(store, args.arguments))
            return
        project = load_project_settings(Path.cwd())
        overrides = store.overrides() | project.overrides
        if model := args.model or os.getenv('CLAI_MODEL'):
            overrides['model'] = model
        if args.request_limit is not None:
            overrides['run.request_limit'] = args.request_limit
        settings = resolve_settings(overrides)
        if args.web:
            from .web import serve_web  # noqa: PLC0415 -- web dependencies are optional.

            if 'run.request_limit' in overrides:
                parser.error(
                    '--web does not support explicit request limits (--request-limit, saved, or project settings). '
                    'Core uses 50 requests per run. Reset run.request_limit or use the terminal.'
                )
            asyncio.run(serve_web(settings=settings, store=store, project=project, port=args.port or 7932))
            return
        asyncio.run(
            chat(
                create_agent(),
                deps=None,
                usage_limits=UsageLimits(request_limit=settings.request_limit),
                settings=settings,
                store=store,
                builtin_plugins=DEFAULT_PLUGINS,
                project=project,
                resume=args.resume,
            )
        )
    except (ValueError, TypeError, ImportError, AttributeError, LookupError, PluginError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        pass
