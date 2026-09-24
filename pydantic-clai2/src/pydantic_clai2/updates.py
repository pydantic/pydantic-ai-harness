"""One bounded PyPI check per plugin activation, owned by the plugin's session."""

import asyncio
import platform
from collections.abc import Awaitable, Callable
from contextlib import suppress
from importlib import metadata

import anyio
import httpx2
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field, ValidationError

from .plugins import PluginHost, SessionEnd, SessionStart
from .update_installation import update_guidance


class ReleaseInfo(BaseModel):
    """Fields used from PyPI's project metadata."""

    version: str = Field(max_length=80)


class ReleaseFile(BaseModel):
    """A yanked artifact does not establish an available update."""

    yanked: bool
    requires_python: str | None = None


class Release(BaseModel):
    """The current release and its published artifacts."""

    info: ReleaseInfo
    urls: list[ReleaseFile]


async def latest_version(*, transport: httpx2.AsyncBaseTransport | None = None) -> Version | None:
    """Read PyPI once, with a five-second deadline and a 1 MiB response limit."""
    try:
        with anyio.fail_after(5):
            async with httpx2.AsyncClient(transport=transport, trust_env=False, timeout=5) as client:
                async with client.stream(
                    'GET', 'https://pypi.org/pypi/pydantic-clai2/json', headers={'Accept-Encoding': 'identity'}
                ) as response:
                    if response.status_code != 200:
                        return None
                    body = bytearray()
                    async for chunk in response.aiter_raw():
                        body.extend(chunk)
                        if len(body) > 1024 * 1024:
                            return None
        release = Release.model_validate_json(body, strict=True)
        version = Version(release.info.version)
        if (
            version.is_prerelease
            or version.is_devrelease
            or version.local
            or not any(
                not file.yanked and SpecifierSet(file.requires_python or '').contains(platform.python_version())
                for file in release.urls
            )
        ):
            return None
        return version
    except (httpx2.HTTPError, TimeoutError, ValidationError, InvalidVersion, InvalidSpecifier):
        return None


def activate(host: PluginHost[None], *, check: Callable[[], Awaitable[Version | None]] | None = None) -> None:
    """Start without waiting for the network; unload cancels and drains the worker."""
    task: asyncio.Task[None] | None = None

    async def run() -> None:
        try:
            installed = metadata.distribution('pydantic-clai2')
            if 'Version' not in installed.metadata:
                return
            current = Version(installed.version)
            latest = await (check or latest_version)()
            if latest is not None and latest > current:
                await host.notify(f'CLAI2 {latest} available (installed {current}). {update_guidance(installed)}')
        except (metadata.PackageNotFoundError, OSError, ValueError, httpx2.HTTPError, TimeoutError):
            return

    @host.on('session_start')
    async def start(event: SessionStart) -> None:
        nonlocal task
        task = asyncio.create_task(run(), name='clai2-updates')

    @host.on('session_end')
    async def stop(event: SessionEnd) -> None:
        if task is not None:
            task.cancel()
            with anyio.CancelScope(shield=True), suppress(asyncio.CancelledError):
                await task
