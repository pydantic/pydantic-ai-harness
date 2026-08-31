"""PostgreSQL fixtures shared by the Absurd integration tests."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from uuid import uuid4

import pytest

pytest.importorskip('absurd_sdk')

import psycopg
from absurd_sdk import AsyncAbsurd
from docker.errors import DockerException
from psycopg import AsyncConnection
from psycopg.rows import TupleRow
from testcontainers.community.postgres import PostgresContainer

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
ABSURD_SQL = (FIXTURES / 'absurd.sql').read_text()
AsyncConn = AsyncConnection[TupleRow]


def _docker_host_env() -> None:  # pragma: no cover - environment-dependent
    """Point Testcontainers at Docker Desktop's user socket on macOS when needed."""
    if 'DOCKER_HOST' in os.environ:
        return
    docker_socket = Path.home() / '.docker' / 'run' / 'docker.sock'
    if docker_socket.exists():
        os.environ['DOCKER_HOST'] = f'unix://{docker_socket}'


def _normalize_dsn(url: str) -> str:
    if url.startswith('postgresql+psycopg2://'):
        return 'postgresql://' + url.split('://', 1)[1]
    return url  # pragma: no cover - Testcontainers currently returns the psycopg2 form


@pytest.fixture(scope='session')
def postgres_container() -> Iterator[PostgresContainer]:
    """Start the disposable PostgreSQL instance used by the integration tests."""
    _docker_host_env()
    if (
        'DOCKER_HOST' not in os.environ and not Path('/var/run/docker.sock').exists()
    ):  # pragma: no cover - local-only skip
        pytest.skip('Docker is unavailable: no Docker socket or `DOCKER_HOST` was found')
    try:
        container = PostgresContainer('postgres:16-alpine')
        container.start()
    except DockerException as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f'Docker is unavailable: {exc}')
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture(scope='session')
def db_dsn(postgres_container: PostgresContainer) -> str:
    """Install Absurd's checked-in schema in the disposable database."""
    dsn = _normalize_dsn(postgres_container.get_connection_url())
    with psycopg.connect(dsn, autocommit=True) as conn:
        # This is a trusted, checked-in schema fixture, not user input.
        conn.execute(ABSURD_SQL)  # pyright: ignore[reportCallIssue, reportArgumentType]
    return dsn


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
async def async_conn(db_dsn: str) -> AsyncIterator[AsyncConn]:
    async with await AsyncConnection.connect(db_dsn, autocommit=True) as conn:
        yield conn


@pytest.fixture
async def absurd(async_conn: AsyncConn) -> AsyncIterator[AsyncAbsurd]:
    """Yield an Absurd client with an isolated queue backed by PostgreSQL."""
    queue = f'test_{uuid4().hex[:8]}'
    client = AsyncAbsurd(async_conn, queue_name=queue)
    await client.create_queue()
    try:
        yield client
    finally:
        await client.drop_queue()
