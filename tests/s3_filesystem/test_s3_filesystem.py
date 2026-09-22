from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from pydantic_ai.workspaces import (
    CommandResult,
    MountableFilesystem,
    SupportsCommands,
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceFileEntry,
    WorkspaceRef,
)

from pydantic_ai_harness.s3_filesystem import S3Filesystem

pytestmark = pytest.mark.anyio

MakeFilesystem = Callable[..., S3Filesystem]


@pytest.fixture
def make_filesystem(monkeypatch: pytest.MonkeyPatch) -> MakeFilesystem:
    """Build an `S3Filesystem` whose lazily created boto3 client is the given fake."""

    def make(client: FakeS3Client, **kwargs: Any) -> S3Filesystem:
        def create_client(*args: Any, **create_kwargs: Any) -> FakeS3Client:
            return client

        monkeypatch.setattr('boto3.client', create_client)
        return S3Filesystem('bucket', **kwargs)

    return make


class FakeBody:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self) -> bytes:
        return self.data


class FakeS3Client:
    def __init__(self, *, page_size: int | None = None) -> None:
        self.objects: dict[str, bytes] = {}
        self.page_size = page_size

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs['Key']
        if key not in self.objects:
            raise _missing()
        return {'Body': FakeBody(self.objects[key])}

    def put_object(self, **kwargs: Any) -> None:
        self.objects[kwargs['Key']] = kwargs['Body']

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs['Key']
        if key not in self.objects:
            raise _missing()
        return {'ContentLength': len(self.objects[key])}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        prefix = kwargs.get('Prefix', '')
        delimiter = kwargs.get('Delimiter')
        keys = sorted(key for key in self.objects if key.startswith(prefix))
        start = int(kwargs.get('ContinuationToken', '0'))
        end = len(keys) if self.page_size is None else start + self.page_size
        page = keys[start:end]
        truncated = end < len(keys)
        next_token = str(end) if truncated else None
        if delimiter is None:
            return {
                'Contents': [{'Key': key, 'Size': len(self.objects[key])} for key in page],
                'IsTruncated': truncated,
                'NextContinuationToken': next_token,
            }

        contents: list[dict[str, Any]] = []
        common: set[str] = set()
        for key in page:
            relative = key[len(prefix) :]
            if not relative:
                contents.append({'Key': key, 'Size': len(self.objects[key])})
            elif delimiter in relative:
                common.add(prefix + relative.split(delimiter, 1)[0] + delimiter)
            else:
                contents.append({'Key': key, 'Size': len(self.objects[key])})
        return {
            'Contents': contents,
            'CommonPrefixes': [{'Prefix': value} for value in sorted(common)],
            'IsTruncated': truncated,
            'NextContinuationToken': next_token,
        }

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        for item in kwargs['Delete']['Objects']:
            self.objects.pop(item['Key'], None)
        return {}


class MissingObject(Exception):
    def __init__(self) -> None:
        self.response: dict[str, object] = {'Error': {'Code': 'NoSuchKey'}}


def _missing() -> MissingObject:
    return MissingObject()


class FakeMountBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem):
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commands: list[WorkspaceCommand] = []
        self.environments: list[Mapping[str, str] | None] = []
        self.mounted = False
        self.foreign_mount = False
        self.healthy = True
        self.tools_available = True
        self.destination_empty = True
        self.mount_exit_code = 0
        self.mount_attempts = 0

    @property
    def ref(self) -> WorkspaceRef | None:
        return None

    async def working_dir(self) -> str:
        return '/workspace'

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del shell, cwd, timeout
        self.commands.append(command)
        self.environments.append(env)
        if isinstance(command, str):
            if '/proc/mounts' in command:
                if self.foreign_mount:
                    return CommandResult(exit_code=3, stdout='', stderr='another filesystem is already mounted')
                if self.mounted and self.healthy:
                    return CommandResult(exit_code=0, stdout='', stderr='')
                if self.mounted:
                    self.mounted = False
                    self.healthy = True
                if not self.tools_available:
                    return CommandResult(exit_code=4, stdout='', stderr='s3fs is not installed')
                if not self.destination_empty:
                    return CommandResult(exit_code=5, stdout='', stderr='mount destination is not empty')
                return CommandResult(exit_code=1, stdout='', stderr='')
            if 's3fs ' in command:
                self.mount_attempts += 1
                mounted = self.mount_exit_code == 0 and self.healthy
                self.mounted = mounted
                return CommandResult(exit_code=0 if mounted else 1, stdout='', stderr='' if mounted else 'mount failed')
            if command.startswith('fusermount -uz '):
                self.mounted = False
            return CommandResult(exit_code=0, stdout='', stderr='')
        return CommandResult(exit_code=0, stdout='', stderr='')

    async def read_bytes(self, path: str) -> bytes:
        try:
            return self.files[path]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def stat(self, path: str) -> WorkspaceFileEntry:
        raise NotImplementedError

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        raise NotImplementedError

    async def make_dir(self, path: str) -> None:
        pass

    async def remove(self, path: str) -> None:
        self.files.pop(path, None)

    async def exists(self, path: str) -> bool:
        return path in self.files


def test_s3_filesystem_validates_credential_configuration() -> None:
    with pytest.raises(ValueError, match='bucket must not be empty'):
        S3Filesystem('')
    with pytest.raises(ValueError, match='must be provided together'):
        S3Filesystem('bucket', access_key_id='access')
    with pytest.raises(ValueError, match='session_token requires'):
        S3Filesystem('bucket', session_token='token')


async def test_s3_filesystem_serves_file_operations(make_filesystem: MakeFilesystem) -> None:
    filesystem = make_filesystem(FakeS3Client(), prefix='project')

    assert isinstance(filesystem, SupportsFilesystem)
    assert isinstance(filesystem, MountableFilesystem)

    await filesystem.make_dir('/reports')
    await filesystem.write_bytes('/reports/result.txt', b'result')

    assert await filesystem.read_bytes('/reports/result.txt') == b'result'
    assert (await filesystem.stat('/reports/result.txt')).size == 6
    assert [(entry.name, entry.is_dir) for entry in await filesystem.list_dir('/')] == [('reports', True)]
    assert [(entry.name, entry.is_dir) for entry in await filesystem.list_dir('/reports')] == [('result.txt', False)]
    assert await filesystem.exists('/reports/result.txt')

    await filesystem.remove('/reports')
    assert not await filesystem.exists('/reports/result.txt')


async def test_s3_filesystem_handles_root_directories_files_and_paginated_lists(
    make_filesystem: MakeFilesystem,
) -> None:
    filesystem = make_filesystem(FakeS3Client(page_size=1))

    assert (await filesystem.stat('/')).is_dir
    await filesystem.make_dir('/')
    await filesystem.write_bytes('/a.txt', b'a')
    await filesystem.write_bytes('/b.txt', b'b')
    await filesystem.write_bytes('/dir/a.txt', b'a')
    await filesystem.write_bytes('/dir/b.txt', b'b')
    assert [entry.name for entry in await filesystem.list_dir('/')] == ['a.txt', 'b.txt', 'dir']

    await filesystem.remove('/dir')
    assert not await filesystem.exists('/dir/a.txt')
    await filesystem.remove('/a.txt')
    assert not await filesystem.exists('/a.txt')
    with pytest.raises(FileNotFoundError):
        await filesystem.remove('/missing')


async def test_s3_filesystem_reports_partial_delete_failures(make_filesystem: MakeFilesystem) -> None:
    class PartialDeleteClient(FakeS3Client):
        def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
            return {'Errors': [{'Key': kwargs['Delete']['Objects'][0]['Key'], 'Code': 'AccessDenied'}]}

    client = PartialDeleteClient()
    client.objects['file.txt'] = b'content'
    with pytest.raises(WorkspaceError, match='failed to remove'):
        await make_filesystem(client).remove('/file.txt')


async def test_s3_filesystem_prefers_directories_when_an_object_has_children(
    make_filesystem: MakeFilesystem,
) -> None:
    client = FakeS3Client()
    client.objects = {'dir': b'object', 'dir/file.txt': b'child'}
    filesystem = make_filesystem(client)

    assert (await filesystem.stat('/dir')).is_dir
    assert [(entry.name, entry.is_dir) for entry in await filesystem.list_dir('/')] == [('dir', True)]
    with pytest.raises(PermissionError, match='Cannot remove the S3 filesystem root'):
        await filesystem.remove('/')


async def test_s3_filesystem_translates_missing_objects_and_enforces_read_only(
    make_filesystem: MakeFilesystem,
) -> None:
    client = FakeS3Client()
    filesystem = make_filesystem(client)

    with pytest.raises(FileNotFoundError):
        await filesystem.read_bytes('/missing.txt')
    with pytest.raises(FileNotFoundError):
        await filesystem.stat('/missing.txt')

    await filesystem.write_bytes('/file.txt', b'content')
    with pytest.raises(NotADirectoryError):
        await filesystem.list_dir('/file.txt')

    read_only = make_filesystem(client, read_only=True)
    with pytest.raises(PermissionError):
        await read_only.write_bytes('/file.txt', b'changed')
    with pytest.raises(PermissionError):
        await read_only.make_dir('/directory')
    with pytest.raises(PermissionError):
        await read_only.remove('/file.txt')


async def test_s3_filesystem_rejects_invalid_paths_and_malformed_responses(
    make_filesystem: MakeFilesystem,
) -> None:
    with pytest.raises(ValueError, match='path must be absolute'):
        await S3Filesystem('bucket').exists('relative')

    class NoBodyClient(FakeS3Client):
        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            return {}

    with pytest.raises(WorkspaceError, match='returned no body'):
        await make_filesystem(NoBodyClient()).read_bytes('/file')

    class FailedClient(FakeS3Client):
        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError('get failed')

        def head_object(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError('head failed')

    with pytest.raises(RuntimeError, match='get failed'):
        await make_filesystem(FailedClient()).read_bytes('/file')
    with pytest.raises(RuntimeError, match='head failed'):
        await make_filesystem(FailedClient()).stat('/file')
    with pytest.raises(RuntimeError, match='head failed'):
        await make_filesystem(FailedClient()).remove('/file')

    class MalformedClient(FakeS3Client):
        def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
            prefix = kwargs.get('Prefix', '')
            return {
                'CommonPrefixes': [{'Prefix': 1}, {'Prefix': prefix}, 'invalid'],
                'Contents': [
                    {'Key': 1},
                    {'Key': prefix},
                    {'Key': f'{prefix}nested/file.txt'},
                    'invalid',
                ],
                'IsTruncated': False,
            }

    malformed_filesystem = make_filesystem(MalformedClient())
    assert await malformed_filesystem.list_dir('/') == ()
    await malformed_filesystem.remove('/missing')

    malformed_error = MissingObject()
    malformed_error.response = {'Error': 'invalid'}

    class MalformedErrorClient(FakeS3Client):
        def get_object(self, **kwargs: Any) -> dict[str, Any]:
            raise malformed_error

    with pytest.raises(type(malformed_error)):
        await make_filesystem(MalformedErrorClient()).read_bytes('/file')


async def test_s3_filesystem_rejects_truncated_responses_without_a_token(
    make_filesystem: MakeFilesystem,
) -> None:
    class TruncatedClient(FakeS3Client):
        def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
            return {'Contents': [], 'IsTruncated': True}

    filesystem = make_filesystem(TruncatedClient())
    with pytest.raises(WorkspaceError, match='without a continuation token'):
        await filesystem.list_dir('/')
    with pytest.raises(WorkspaceError, match='without a continuation token'):
        await filesystem.remove('/directory')


async def test_s3_filesystem_creates_its_boto_client_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeS3Client()
    calls: list[tuple[str, dict[str, Any]]] = []

    def create_client(service: str, **kwargs: Any) -> FakeS3Client:
        calls.append((service, kwargs))
        return client

    monkeypatch.setattr('boto3.client', create_client)
    filesystem = S3Filesystem('bucket', region='region', endpoint_url='https://s3.example.test')

    assert not await filesystem.exists('/missing')
    assert not await filesystem.exists('/still-missing')
    assert calls == [
        (
            's3',
            {
                'region_name': 'region',
                'endpoint_url': 'https://s3.example.test',
                'aws_access_key_id': None,
                'aws_secret_access_key': None,
                'aws_session_token': None,
            },
        )
    ]


async def test_s3_filesystem_mount_is_idempotent_and_uses_the_same_source() -> None:
    backend = FakeMountBackend()
    filesystem = S3Filesystem(
        'bucket',
        prefix='project',
        region='eu-west-1',
        endpoint_url='https://s3.example.test',
        access_key_id='access',
        secret_access_key='secret',
        session_token='token',
        read_only=True,
    )
    target = Workspace(backend)

    await filesystem.ensure_mounted(target, '/data')
    await filesystem.ensure_mounted(target, '/data')

    assert backend.mount_attempts == 1
    mount_command = next(
        command for command in backend.commands if isinstance(command, str) and 's3fs bucket:' in command
    )
    assert 's3fs bucket:/project /data' in mount_command
    assert '-o fsname=pydantic-ai-s3:bucket:project:https://s3.example.test' in mount_command
    assert '-o use_session_token' in mount_command
    assert '-o ro' in mount_command
    assert 'secret' not in mount_command
    mount_index = backend.commands.index(mount_command)
    assert backend.environments[mount_index] == {
        'AWS_ACCESS_KEY_ID': 'access',
        'AWSACCESSKEYID': 'access',
        'AWS_SECRET_ACCESS_KEY': 'secret',
        'AWSSECRETACCESSKEY': 'secret',
        'AWS_SESSION_TOKEN': 'token',
        'AWSSESSIONTOKEN': 'token',
    }


async def test_s3_filesystem_replaces_its_unhealthy_mount() -> None:
    backend = FakeMountBackend()
    filesystem = S3Filesystem('bucket')
    target = Workspace(backend)

    await filesystem.ensure_mounted(target, '/data')
    backend.healthy = False
    await filesystem.ensure_mounted(target, '/data')

    assert backend.mount_attempts == 2


async def test_s3_filesystem_refuses_foreign_mounts() -> None:
    backend = FakeMountBackend()
    backend.foreign_mount = True

    with pytest.raises(WorkspaceError, match='another filesystem is already mounted'):
        await S3Filesystem('bucket').ensure_mounted(Workspace(backend), '/data')


@pytest.mark.parametrize(
    ('failure', 'message'),
    [
        ('tools', 's3fs is not installed'),
        ('destination', 'mount destination is not empty'),
        ('mount', 'mount failed'),
        ('readiness', 'mount failed'),
    ],
)
async def test_s3_filesystem_mount_failures_are_strict_and_clean_up(failure: str, message: str) -> None:
    backend = FakeMountBackend()
    if failure == 'tools':
        backend.tools_available = False
    elif failure == 'destination':
        backend.destination_empty = False
    elif failure == 'mount':
        backend.mount_exit_code = 1
    else:
        backend.healthy = False

    filesystem = S3Filesystem(
        'bucket',
        access_key_id='access',
        secret_access_key='secret',
    )
    with pytest.raises(WorkspaceError, match=message):
        await filesystem.ensure_mounted(Workspace(backend), '/data')

    assert not backend.mounted
