from __future__ import annotations

import posixpath
import shlex
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Protocol, TypeVar, cast

import anyio
from pydantic_ai.workspaces import (
    FileEntry,
    MountableFilesystem,
    Workspace,
    WorkspaceError,
    WorkspaceFileEntry,
)

__all__ = ('S3Filesystem',)

_T = TypeVar('_T')
_MISSING_CODES = {'404', 'NoSuchKey', 'NotFound'}


class _S3Client(Protocol):  # pragma: no cover
    def get_object(self, **kwargs: Any) -> Mapping[str, object]: ...

    def put_object(self, **kwargs: Any) -> object: ...

    def head_object(self, **kwargs: Any) -> Mapping[str, object]: ...

    def list_objects_v2(self, **kwargs: Any) -> Mapping[str, object]: ...

    def delete_objects(self, **kwargs: Any) -> Mapping[str, object]: ...


class S3Filesystem(MountableFilesystem):
    """An S3 bucket that file tools access through the API and commands access through s3fs.

    Construction performs no network or workspace I/O. The boto3 client is created on the first
    filesystem operation, while the command mount is established by `Workspace` before its first
    command. The workspace image must contain `s3fs`, `fusermount`, and FUSE support.

    S3 is not a POSIX filesystem: renames are non-atomic copies, and random writes or appends may
    rewrite an entire object. The direct filesystem API deliberately exposes only whole-file
    operations.
    """

    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = '',
        region: str | None = None,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        read_only: bool = False,
    ) -> None:
        if not bucket:
            raise ValueError('bucket must not be empty')
        if (access_key_id is None) != (secret_access_key is None):
            raise ValueError('access_key_id and secret_access_key must be provided together')
        if session_token is not None and access_key_id is None:
            raise ValueError('session_token requires access_key_id and secret_access_key')

        self._bucket = bucket
        self._prefix = prefix.strip('/')
        self._region = region
        self._endpoint_url = endpoint_url
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._read_only = read_only
        self._client: _S3Client | None = None

    async def read_bytes(self, path: str) -> bytes:
        key = self._key(path)
        try:
            response = await self._call(self._get_client().get_object, Bucket=self._bucket, Key=key)
        except Exception as error:
            self._raise_missing(error, path)
            raise
        body = response.get('Body')
        if body is None or not hasattr(body, 'read'):
            raise WorkspaceError(f'S3 returned no body for {path!r}')
        read = cast(Callable[[], bytes], getattr(body, 'read'))
        data = await anyio.to_thread.run_sync(read)
        return bytes(data)

    async def write_bytes(self, path: str, data: bytes) -> None:
        self._require_writable(path)
        await self._call(self._get_client().put_object, Bucket=self._bucket, Key=self._key(path), Body=data)

    async def stat(self, path: str) -> WorkspaceFileEntry:
        path = _normalize(path)
        if path == '/':
            return FileEntry(name='/', path='/', is_dir=True, size=None)
        key = self._key(path)
        children = await self._call(
            self._get_client().list_objects_v2,
            Bucket=self._bucket,
            Prefix=key.rstrip('/') + '/',
            MaxKeys=1,
        )
        if children.get('Contents'):
            return FileEntry(name=posixpath.basename(path), path=path, is_dir=True, size=None)
        try:
            response = await self._call(self._get_client().head_object, Bucket=self._bucket, Key=key)
        except Exception as error:
            if _is_missing(error):
                raise FileNotFoundError(path) from error
            raise
        size = response.get('ContentLength')
        return FileEntry(
            name=posixpath.basename(path),
            path=path,
            is_dir=False,
            size=size if isinstance(size, int) else None,
        )

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        path = _normalize(path)
        if path != '/':
            entry = await self.stat(path)
            if not entry.is_dir:
                raise NotADirectoryError(path)
        prefix = self._directory_key(path)
        entries: dict[str, FileEntry] = {}
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, object] = {
                'Bucket': self._bucket,
                'Prefix': prefix,
                'Delimiter': '/',
            }
            if continuation_token is not None:
                kwargs['ContinuationToken'] = continuation_token
            response = await self._call(self._get_client().list_objects_v2, **kwargs)
            self._add_list_entries(entries, path, prefix, response)
            if not response.get('IsTruncated'):
                break
            token = response.get('NextContinuationToken')
            if not isinstance(token, str):
                raise WorkspaceError('S3 returned a truncated listing without a continuation token')
            continuation_token = token
        return tuple(entries[name] for name in sorted(entries))

    async def make_dir(self, path: str) -> None:
        self._require_writable(path)
        key = self._directory_key(path)
        if key:
            await self._call(self._get_client().put_object, Bucket=self._bucket, Key=key, Body=b'')

    async def remove(self, path: str) -> None:
        self._require_writable(path)
        path = _normalize(path)
        if path == '/':
            raise PermissionError('Cannot remove the S3 filesystem root.')
        key = self._key(path)
        keys: set[str] = set()
        try:
            await self._call(self._get_client().head_object, Bucket=self._bucket, Key=key)
        except Exception as error:
            if not _is_missing(error):
                raise
        else:
            keys.add(key)
        prefix = self._directory_key(path)
        continuation_token: str | None = None
        while True:
            kwargs: dict[str, object] = {'Bucket': self._bucket, 'Prefix': prefix}
            if continuation_token is not None:
                kwargs['ContinuationToken'] = continuation_token
            response = await self._call(self._get_client().list_objects_v2, **kwargs)
            for item in _mapping_items(response.get('Contents')):
                object_key = item.get('Key')
                if isinstance(object_key, str):
                    keys.add(object_key)
            if not response.get('IsTruncated'):
                break
            token = response.get('NextContinuationToken')
            if not isinstance(token, str):
                raise WorkspaceError('S3 returned a truncated listing without a continuation token')
            continuation_token = token
        if not keys:
            raise FileNotFoundError(path)
        sorted_keys = sorted(keys)
        for start in range(0, len(sorted_keys), 1000):
            response = await self._call(
                self._get_client().delete_objects,
                Bucket=self._bucket,
                Delete={'Objects': [{'Key': item} for item in sorted_keys[start : start + 1000]]},
            )
            if response.get('Errors'):
                raise WorkspaceError(f'S3 failed to remove one or more objects under {path!r}')

    async def exists(self, path: str) -> bool:
        try:
            await self.stat(path)
        except FileNotFoundError:
            return False
        return True

    async def ensure_mounted(self, target: Workspace, path: str) -> None:
        """Ensure s3fs serves this bucket at `path` in the existing command environment."""
        quoted_path = shlex.quote(path)
        mount_path = _proc_mount_path(path)
        mount_name = f'pydantic-ai-s3:{self._bucket}:{self._prefix}:{self._endpoint_url or ""}'
        expected_mount = shlex.quote(f'{_proc_mount_path(mount_name)} {mount_path} fuse.s3fs ')
        any_mount = shlex.quote(f' {mount_path} ')
        preparation = await target.run(
            f'if grep -Fqs -- {expected_mount} /proc/mounts; then '
            f'test -d {quoted_path} 2>/dev/null && exit 0; '
            f'fusermount -uz {quoted_path} || exit 6; '
            f'elif grep -Fqs -- {any_mount} /proc/mounts; then '
            f'echo "another filesystem is already mounted at {quoted_path}" >&2; exit 3; fi; '
            f'command -v s3fs >/dev/null || {{ echo "s3fs is not installed" >&2; exit 4; }}; '
            f'mkdir -p {quoted_path} || {{ echo "cannot create mount destination" >&2; exit 7; }}; '
            f'test -z "$(ls -A {quoted_path})" '
            f'|| {{ echo "mount destination is not empty" >&2; exit 5; }}; exit 1',
            shell=True,
            timeout=30,
        )
        if preparation.exit_code == 0:
            return
        if preparation.exit_code != 1:
            detail = preparation.stderr.strip() or f'preparation exited {preparation.exit_code}'
            raise WorkspaceError(f'Cannot mount s3://{self._bucket} at {path!r}: {detail}')

        mount_env: dict[str, str] | None = None
        if self._access_key_id is not None:
            assert self._secret_access_key is not None
            # s3fs 1.90+ reads the AWS_* names first; 1.89 and earlier read only the legacy
            # names. Setting both families works everywhere and beats ambient sandbox
            # credentials in the 1.90+ precedence order.
            mount_env = {
                'AWS_ACCESS_KEY_ID': self._access_key_id,
                'AWSACCESSKEYID': self._access_key_id,
                'AWS_SECRET_ACCESS_KEY': self._secret_access_key,
                'AWSSECRETACCESSKEY': self._secret_access_key,
            }
            if self._session_token is not None:
                mount_env['AWS_SESSION_TOKEN'] = self._session_token
                mount_env['AWSSESSIONTOKEN'] = self._session_token
        try:
            command = shlex.join(self._mount_command(path, mount_name))
            result = await target.run(
                f'{command} && ls -A {quoted_path} >/dev/null',
                shell=True,
                env=mount_env,
                timeout=90,
            )
            if result.exit_code == 0:
                return
            detail = result.stderr.strip() or f's3fs exited {result.exit_code}'
            raise WorkspaceError(f'Could not mount s3://{self._bucket} at {path!r}: {detail}')
        except BaseException:
            with anyio.move_on_after(10, shield=True):
                await target.run(
                    f'fusermount -uz {quoted_path} 2>/dev/null; rmdir {quoted_path} 2>/dev/null',
                    shell=True,
                )
            raise

    def _get_client(self) -> _S3Client:
        if self._client is None:
            try:
                import boto3  # pyright: ignore[reportMissingTypeStubs]
            except ImportError as error:  # pragma: no cover
                raise ImportError('Install `pydantic-ai-harness[s3]` to use S3Filesystem.') from error
            self._client = cast(
                _S3Client,
                boto3.client(  # pyright: ignore[reportUnknownMemberType]
                    's3',
                    region_name=self._region,
                    endpoint_url=self._endpoint_url,
                    aws_access_key_id=self._access_key_id,
                    aws_secret_access_key=self._secret_access_key,
                    aws_session_token=self._session_token,
                ),
            )
        return self._client

    async def _call(self, function: Callable[..., _T], **kwargs: object) -> _T:
        return await anyio.to_thread.run_sync(partial(function, **kwargs))

    def _key(self, path: str) -> str:
        relative = _normalize(path).lstrip('/')
        return '/'.join(part for part in (self._prefix, relative) if part)

    def _directory_key(self, path: str) -> str:
        key = self._key(path).rstrip('/')
        return f'{key}/' if key else ''

    def _require_writable(self, path: str) -> None:
        if self._read_only:
            raise PermissionError(f'S3 filesystem is read-only: {path!r}')

    def _raise_missing(self, error: Exception, path: str) -> None:
        if _is_missing(error):
            raise FileNotFoundError(path) from error

    def _add_list_entries(
        self,
        entries: dict[str, FileEntry],
        path: str,
        prefix: str,
        response: Mapping[str, object],
    ) -> None:
        for item in _mapping_items(response.get('CommonPrefixes')):
            item_prefix = item.get('Prefix')
            if not isinstance(item_prefix, str):
                continue
            name = item_prefix[len(prefix) :].rstrip('/')
            if name:
                entry_path = '/' + name if path == '/' else f'{path}/{name}'
                entries[name] = FileEntry(name=name, path=entry_path, is_dir=True, size=None)
        for item in _mapping_items(response.get('Contents')):
            key = item.get('Key')
            if not isinstance(key, str):
                continue
            name = key[len(prefix) :]
            if not name or '/' in name or (name in entries and entries[name].is_dir):
                continue
            size = item.get('Size')
            entry_path = '/' + name if path == '/' else f'{path}/{name}'
            entries[name] = FileEntry(
                name=name,
                path=entry_path,
                is_dir=False,
                size=size if isinstance(size, int) else None,
            )

    def _mount_command(self, path: str, mount_name: str) -> list[str]:
        source = self._bucket if not self._prefix else f'{self._bucket}:/{self._prefix}'
        command = ['s3fs', source, path]
        options = [f'fsname={mount_name}']
        if self._session_token is not None:
            options.append('use_session_token')
        if self._region is not None:
            options.append(f'endpoint={self._region}')
        if self._endpoint_url is not None:
            options.extend((f'url={self._endpoint_url}', 'use_path_request_style'))
        if self._read_only:
            options.append('ro')
        for option in options:
            command.extend(('-o', option))
        return command


def _proc_mount_path(path: str) -> str:
    return path.replace('\\', r'\134').replace(' ', r'\040').replace('\t', r'\011').replace('\n', r'\012')


def _normalize(path: str) -> str:
    if not posixpath.isabs(path):
        raise ValueError(f'path must be absolute, got {path!r}')
    return '/' + posixpath.normpath(path).lstrip('/')


def _is_missing(error: Exception) -> bool:
    response_value = getattr(error, 'response', None)
    if not isinstance(response_value, Mapping):
        return False
    response = cast(Mapping[object, object], response_value)
    details_value = response.get('Error')
    if not isinstance(details_value, Mapping):
        return False
    details = cast(Mapping[object, object], details_value)
    code = details.get('Code')
    return isinstance(code, str) and code in _MISSING_CODES


def _mapping_items(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    sequence = cast(Sequence[object], value)
    return tuple(cast(Mapping[str, object], item) for item in sequence if isinstance(item, Mapping))
