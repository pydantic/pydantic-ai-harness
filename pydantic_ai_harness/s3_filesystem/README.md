# S3 Filesystem

`S3Filesystem` exposes an S3 bucket through workspace file operations and can mount the same bucket into a command workspace with [s3fs-fuse](https://github.com/s3fs-fuse/s3fs-fuse).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/s3_filesystem/)

```bash
uv add "pydantic-ai-harness[s3,e2b]"
```

The sandbox image must contain `s3fs`, `fusermount`, FUSE support, and permission to mount filesystems.

```python
from e2b import AsyncSandbox
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.e2b_workspace import E2BWorkspaceBackend
from pydantic_ai_harness.s3_filesystem import S3Filesystem


def workspace_with_reports(sandbox: AsyncSandbox) -> Workspace:
    filesystem = S3Filesystem(
        'reports-bucket',
        prefix='agent-data',
        region='us-east-1',
        read_only=True,
    )
    return Workspace(
        E2BWorkspaceBackend(sandbox),
        mounts={'/data': filesystem},
    )
```

The caller creates `sandbox` and is responsible for using an image with `s3fs` and FUSE support.

File tools access the bucket through the S3 API. Before the first command, the workspace asks the filesystem to mount the same bucket and prefix at `/data`. If mounting fails, the command does not run.

Without explicit credentials, file operations use boto3's normal credential chain, while the mount relies on credentials already present in the sandbox, such as an instance role. Explicit `access_key_id`, `secret_access_key`, and optional `session_token` apply to both: boto3 receives them directly and the `s3fs` process receives them through its environment. `read_only=True` applies to both as well: file operations reject writes and the bucket is mounted with the `ro` option.

S3 does not provide full POSIX semantics. In particular, renames are non-atomic copies, and random writes or appends may rewrite an entire object. Use a POSIX network filesystem when applications depend on stronger semantics.

Command access through `s3fs` can be slower than direct file operations, especially for metadata-heavy workloads and small or random reads and writes. Each operation crosses the workspace kernel's FUSE interface into `s3fs`, which may need one or more S3 requests to emulate POSIX behavior. Large sequential transfers may still be dominated by network throughput. Prefer the workspace file methods for whole-file operations; use the mount when a command specifically needs a filesystem path.
