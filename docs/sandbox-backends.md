---
title: Writing Sandbox Backends
description: Share lazy native SDK acquisition across sandbox backends.
---

# Writing Sandbox Backends

`LazySandbox` is optional authoring support for backends that acquire a native SDK
sandbox. It supplies an awaitable `sandbox` property, caches the acquired handle,
and coordinates concurrent first use. Provider authors implement `create_or_attach()`
instead of repeating a lock and cache in every backend.

[Source code](https://github.com/pydantic/pydantic-ai-harness/blob/main/pydantic_ai_harness/sandbox.py)

## Author and caller responsibilities

This outline shows the acquisition part of a provider backend. `NativeSandbox`
stands for its SDK's type; the provider also implements core's `ref`, `run`, and
`working_dir` contract and any supported filesystem methods.

```python
from pydantic_ai_harness.sandbox import LazySandbox


class MyBackend(LazySandbox[NativeSandbox]):
    async def create_or_attach(self) -> NativeSandbox:
        # Use the SDK to acquire a handle, record its identity, or raise.
        ...
```

`create_or_attach()` is the implementation hook. Backend operations and callers
use the inherited property:

```python
native = await self.sandbox
```

Authors do not implement the property or initialize its cache and lock. A custom
constructor calls `super().__init__()`, optionally passing an already acquired
native handle. The provider owns configuration, identity, SDK error translation,
and cleanup if acquisition fails or is cancelled.

The helper does not implement core's `SandboxBackend` protocol by itself. A local
backend that has no native SDK handle does not need to inherit it.

## Acquisition behavior

Reading `backend.sandbox` does no I/O. Awaiting it acquires the handle on first
use and returns the same cached native object on subsequent awaits. The property
remains `Awaitable[NativeSandbox]` even after caching, so using SDK attributes
without `await` is a type error and raises an attribute error at runtime.

Concurrent acquisition is serialized on one helper instance. After success,
waiting callers get the cached handle. After failure or cancellation, the lock
is released and a waiting or later caller may attempt acquisition again. The
helper does not automatically retry within the failed call. Cancelling a waiting
caller does not cancel the caller performing acquisition. The lock is released
before callers execute operations on the native object.

This coordination is per instance, not across processes or separate backend
objects. Providers still own name conflicts and remote identity rules. Acquisition
should return a usable native handle; a cached handle is not a continuous health
check, and later SDK operations may fail if the environment disappears.

The helper does not close clients or stop, pause, or destroy remote sandboxes.
Their lifetimes remain provider and application responsibilities.

::: pydantic_ai_harness.sandbox.LazySandbox
