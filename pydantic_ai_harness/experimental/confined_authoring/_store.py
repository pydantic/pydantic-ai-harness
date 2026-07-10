"""Disk-backed store for authored tool slots.

All slots for one capability live in a single `slots.json` manifest under
`directory`: name, kind, description, parameters, `uses` allowlist, return type,
the Monty source, status, and last validation error. The manifest is the surface
a UI reads and the source the serving toolset reloads each run. Writes go through
a temp file and an atomic replace, so a crash mid-write never leaves a manifest
that reads as "no slots".

The store keeps the live injected-function pool only to validate against it; the
callables are never serialized. A slot re-validates against the current pool
every time it is loaded to serve, so removing a function the slot needs sends
the slot back to `draft` with a truthful error rather than serving broken code.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic

from pydantic import BaseModel, ValidationError
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.experimental.confined_authoring._slots import (
    AuthoredSlot,
    InjectedFunction,
    SlotParameter,
    SlotValueType,
    index_functions,
)
from pydantic_ai_harness.experimental.confined_authoring._validate import (
    SlotValidationError,
    validate_tool_slot,
)

_NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')
_RESERVED_NAMES = frozenset({'author_tool_slot', 'list_tool_slots', 'disable_tool_slot'})
_SERVABLE_STATUSES = frozenset({'validated', 'active'})


class _Manifest(BaseModel):
    slots: list[AuthoredSlot] = []


@dataclass
class SlotStore(Generic[AgentDepsT]):
    """Read/write index of authored slots under `directory`.

    Construct one over the same `directory` and injected-function pool from any
    process to pick up previously authored slots.
    """

    directory: Path
    """Directory holding the `slots.json` manifest."""

    functions: Sequence[InjectedFunction[AgentDepsT]] = ()
    """The capability-scoped injected-function pool, used to validate slots. Callables are never persisted."""

    _by_name: dict[str, InjectedFunction[AgentDepsT]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_name = index_functions(self.functions)

    @property
    def pool(self) -> Mapping[str, InjectedFunction[AgentDepsT]]:
        """The injected-function pool indexed by name."""
        return self._by_name

    @property
    def _manifest_path(self) -> Path:
        return self.directory / 'slots.json'

    def _load(self) -> _Manifest:
        path = self._manifest_path
        if not path.exists():
            return _Manifest()
        try:
            return _Manifest.model_validate_json(path.read_text(encoding='utf-8'))
        except (OSError, ValidationError, ValueError):
            return _Manifest()

    def _save(self, manifest: _Manifest) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix='slots.', suffix='.json.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
                tmp.write(manifest.model_dump_json(indent=2))
            os.replace(tmp_name, self._manifest_path)
        except BaseException:
            os.unlink(tmp_name)
            raise

    def _upsert(self, record: AuthoredSlot) -> None:
        manifest = self._load()
        for index, existing in enumerate(manifest.slots):
            if existing.name == record.name:
                manifest.slots[index] = record
                break
        else:
            manifest.slots.append(record)
        self._save(manifest)

    def author_tool(
        self,
        *,
        name: str,
        description: str,
        code: str,
        parameters: Sequence[SlotParameter] = (),
        uses: Sequence[str] = (),
        returns: SlotValueType | None = None,
    ) -> AuthoredSlot:
        """Validate and persist a tool slot, upserting by name.

        Raises `ValueError` for an unusable name before persisting anything. A
        slot that fails static validation is still written (so the model can
        inspect and re-author it) with `status='draft'` and `last_error` set;
        `load_servable` skips it.
        """
        if not _NAME_RE.match(name):
            raise ValueError(
                f'invalid slot name {name!r}: use lowercase letters, digits, and underscores, starting with a letter'
            )
        if name in _RESERVED_NAMES:
            raise ValueError(f'slot name {name!r} is reserved by the authoring tools; choose another')

        parameters = list(parameters)
        uses = list(uses)
        try:
            validate_tool_slot(parameters=parameters, uses=uses, code=code, returns=returns, functions=self._by_name)
        except SlotValidationError as exc:
            record = AuthoredSlot(
                name=name,
                description=description,
                parameters=parameters,
                uses=uses,
                returns=returns,
                code=code,
                status='draft',
                last_error=str(exc),
            )
        else:
            record = AuthoredSlot(
                name=name,
                description=description,
                parameters=parameters,
                uses=uses,
                returns=returns,
                code=code,
                status='validated',
                last_error=None,
            )
        self._upsert(record)
        return record

    def disable(self, name: str) -> bool:
        """Mark the named slot disabled so it is no longer served. Returns whether it existed."""
        manifest = self._load()
        found = False
        for record in manifest.slots:
            if record.name == name:
                record.status = 'disabled'
                found = True
        if found:
            self._save(manifest)
        return found

    def list_all(self) -> list[AuthoredSlot]:
        """Return every slot in the manifest, in insertion order."""
        return self._load().slots

    def load_servable(self) -> list[AuthoredSlot]:
        """Re-validate the enabled slots against the current pool and return the servable ones.

        A slot is enabled while its status is `validated` or `active` (anything
        but `draft`/`disabled`). Each enabled slot is re-checked against the live
        function pool: one that validates is served and recorded `active` with
        `last_error` cleared; one that no longer validates (e.g. a function it
        uses was removed) keeps its status but records the error and is not
        served. `last_error` stays truthful about a slot's current health, and an
        unchanged outcome does not rewrite the manifest.
        """
        manifest = self._load()
        servable: list[AuthoredSlot] = []
        changed = False
        for record in manifest.slots:
            if record.status not in _SERVABLE_STATUSES:
                continue
            try:
                validate_tool_slot(
                    parameters=record.parameters,
                    uses=record.uses,
                    code=record.code,
                    returns=record.returns,
                    functions=self._by_name,
                )
            except SlotValidationError as exc:
                if record.last_error != str(exc):
                    record.last_error = str(exc)
                    changed = True
                continue
            if record.status != 'active' or record.last_error is not None:
                record.status = 'active'
                record.last_error = None
                changed = True
            servable.append(record)
        if changed:
            self._save(manifest)
        return servable
