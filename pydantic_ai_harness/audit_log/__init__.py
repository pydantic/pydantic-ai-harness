"""Structured, redacted audit records of tool calls and run outcomes to a pluggable sink."""

from pydantic_ai_harness.audit_log._capability import AuditLog
from pydantic_ai_harness.audit_log._redact import Redactor, default_secret_redactor
from pydantic_ai_harness.audit_log._sink import (
    AuditSink,
    InMemoryAuditSink,
    JsonlAuditSink,
    SqliteAuditSink,
)
from pydantic_ai_harness.audit_log._types import RunAuditRecord, RunOutcome, ToolCallRecord

__all__ = [
    'AuditLog',
    'AuditSink',
    'InMemoryAuditSink',
    'JsonlAuditSink',
    'Redactor',
    'RunAuditRecord',
    'RunOutcome',
    'SqliteAuditSink',
    'ToolCallRecord',
    'default_secret_redactor',
]
