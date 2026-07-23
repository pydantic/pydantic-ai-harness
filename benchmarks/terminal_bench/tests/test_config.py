"""Tests for run configuration: model mapping, usage limits, cost table."""

from __future__ import annotations

import pytest

from pydantic_ai_harness_terminal_bench.config import (
    COST_ESTIMATES,
    DEFAULT_MODEL,
    DEFAULT_REQUEST_LIMIT,
    DEFAULT_TOOL_CALLS_LIMIT,
    build_usage_limits,
    convert_model_name,
    parse_trial_ids,
    resolve_model_name,
)


@pytest.mark.parametrize(
    ('harbor_name', 'expected'),
    [
        ('anthropic/claude-opus-4-6', 'anthropic:claude-opus-4-6'),
        ('openai/gpt-5.2', 'openai:gpt-5.2'),
        ('vertex/gemini-3.1-pro', 'google-cloud:gemini-3.1-pro'),
        ('google-cloud/gemini-3.1-pro', 'google-cloud:gemini-3.1-pro'),
        ('gemini/gemini-3.1-pro', 'google:gemini-3.1-pro'),
        ('google/gemini-3.1-pro', 'google:gemini-3.1-pro'),
        # Already a Pydantic AI id: unchanged.
        ('anthropic:claude-opus-4-6', 'anthropic:claude-opus-4-6'),
        # Bare model name, no provider: unchanged.
        ('gpt-5.2', 'gpt-5.2'),
    ],
)
def test_convert_model_name(harbor_name: str, expected: str) -> None:
    assert convert_model_name(harbor_name) == expected


def test_usage_limits_defaults() -> None:
    limits = build_usage_limits()
    assert limits.request_limit == DEFAULT_REQUEST_LIMIT
    assert limits.tool_calls_limit == DEFAULT_TOOL_CALLS_LIMIT


def test_usage_limits_override() -> None:
    limits = build_usage_limits(request_limit=5, tool_calls_limit=7, total_tokens_limit=9)
    assert limits.request_limit == 5
    assert limits.tool_calls_limit == 7
    assert limits.total_tokens_limit == 9


@pytest.mark.parametrize(
    ('name', 'use_gateway', 'expected'),
    [
        ('anthropic/claude-sonnet-4-6', False, 'anthropic:claude-sonnet-4-6'),
        ('anthropic/claude-sonnet-4-6', True, 'gateway/anthropic:claude-sonnet-4-6'),
        # Already a Pydantic AI id, gated through the gateway prefix.
        ('anthropic:claude-sonnet-4-6', True, 'gateway/anthropic:claude-sonnet-4-6'),
        # None falls back to the default model, still gateway-prefixable.
        (None, False, DEFAULT_MODEL),
        (None, True, f'gateway/{DEFAULT_MODEL}'),
    ],
)
def test_resolve_model_name(name: str | None, use_gateway: bool, expected: str) -> None:
    assert resolve_model_name(name, use_gateway=use_gateway) == expected


@pytest.mark.parametrize(
    ('session_id', 'expected'),
    [
        ('fix-git__bZZeEkw__env', ('fix-git', 'fix-git__bZZeEkw')),
        ('openssl-selfsigned-cert__aQ1__agent', ('openssl-selfsigned-cert', 'openssl-selfsigned-cert__aQ1')),
        # Role-less / malformed ids degrade to the raw id for both.
        ('bare', ('bare', 'bare')),
        ('', ('', '')),
    ],
)
def test_parse_trial_ids(session_id: str, expected: tuple[str, str]) -> None:
    assert parse_trial_ids(session_id) == expected


def test_cost_estimates_shape() -> None:
    assert COST_ESTIMATES
    for row in COST_ESTIMATES:
        label, shape, usd = row
        assert label and shape
        assert usd.startswith('$')
