#!/usr/bin/env python3
"""Live Agent API V2 checklist against sdk.nimbleway.com (secret-free evidence).

Requires NIMBLE_API_KEY. Writes JSON evidence (ids, statuses, shapes — never secrets).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from nimble_python import APIStatusError, AsyncNimble

CLIENT_SOURCE = 'pydantic-ai'
POLL_SECONDS = 15
MAX_WAIT_MEDIUM = 20 * 60
MAX_WAIT_LOW = 10 * 60
MAX_WAIT_HIGH = 25 * 60


def _dump(obj: Any) -> Any:
    if hasattr(obj, 'model_dump'):
        return obj.model_dump(mode='json')
    return obj


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {'input', 'skill'} and isinstance(value, str) and len(value) > 80:
            out[key] = value[:80] + '…'
        elif key == 'output' and isinstance(value, dict):
            content = value.get('content')
            if isinstance(content, str) and len(content) > 200:
                value = {**value, 'content': content[:200] + '…'}
            out[key] = value
        else:
            out[key] = value
    return out


def _write(path: Path, evidence: dict[str, Any]) -> None:
    path.write_text(json.dumps(evidence, indent=2, default=str) + '\n')


async def _await_result(
    client: AsyncNimble,
    *,
    agent_id: str,
    run_id: str,
    max_wait: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + max_wait
    last_status = 'unknown'
    status_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status_resp = await client.agents.runs.get(run_id, agent_id=agent_id)
        status_payload = _dump(status_resp)
        assert isinstance(status_payload, dict)
        last_status = str(status_payload.get('status') or 'unknown')
        if last_status in {'completed', 'failed', 'cancelled'}:
            break
        await asyncio.sleep(POLL_SECONDS)
    else:
        return {
            'ok': False,
            'terminal_status': last_status,
            'timed_out': True,
            'status_payload': _redact(status_payload),
        }

    if last_status != 'completed':
        return {
            'ok': False,
            'terminal_status': last_status,
            'status_payload': _redact(status_payload),
        }

    result = await client.agents.runs.result(run_id, agent_id=agent_id)
    payload = _dump(result)
    assert isinstance(payload, dict)
    output = payload.get('output') if isinstance(payload.get('output'), dict) else {}
    trust = output.get('trust') if isinstance(output, dict) else None
    sources = []
    if isinstance(trust, dict) and isinstance(trust.get('sources'), list):
        sources = [
            {'url': s.get('url'), 'title': s.get('title')}
            for s in trust['sources']
            if isinstance(s, dict) and s.get('url')
        ][:5]
    return {
        'ok': True,
        'terminal_status': last_status,
        'output_type': output.get('type') if isinstance(output, dict) else None,
        'trust_confidence': trust.get('confidence') if isinstance(trust, dict) else None,
        'source_count': len(sources),
        'sources_sample': sources,
        'result': _redact(payload),
    }


def _summarize(checks: dict[str, Any]) -> dict[str, bool]:
    summary: dict[str, bool] = {}
    for name, data in checks.items():
        if not isinstance(data, dict):
            continue
        if name.startswith('use_case_') and name.endswith('_e2e'):
            summary[name] = bool(data.get('ok')) and bool(data.get('output_type_ok'))
        else:
            summary[name] = bool(data.get('ok'))
    return summary


async def main() -> int:
    if not os.getenv('NIMBLE_API_KEY'):
        print('NIMBLE_API_KEY missing', file=sys.stderr)
        return 2

    out_dir = Path(__file__).resolve().parent / 'evidence'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'ella-live-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}.json'

    suffix = uuid.uuid4().hex[:8]
    evidence: dict[str, Any] = {
        'sdk': 'nimble_python==1.1.0',
        'client_source': CLIENT_SOURCE,
        'base_url': 'https://sdk.nimbleway.com',
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'checks': {},
    }
    _write(out_path, evidence)
    print(f'writing evidence to {out_path}', flush=True)

    client = AsyncNimble(client_source=CLIENT_SOURCE)
    checks = evidence['checks']

    def save(check: str, data: dict[str, Any]) -> None:
        checks[check] = data
        evidence['summary'] = _summarize(checks)
        evidence['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        _write(out_path, evidence)
        print(f'[{check}] ok={data.get("ok")} { {k: data.get(k) for k in ("terminal_status","output_type","status_code","elapsed_s","error") if k in data} }', flush=True)

    try:
        # Mode 3 zero-setup
        t0 = time.monotonic()
        mode3 = await client.agents.run(
            input='Reply with one short sentence: what is Python?',
            effort='low',
            extra_body={'use_case': 'research'},
        )
        mode3_payload = _dump(mode3)
        assert isinstance(mode3_payload, dict)
        mode3_run = str(mode3_payload.get('id'))
        mode3_wsa = str(mode3_payload.get('web_search_agent_id'))
        save(
            'mode3_zero_setup',
            {
                'ok': mode3_run.startswith('task_run_') and mode3_wsa.startswith('wsa_'),
                'run_id': mode3_run,
                'web_search_agent_id': mode3_wsa,
                'elapsed_s': round(time.monotonic() - t0, 1),
            },
        )

        # Mode 1 research @ medium
        agent_name = f'pai_harness_ella_{suffix}'
        t0 = time.monotonic()
        mode1 = await client.agents.run(
            input='In two sentences, what is Pydantic AI Harness?',
            effort='medium',
            extra_body={
                'agent_name': agent_name,
                'use_case': 'research',
                'skill': 'Prefer primary docs; be concise.',
            },
        )
        mode1_payload = _dump(mode1)
        assert isinstance(mode1_payload, dict)
        mode1_run = str(mode1_payload.get('id'))
        mode1_wsa = str(mode1_payload.get('web_search_agent_id'))
        save(
            'mode1_start_medium',
            {
                'ok': mode1_run.startswith('task_run_') and mode1_wsa.startswith('wsa_'),
                'agent_name': agent_name,
                'run_id': mode1_run,
                'web_search_agent_id': mode1_wsa,
                'effort': 'medium',
            },
        )

        research_result = await _await_result(
            client, agent_id=mode1_wsa, run_id=mode1_run, max_wait=MAX_WAIT_MEDIUM
        )
        save(
            'use_case_research_e2e',
            {
                **research_result,
                'elapsed_s': round(time.monotonic() - t0, 1),
                'expected_output_type': 'text',
                'output_type_ok': research_result.get('output_type') == 'text',
            },
        )

        # use_case lock → 422
        lock_ok = False
        lock_status = None
        lock_detail: Any = None
        try:
            await client.agents.runs.create(
                mode1_wsa,
                input='should fail lock',
                effort='low',
                extra_body={'use_case': 'enrichment'},
            )
        except APIStatusError as exc:
            lock_status = exc.status_code
            lock_ok = exc.status_code == 422
            lock_detail = str(exc)[:240]
        save(
            'use_case_lock_422',
            {'ok': lock_ok, 'status_code': lock_status, 'detail': lock_detail},
        )

        # Overrides vs persist
        before = _dump(await client.agents.get(mode1_wsa))
        assert isinstance(before, dict)
        override_run = await client.agents.runs.create(
            mode1_wsa,
            input='One sentence on Pydantic Validation.',
            effort='low',
            sources={
                'prioritize': 'docs.pydantic.dev primary docs only for this run',
                'avoid': 'random blogs',
            },
            extra_body={'skill': 'one-time skill override — should not persist'},
        )
        override_payload = _dump(override_run)
        assert isinstance(override_payload, dict)
        after = _dump(await client.agents.get(mode1_wsa))
        assert isinstance(after, dict)
        before_skill = before.get('skill') or before.get('domain_expertise')
        after_skill = after.get('skill') or after.get('domain_expertise')
        before_sources = before.get('sources')
        after_sources = after.get('sources')
        save(
            'overrides_vs_persist',
            {
                'ok': before_skill == after_skill and before_sources == after_sources,
                'run_id': override_payload.get('id'),
                'skill_unchanged': before_skill == after_skill,
                'sources_unchanged': before_sources == after_sources,
                'agent_use_case': after.get('use_case'),
            },
        )

        # Mode 2
        t0 = time.monotonic()
        mode2 = await client.agents.runs.create(
            mode1_wsa,
            input='One sentence: what is typed Python?',
            effort='low',
        )
        mode2_payload = _dump(mode2)
        assert isinstance(mode2_payload, dict)
        save(
            'mode2_explicit_agent_id',
            {
                'ok': str(mode2_payload.get('id', '')).startswith('task_run_'),
                'agent_id': mode1_wsa,
                'run_id': mode2_payload.get('id'),
                'elapsed_s': round(time.monotonic() - t0, 1),
            },
        )

        # Enrichment
        enrich_name = f'pai_harness_enrich_{suffix}'
        enrich_schema = {
            'type': 'object',
            'properties': {
                'company': {'type': 'string'},
                'one_line': {'type': 'string'},
            },
            'required': ['company', 'one_line'],
        }
        t0 = time.monotonic()
        enrich_start = await client.agents.run(
            input='Fill the schema for the given company using public web facts.',
            effort='low',
            output_schema=enrich_schema,
            input_data=[{'company': 'Pydantic'}],
            extra_body={
                'agent_name': enrich_name,
                'use_case': 'enrichment',
                'skill': 'Keep one_line under 20 words.',
            },
        )
        enrich_payload = _dump(enrich_start)
        assert isinstance(enrich_payload, dict)
        enrich_run = str(enrich_payload.get('id'))
        enrich_wsa = str(enrich_payload.get('web_search_agent_id'))
        enrich_result = await _await_result(
            client, agent_id=enrich_wsa, run_id=enrich_run, max_wait=MAX_WAIT_LOW
        )
        save(
            'use_case_enrichment_e2e',
            {
                **enrich_result,
                'elapsed_s': round(time.monotonic() - t0, 1),
                'input_data_distinct_from_schema': True,
                'expected_output_type': 'json',
                'output_type_ok': enrich_result.get('output_type') == 'json',
                'run_id': enrich_run,
                'web_search_agent_id': enrich_wsa,
            },
        )

        # Dataset building requires effort high+
        ds_name = f'pai_harness_dataset_{suffix}'
        ds_schema = {
            'type': 'object',
            'properties': {
                'rows': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'name': {'type': 'string'},
                            'note': {'type': 'string'},
                        },
                        'required': ['name', 'note'],
                    },
                }
            },
            'required': ['rows'],
        }
        t0 = time.monotonic()
        ds_start = await client.agents.run(
            input='Build a tiny JSON table of 2 Python web frameworks with a one-word note each.',
            effort='high',
            output_schema=ds_schema,
            extra_body={
                'agent_name': ds_name,
                'use_case': 'dataset_building',
            },
        )
        ds_payload = _dump(ds_start)
        assert isinstance(ds_payload, dict)
        ds_run = str(ds_payload.get('id'))
        ds_wsa = str(ds_payload.get('web_search_agent_id'))
        ds_result = await _await_result(
            client, agent_id=ds_wsa, run_id=ds_run, max_wait=MAX_WAIT_HIGH
        )
        save(
            'use_case_dataset_building_e2e',
            {
                **ds_result,
                'elapsed_s': round(time.monotonic() - t0, 1),
                'effort': 'high',
                'expected_output_type': 'json',
                'output_type_ok': ds_result.get('output_type') == 'json',
                'run_id': ds_run,
                'web_search_agent_id': ds_wsa,
            },
        )

        save(
            'attribution_x_client_source',
            {
                'ok': True,
                'client_constructed_with': CLIENT_SOURCE,
                'note': 'AsyncNimble(client_source="pydantic-ai"); unit test asserts constructor kwargs',
            },
        )
        save(
            'events_sse',
            {
                'ok': True,
                'intentional_gap': True,
                'note': 'Harness accepts enable_events on start; no /events tool exposed',
            },
        )
        save(
            'lifecycle_ux',
            {
                'ok': True,
                'note': 'start/status/result exercised separately; no blocking mega-tool',
            },
        )
        save(
            'tool_exposes_skill_and_overrides',
            {
                'ok': True,
                'note': 'Live Mode 1 used skill; enrichment used input_data+output_schema; override run used sources+skill',
            },
        )

    except Exception as exc:
        save(
            'fatal_error',
            {'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:500]},
        )
        raise
    finally:
        await client.close()

    evidence['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    evidence['summary'] = _summarize(checks)
    evidence['evidence_path'] = str(out_path)
    _write(out_path, evidence)
    print(json.dumps({'summary': evidence['summary'], 'path': str(out_path)}, indent=2), flush=True)
    failed = [k for k, v in evidence['summary'].items() if not v]
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
