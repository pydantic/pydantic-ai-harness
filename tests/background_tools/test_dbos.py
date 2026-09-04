"""DBOS integration tests for `BackgroundTools`."""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip('dbos')


def test_durable_background_step_survives_dbos_process_recovery(tmp_path: Path) -> None:
    markers = tmp_path / 'markers'
    markers.mkdir()
    database = tmp_path / 'recovery.sqlite'
    workflow_id = str(uuid.uuid4())
    runner = Path(__file__).parents[2] / 'integration_tests' / 'dbos' / 'background_tools_recovery_runner.py'
    command = [sys.executable, str(runner), str(database), str(markers), workflow_id]

    first = subprocess.Popen([*command, 'first'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 15
        while not (markers / 'step-finished').exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (markers / 'step-finished').exists()
        first.kill()
        first.wait(timeout=5)
    finally:
        if first.poll() is None:  # pragma: no cover -- defensive cleanup after an earlier assertion
            first.kill()
            first.wait(timeout=5)

    (markers / 'release-delivery').touch()
    recovered = subprocess.run(
        [*command, 'second'],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert (markers / 'result').read_text().splitlines() == ['done', '1']
    assert len((markers / 'step-calls').read_text().splitlines()) == 1
