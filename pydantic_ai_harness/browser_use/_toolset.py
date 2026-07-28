"""Browser Use toolset: runs model-written Python through the `browser-use` CLI"""
# ruff: noqa: D415

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import anyio
import anyio.abc
import anyio.to_thread
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from pydantic_ai_harness.browser_use._progress import narrate_call, narrate_error, narrate_result
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS

_INSTALL_HINT = (
    'The `browser-use` CLI was not found on PATH, and no `uvx` fallback was available. Install it with:\n'
    '    uv tool install browser-use\n'
    'then run `browser-use --doctor` once to verify the browser connection.'
)

_SKILL_FETCH_TIMEOUT = 30.0

_MAX_TIMEOUT = 1800.0  # ceiling on a model-supplied timeout_seconds
_MAX_OUTPUT_CHARS = 50_000
_CLOUD_TIMEOUT_MINUTES = 60  # server-side backstop on a per-run cloud browser

_ERROR_TAIL_CHARS = 2000
"""Tail of CLI output shown when cloud provisioning fails"""

SKILL_HEADER = """

--- Browser Use CLI reference ---

"""

_SKILL_CACHE: dict[str, str | None] = {}
"""Skill text per command, fetched once per process"""

_HEREDOC_RE = re.compile(
    r"(?:BU_NAME=(?P<session>\S+)[ \t]+)?browser-use[ \t]*<<'PY'\n(?P<code>.*?)\nPY\n?",
    re.DOTALL,
)
"""One heredoc invocation, with optional `BU_NAME=` prefix"""

_BASH_BLOCK_RE = re.compile(r'```bash\n(?P<body>.*?)```', re.DOTALL)

_PROSE_SUBS = (
    (
        re.compile(r'- Invoke as `browser-use`\. Use heredocs for multi-line commands\.'),
        "- Pass the Python as this tool's `code` argument.",
    ),
    (re.compile(r'BU_NAME=(?P<name>\S+)'), r'session=\g<name>'),
    (re.compile(r'`BU_NAME`'), '`session`'),
)


def _heredoc_to_code(match: re.Match[str]) -> str:
    """Unwrap one shell invocation into the Python it would have piped"""
    code = match.group('code')
    session = match.group('session')
    if session is None:
        return f'{code}\n'
    return f'# pass session={session!r} to this tool\n{code}\n'


def _rewrite_bash_block(match: re.Match[str]) -> str:
    """Turn a fenced shell block of heredocs into a Python block, leaving other shell blocks alone"""
    body = match.group('body')
    if "<<'PY'" not in body:
        return match.group(0)
    return '```python\n' + _HEREDOC_RE.sub(_heredoc_to_code, body).strip() + '\n```'


def adapt_skill_to_tool(text: str) -> str:
    """Rewrite the CLI's skill documentation to address a tool caller instead of a shell user"""
    text = _BASH_BLOCK_RE.sub(_rewrite_bash_block, text)
    for pattern, replacement in _PROSE_SUBS:
        text = pattern.sub(replacement, text)
    return text


_SESSION_BOOT = """
import sys, io, json, traceback
from contextlib import redirect_stdout, redirect_stderr
_bu_ns = dict(globals())
while True:
    with open({inp!r}) as _bu_f:
        _bu_code = _bu_f.read()
    if not _bu_code.strip():
        break
    _bu_buf = io.StringIO()
    try:
        with redirect_stdout(_bu_buf), redirect_stderr(_bu_buf):
            exec(_bu_code, _bu_ns)
        _bu_err = 0
    except BaseException:
        traceback.print_exc(file=_bu_buf)
        _bu_err = 1
    with open({outp!r}, "w") as _bu_f:
        _bu_f.write(json.dumps({{"exit": _bu_err, "out": _bu_buf.getvalue()}}))
"""


class _Session:
    """One long-lived CLI interpreter"""

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str] | None) -> None:
        self._argv = argv
        self._cwd = cwd
        self._env = env
        self._dir: Path | None = None
        self._proc: anyio.abc.Process | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Spawn the CLI running the session loop"""
        self._dir = Path(tempfile.mkdtemp(prefix='harness_browser_session_'))
        inp, outp = self._dir / 'in', self._dir / 'out'
        await anyio.to_thread.run_sync(os.mkfifo, inp)
        await anyio.to_thread.run_sync(os.mkfifo, outp)
        boot = _SESSION_BOOT.format(inp=str(inp), outp=str(outp))
        self._proc = await anyio.open_process(
            self._argv,
            cwd=self._cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=self._env,
        )
        assert self._proc.stdin is not None
        await self._proc.stdin.send(boot.encode())
        await self._proc.stdin.aclose()

    async def call(self, code: str, timeout: float) -> tuple[int, str]:
        """Run `code` in the session, and return its exit flag and combined output"""
        assert self._dir is not None
        inp, outp = self._dir / 'in', self._dir / 'out'
        deadline = time.monotonic() + timeout
        await self._write(inp, code, deadline)
        raw = await self._read(outp, deadline)
        payload = json.loads(raw)
        return int(payload['exit']), str(payload['out'])

    async def _write(self, fifo: Path, text: str, deadline: float) -> None:
        """Write once a reader attaches, giving up at the deadline"""
        while True:
            try:
                fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            except OSError:  # no reader yet
                if time.monotonic() >= deadline or not self.alive:
                    raise TimeoutError from None
                await anyio.sleep(0.05)
                continue
            with os.fdopen(fd, 'w') as handle:
                handle.write(text)
            return

    async def _read(self, fifo: Path, deadline: float) -> str:
        """Collect the reply, giving up at the deadline"""
        fd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
        chunks: list[bytes] = []
        try:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                except BlockingIOError:  # pragma: no cover - writer attached but idle
                    chunk = b''
                if chunk:
                    chunks.append(chunk)
                elif chunks:
                    return b''.join(chunks).decode('utf-8', errors='replace')
                if time.monotonic() >= deadline:
                    raise TimeoutError
                await anyio.sleep(0.02)
        finally:
            os.close(fd)

    async def close(self) -> None:
        """Kill the interpreter and delete its pipes"""
        proc, self._proc = self._proc, None
        directory, self._dir = self._dir, None
        if proc is not None:  # pragma: no branch
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:  # pragma: no cover - already exited
                pass
            with anyio.CancelScope(shield=True):
                await proc.wait()
            await proc.aclose()
        if directory is not None:  # pragma: no branch
            shutil.rmtree(directory, ignore_errors=True)


_SCREENSHOT_RE = re.compile(r'(?<![\w/])(/[^\s\'"]+\.(?:png|jpe?g|webp))(?![\w])', re.IGNORECASE)
"""Absolute image path in output, as printed by `capture_screenshot()`"""

_FILE_RE = re.compile(r'(?<![\w/])(/[^\s\'"]+\.[A-Za-z0-9]{1,8})(?![\w])')
"""Any absolute file path the code printed"""

_MAX_FILES = 16

_MAX_SCREENSHOTS = 4

_MAX_SCREENSHOT_BYTES = 8 * 1024 * 1024  # bigger than this stays out of context

_MEDIA_TYPES = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp'}


def collect_files(output: str, *, newer_than: float) -> list[dict[str, str | int]]:
    """Files the executed code printed and wrote during this call"""
    import mimetypes

    found: list[dict[str, str | int]] = []
    for match in dict.fromkeys(_FILE_RE.findall(output)):
        path = Path(match)
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_mtime < newer_than - 1 or not path.is_file():
            continue
        media_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        found.append({'path': str(path), 'media_type': media_type, 'bytes': stat.st_size})
        if len(found) == _MAX_FILES:
            break
    return found


def collect_screenshots(output: str, *, newer_than: float) -> list[BinaryContent]:
    """Read image files the executed code printed, so the model can see the page"""
    found: list[BinaryContent] = []
    for match in dict.fromkeys(_SCREENSHOT_RE.findall(output)):
        path = Path(match)
        try:
            stat = path.stat()
            if stat.st_mtime < newer_than - 1 or stat.st_size > _MAX_SCREENSHOT_BYTES:
                continue
            data = path.read_bytes()
        except OSError:  # pragma: no cover - printed a path that is not readable
            continue
        found.append(BinaryContent(data=data, media_type=_MEDIA_TYPES[path.suffix.lower()]))
        if len(found) == _MAX_SCREENSHOTS:
            break
    return found


_CHROME_CANDIDATES = (
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
)

_CHROME_COMMANDS = ('google-chrome-stable', 'google-chrome', 'chromium-browser', 'chromium', 'chrome')

_HEADLESS_STARTUP_TIMEOUT = 30.0  # seconds until DevToolsActivePort must appear


def find_chrome() -> str | None:
    """Locate a Chrome or Chromium binary to launch headlessly"""
    for variable in ('BH_CHROME_PATH', 'CHROME_PATH'):
        if configured := os.environ.get(variable):
            return configured
    for command in _CHROME_COMMANDS:
        if found := shutil.which(command):
            return found
    return next((path for path in _CHROME_CANDIDATES if Path(path).exists()), None)


SESSION_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')
"""The CLI's `BU_NAME` naming rules"""


class BrowserUseToolset(FunctionToolset[AgentDepsT]):
    """A single `browser_exec` tool backed by a persistent CLI interpreter session"""

    def __init__(
        self,
        *,
        default_timeout: float = 300.0,
        browser: Literal['local', 'headless', 'cloud'] = 'local',
        scope: Literal['run', 'agent'] = 'run',
        progress: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self._cwd = Path.cwd()
        self._default_timeout = default_timeout
        self._scope: Literal['run', 'agent'] = scope
        self._progress = progress
        self._enter_count = 0
        self._session: _Session | None = None
        self._browser: Literal['local', 'headless', 'cloud'] = browser
        self._headless_proc: anyio.abc.Process | None = None
        self._headless_profile: Path | None = None
        self._headless_cdp_url: str | None = None
        self._owned_session: str | None = None
        self.add_function(self.browser_exec, name='browser_exec')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Give each run its own interpreter state, unless scoped to the agent"""
        if self._scope == 'agent':
            return self
        return BrowserUseToolset[AgentDepsT](
            default_timeout=self._default_timeout,
            browser=self._browser,
            scope=self._scope,
            progress=self._progress,
        )

    async def __aenter__(self) -> BrowserUseToolset[AgentDepsT]:
        """Track nesting so an agent-scoped toolset survives per-run exits"""
        self._enter_count += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        """End the interpreter and stop any owned browser, at the outermost exit only"""
        self._enter_count = max(0, self._enter_count - 1)
        if self._enter_count > 0:
            return
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._owned_session is not None:
            session, self._owned_session = self._owned_session, None
            # shielded: a browser that bills must stop even on cancel
            with anyio.CancelScope(shield=True):
                try:
                    await self._run_cli(
                        self._resolve_argv(),
                        f'stop_remote_daemon({session!r})\n',
                        self._default_timeout,
                        self._build_env(None),
                        raise_on_timeout=False,
                    )
                except ModelRetry:  # pragma: no cover - the CLI vanished mid-run
                    pass
        await self._stop_headless_browser()

    async def _ensure_headless_browser(self) -> str | None:
        """Launch headless Chrome on first use and return its DevTools URL"""
        if self._browser != 'headless':
            return None
        if self._headless_cdp_url is not None:
            return self._headless_cdp_url

        binary = find_chrome()
        if binary is None:
            raise ModelRetry(
                'No Chrome or Chromium binary was found to launch headless. Install Chrome, or set '
                "BH_CHROME_PATH (or the capability's `chrome_path`) to the executable."
            )
        profile = Path(tempfile.mkdtemp(prefix='harness_browser_profile_'))
        try:
            proc = await anyio.open_process(
                [
                    binary,
                    '--headless=new',
                    '--remote-debugging-port=0',
                    f'--user-data-dir={profile}',
                    '--no-first-run',
                    '--no-default-browser-check',
                    'about:blank',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            shutil.rmtree(profile, ignore_errors=True)
            raise ModelRetry(f'Failed to launch headless Chrome ({binary!r}): {e}') from e

        self._headless_proc = proc
        self._headless_profile = profile
        port_file = profile / 'DevToolsActivePort'
        try:
            with anyio.fail_after(_HEADLESS_STARTUP_TIMEOUT):
                while True:
                    if port_file.exists() and (port := port_file.read_text().splitlines()[:1]):
                        break
                    await anyio.sleep(0.1)
        except TimeoutError:
            await self._stop_headless_browser()
            raise ModelRetry(
                f'Headless Chrome did not report a DevTools port within {_HEADLESS_STARTUP_TIMEOUT}s.'
            ) from None
        self._headless_cdp_url = f'http://127.0.0.1:{port[0]}'
        return self._headless_cdp_url

    async def _stop_headless_browser(self) -> None:
        """Kill the launched headless Chrome and delete its profile"""
        proc, self._headless_proc = self._headless_proc, None
        profile, self._headless_profile = self._headless_profile, None
        self._headless_cdp_url = None
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:  # pragma: no cover - already exited
                pass
            with anyio.CancelScope(shield=True):
                await proc.wait()
            await proc.aclose()
        if profile is not None:
            shutil.rmtree(profile, ignore_errors=True)

    async def _ensure_cloud_session(self) -> str | None:
        """Return the cloud browser name for this run, provisioning one on first use"""
        if self._browser != 'cloud':
            return None
        if self._owned_session is not None:
            return self._owned_session
        name = f'pyai{uuid.uuid4().hex[:12]}'
        exit_code, stdout, stderr = await self._run_cli(
            self._resolve_argv(),
            f'start_remote_daemon({name!r}, timeout={_CLOUD_TIMEOUT_MINUTES})\n',
            self._default_timeout,
            self._build_env(None),
        )
        if exit_code != 0:
            raise ModelRetry(
                'Could not start a Browser Use cloud browser. Sign in with `browser-use auth login` '
                f'or set BROWSER_USE_API_KEY.\n\n{(stdout + stderr)[-_ERROR_TAIL_CHARS:]}'
            )
        self._owned_session = name
        return name

    async def cli_skill_text(self) -> str | None:
        """Return the CLI's skill documentation, fetching it at most once per command"""
        try:
            argv = self._resolve_argv()
        except ModelRetry:
            return None
        if argv[0] in _SKILL_CACHE:
            return _SKILL_CACHE[argv[0]]
        exit_code, stdout, _ = await self._run_cli(
            [*argv, 'skill', 'show'],
            '',
            min(_SKILL_FETCH_TIMEOUT, self._default_timeout),
            self._build_env(None),
            raise_on_timeout=False,
        )
        skill = adapt_skill_to_tool(stdout.strip()) if exit_code == 0 and stdout.strip() else None
        _SKILL_CACHE[argv[0]] = skill
        return skill

    async def browser_exec(
        self, code: str, session: str | None = None, timeout_seconds: float | None = None
    ) -> ToolReturn[str]:
        """Run Python code in a real web browser via the Browser Use CLI.

        Browser helpers pre-imported: `new_tab(url)`, `goto_url(url)`,
        `page_info()`, `js(expression)`, `click_at_xy(x, y)`, `fill_input(...)`,
        `type_text(...)`, `press_key(...)`, `scroll(...)`, `wait_for_load()`,
        `wait_for_element(...)`, `capture_screenshot()`, `list_tabs()`,
        `switch_tab(...)`, and more. Use `print(...)` for any
        data you need back -- the tool returns what the code prints. Start
        `code` with a one-line `#` comment describing the step for the user in
        plain, non-technical language (under 60 characters); it is shown as the
        step's label while the call runs.

        Your calls run in one persistent Python session, so variables you
        assign survive to your next call, as does the browser itself. If a call
        times out the session restarts (the browser survives); re-derive what
        you need from the page. Batching a whole sub-procedure (navigate, wait,
        extract) into one call is still faster than one call per action.
        """
        if self._progress is None:
            return await self._exec(code, session, timeout_seconds)
        narrate_call(self._progress, code, 'steps')
        try:
            result = await self._exec(code, session, timeout_seconds)
        except ModelRetry as error:
            narrate_error(self._progress, str(error))
            raise
        images = sum(1 for item in (result.content or []) if not isinstance(item, str))
        narrate_result(self._progress, str(result.return_value), images, 'steps')
        return result

    async def _exec(self, code: str, session: str | None, timeout_seconds: float | None) -> ToolReturn[str]:
        """Validate, run, and package one call"""
        if not code.strip():
            raise ModelRetry('The `code` argument was empty. Pass Python that uses the pre-imported browser helpers.')
        if session is not None and SESSION_RE.match(session) is None:
            raise ModelRetry(
                f'Invalid session name {session!r}: use 1-64 characters from [A-Za-z0-9_-], starting alphanumeric.'
            )

        session = await self._ensure_cloud_session() or session
        await self._ensure_headless_browser()
        if timeout_seconds is None or timeout_seconds <= 0:
            timeout = self._default_timeout
        else:
            # floor: the first call in a session pays interpreter + daemon cold start
            timeout = min(max(timeout_seconds, min(60.0, self._default_timeout)), _MAX_TIMEOUT)

        started = time.time()
        exit_code, stdout, stderr = await self._execute(code, timeout, session)

        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if exit_code != 0:
            if stderr:
                parts.append(f'[stderr]\n{stderr}')
            parts.append(f'[exit code: {exit_code}]')
        text = self._truncate('\n'.join(parts)) if parts else '(no output -- use print() in the code to return data)'

        images = collect_screenshots(stdout, newer_than=started)
        files = collect_files(stdout, newer_than=started)
        metadata = {'files': files} if files else None
        if not images:
            return ToolReturn[str](return_value=text, metadata=metadata)
        label = 'screenshot' if len(images) == 1 else 'screenshots'
        return ToolReturn[str](
            return_value=text,
            content=[f'The {label} this call captured, attached so you can look at the page:', *images],
            metadata=metadata,
        )

    async def _execute(self, code: str, timeout: float, session: str | None) -> tuple[int, str, str]:
        """Run the model's code in this run's interpreter, starting one if needed"""
        if self._session is None or not self._session.alive:
            self._session = _Session(self._resolve_argv(), self._cwd, self._build_env(session))
            await self._session.start()
        try:
            exit_code, output = await self._session.call(code, timeout)
        except TimeoutError:
            await self._session.close()
            self._session = None
            raise ModelRetry(
                f'browser_exec timed out after {timeout}s, so the browser session was restarted. '
                'The browser itself is still open; re-read what you need from the page.'
            ) from None
        except (OSError, ValueError) as e:  # pragma: no cover - the session died mid-call
            await self._session.close()
            self._session = None
            raise ModelRetry(f'The browser session ended unexpectedly ({e}). Try the call again.') from e
        return exit_code, output, ''

    def _resolve_argv(self) -> list[str]:
        """Locate the CLI, falling back to `uvx browser-use`"""
        resolved = shutil.which('browser-use')
        if resolved is not None:
            return [resolved]
        uvx = shutil.which('uvx')
        if uvx is not None:
            return [uvx, 'browser-use']
        raise ModelRetry(_INSTALL_HINT)

    def _build_env(self, session: str | None) -> dict[str, str]:
        """Inherited environment minus LLM provider keys, plus the browser overlays"""
        env = {
            name: value
            for name, value in os.environ.items()
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in LLM_API_KEY_ENV_PATTERNS)
        }
        if session is not None:
            env['BU_NAME'] = session
        if self._headless_cdp_url is not None:
            env['BU_CDP_URL'] = self._headless_cdp_url
        return env

    def _truncate(self, text: str) -> str:
        """Cap output, keeping the tail"""
        if len(text) <= _MAX_OUTPUT_CHARS:
            return text
        marker = f'[... output truncated, showing last {_MAX_OUTPUT_CHARS} chars]\n'
        return marker + text[-_MAX_OUTPUT_CHARS:]

    async def _run_cli(
        self,
        args: list[str],
        code: str,
        timeout: float,
        env: dict[str, str] | None,
        *,
        raise_on_timeout: bool = True,
    ) -> tuple[int, str, str]:
        """One-shot CLI invocation: pipe `code` on stdin, return (exit, out, err)"""
        try:
            proc = await anyio.open_process(
                args,
                cwd=self._cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=env,
            )
        except OSError as e:
            # which() passed but exec failed (lost +x, TOCTOU)
            raise ModelRetry(f'Failed to launch the browser-use CLI ({args[0]!r}): {e}') from e
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        try:
            assert proc.stdin is not None
            assert proc.stdout is not None
            assert proc.stderr is not None

            async def _write_stdin() -> None:
                assert proc.stdin is not None
                try:
                    await proc.stdin.send(code.encode('utf-8'))
                    await proc.stdin.aclose()
                except (anyio.BrokenResourceError, anyio.ClosedResourceError):  # pragma: no cover
                    pass  # CLI exited before draining stdin

            async def _read_stdout() -> None:
                assert proc.stdout is not None
                async for chunk in proc.stdout:
                    stdout_chunks.append(chunk)

            async def _read_stderr() -> None:
                assert proc.stderr is not None
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)

            try:
                with anyio.fail_after(timeout):
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(_write_stdin)
                        tg.start_soon(_read_stdout)
                        tg.start_soon(_read_stderr)
                    await proc.wait()
            except TimeoutError:
                await self._terminate(proc)
                if not raise_on_timeout:
                    return 1, '', f'timed out after {timeout}s'
                raise ModelRetry(
                    f'browser_exec timed out after {timeout}s. The browser daemon may still be working; '
                    'retry with a larger timeout_seconds, or split the work into smaller calls that '
                    'append intermediate results to files in the workspace.'
                ) from None
        finally:
            await proc.aclose()
        stdout = b''.join(stdout_chunks).decode('utf-8', errors='replace')
        stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')
        exit_code = proc.returncode if proc.returncode is not None else 0
        return exit_code, stdout, stderr

    async def _terminate(self, proc: anyio.abc.Process) -> None:
        """Kill the CLI's process group and reap it"""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:  # pragma: no cover - process already exited
            pass
        with anyio.CancelScope(shield=True):
            await proc.wait()
