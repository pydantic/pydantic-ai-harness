"""Remote process-group supervision for Sprites command transport.

The SDK disconnects on timeout without exposing the new exec session ID. A private
control directory coordinates cancellation even when it arrives before process startup.
https://sprites.dev/api/sprites/exec

`RUN` kills the command's process group when the command exits, so a plain `cmd &` child
does not outlive its command; a process that starts its own session (`setsid`, as `Shell`
background jobs do) does.

The control directory is not a security boundary: the command runs as the same user as its
supervisor, so it can delete the directory (after which `CANCEL` finds nothing to signal) or
simply `setsid` out of its group. The Sprite is the boundary; the deadline and cancellation
cover commands that do not work against them.
"""

SPAWN_FAILED = 'pydantic-ai-spawn-failed:'
"""Prefix of the stderr `RUN` writes, exiting `SPAWN_FAILED_EXIT`, when the command cannot be started."""
SPAWN_FAILED_EXIT = 127

RUN = f"""
SPAWN_FAILED, SPAWN_FAILED_EXIT = {SPAWN_FAILED!r}, {SPAWN_FAILED_EXIT}
import contextlib, fcntl, json, os, shutil, signal, subprocess, sys
control, options = sys.argv[1], json.loads(sys.argv[2])
os.makedirs(control, mode=0o700, exist_ok=True)
lock = open(control + '/lock', 'w')
process = None
try:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if os.path.exists(control + '/cancel'):
        sys.exit(0)
    # The base is the Sprite's own exec environment. `BASH_ENV` and `ENV` are dropped because they
    # make a shell source the file they name, which would run arbitrary startup code before every
    # command.
    env = dict(os.environ, **options['env'])
    env.pop('BASH_ENV', None)
    env.pop('ENV', None)
    try:
        process = subprocess.Popen(options['args'], cwd=options['cwd'], env=env, start_new_session=True)
    except OSError as error:
        # Reported in a form the backend turns back into the builtin error, not a traceback exit.
        failure = [type(error).__name__, error.errno, error.strerror, error.filename]
        sys.stderr.write(SPAWN_FAILED + json.dumps(failure))
        sys.exit(SPAWN_FAILED_EXIT)
    with open(control + '/pid', 'w') as pid:
        pid.write(str(process.pid))
    fcntl.flock(lock, fcntl.LOCK_UN)
    # Wait without reaping: until the exited leader is reaped below, its PID, and so the group ID
    # in the pid file, cannot be reused.
    os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT)
finally:
    code = 0
    if process is not None:
        # Under the lock, the pid file goes before the leader is reaped, so `CANCEL` only ever
        # signals a group whose ID still belongs to this command.
        fcntl.flock(lock, fcntl.LOCK_EX)
        with contextlib.suppress(FileNotFoundError):
            os.remove(control + '/pid')
        # EPERM is how macOS (where the test fake runs this) answers for a group holding only a zombie.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        code = process.wait()
    shutil.rmtree(control, ignore_errors=True)
sys.exit(code if code >= 0 else 128 - code)
"""

CANCEL = """
import fcntl, os, signal, sys
control = sys.argv[1]
os.makedirs(control, mode=0o700, exist_ok=True)
try:
    with open(control + '/lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        open(control + '/cancel', 'w').close()
        try:
            with open(control + '/pid') as pid:
                os.killpg(int(pid.read()), signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
except FileNotFoundError:
    pass
"""
