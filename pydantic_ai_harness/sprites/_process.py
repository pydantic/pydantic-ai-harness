"""Remote process-group supervision for Sprites command transport.

The SDK disconnects on timeout without exposing the new exec session ID. A private
control directory coordinates cancellation even when it arrives before process startup.
https://sprites.dev/api/sprites/exec
"""

RUN = """
import fcntl, json, os, shutil, signal, subprocess, sys
control, options = sys.argv[1], json.loads(sys.argv[2])
os.makedirs(control, mode=0o700, exist_ok=True)
process = None
try:
    with open(control + '/lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(control + '/cancel'):
            sys.exit(0)
        env = dict(os.environ, **options['env'])
        env.pop('BASH_ENV', None)
        env.pop('ENV', None)
        process = subprocess.Popen(options['args'], cwd=options['cwd'], env=env, start_new_session=True)
        with open(control + '/pid', 'w') as pid:
            pid.write(str(process.pid))
    code = process.wait()
    sys.exit(code if code >= 0 else 128 - code)
finally:
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    shutil.rmtree(control, ignore_errors=True)
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
