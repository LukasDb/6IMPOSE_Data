"""Stack dumping for hung processes.

A worker can wedge inside a C extension (the GL/EGL driver, CUDA, pybullet)
while holding the GIL: every thread of the process then sits in `futex_wait`
and no Python code -- not even a watchdog thread inside that process -- ever
runs again. Attaching py-spy/gdb after the fact needs ptrace permission on a
process that is not a descendant of the debugging shell, which the default
`kernel.yama.ptrace_scope=1` denies.

`faulthandler` works in both cases: its handler runs on the signal stack and
walks the interpreter frames without acquiring the GIL. So every simpose
process installs a SIGUSR1 handler at import time that dumps all thread stacks
to a per-pid file, and the generator's watchdog signals a stalled worker before
killing it -- the traceback outlives the process that produced it.

    kill -USR1 <pid>            # then read /tmp/simpose_stacks/<pid>.txt

Override the directory with SIMPOSE_DIAG_DIR.
"""

import atexit
import faulthandler
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import IO

# module-level, because faulthandler keeps only the file *descriptor*: if the
# object is garbage collected the fd is closed and the dump goes nowhere
_dump_file: IO[str] | None = None


def stack_dump_dir() -> Path:
    return Path(os.environ.get("SIMPOSE_DIAG_DIR", tempfile.gettempdir())) / "simpose_stacks"


def stack_dump_path(pid: int | None = None) -> Path:
    return stack_dump_dir() / f"{pid if pid is not None else os.getpid()}.txt"


def _discard_if_empty(path: Path) -> None:
    try:
        if path.stat().st_size == 0:
            path.unlink()
    except OSError:
        pass


def install_stack_dumper() -> None:
    """dump all thread stacks on SIGUSR1 (and on a hard crash) to a per-pid file"""
    global _dump_file
    if _dump_file is not None:
        return
    try:
        stack_dump_dir().mkdir(parents=True, exist_ok=True)
        # faulthandler needs the fd up front, so the (empty) file is created for
        # every process; drop it again on a clean exit so only real dumps remain.
        # A killed worker skips this, but its file is exactly the one worth keeping
        _dump_file = stack_dump_path().open("a", buffering=1)
        atexit.register(_discard_if_empty, stack_dump_path())
        faulthandler.enable(file=_dump_file, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=_dump_file, all_threads=True, chain=False)
    except (OSError, ValueError, AttributeError):
        # diagnostics must never take the run down with them
        _dump_file = None


def request_stack_dump(pid: int | None, grace: float = 2.0) -> Path | None:
    """ask another simpose process for its thread stacks and give the handler
    `grace` seconds to write them. Call this *before* killing a hung worker."""
    if pid is None:
        return None
    try:
        os.kill(pid, signal.SIGUSR1)
    except (ProcessLookupError, PermissionError, OSError):
        return None
    time.sleep(grace)
    return stack_dump_path(pid)
