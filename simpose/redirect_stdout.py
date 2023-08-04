import os
import sys
from contextlib import contextmanager
import logging


@contextmanager
def redirect_stdout():
    mute = logging.getLogger().level >= logging.DEBUG
    if not mute:
        yield  # do nothing
        return

    fd_devnull = os.devnull
    fd_out = sys.stdout.fileno()
    fd_err = sys.stderr.fileno()

    def _redirect_stdout(to_out, to_err):
        sys.stdout.close()  # + implicit flush()
        sys.stderr.close()  # + implicit flush()

        os.dup2(to_out.fileno(), fd_out)  # fd writes to 'to' file
        os.dup2(to_err.fileno(), fd_err)  # fd writes to 'to' file

        sys.stdout = os.fdopen(fd_out, "w")  # Python writes to fd
        sys.stderr = os.fdopen(fd_err, "w")  # Python writes to fd
        

    with os.fdopen(os.dup(fd_out), "w") as old_stdout, os.fdopen(os.dup(fd_err), "w") as old_stderr:
        with open(fd_devnull, "w") as devnull:
            _redirect_stdout(to_out=devnull, to_err=devnull)
        try:
            yield
        finally:
            _redirect_stdout(to_out=old_stdout, to_err=old_stderr)
