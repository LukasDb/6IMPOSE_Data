import os
import sys
from contextlib import contextmanager
import logging


@contextmanager
def redirect_stdout():
    to = os.devnull if logging.getLogger().level >= logging.INFO else sys.stdout
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)
