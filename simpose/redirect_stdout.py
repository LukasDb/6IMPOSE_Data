import simpose as sp
import copy
import os
import sys
import contextlib
from contextlib import contextmanager
import logging
import multiprocessing as mp


@contextmanager
def redirect_stdout():
    # if called from blender built-in python, dont use this
    py_path = sys.executable
    is_blender_builtin = "Blender.app" in py_path

    if is_blender_builtin or sp.logger.level < logging.DEBUG:
        yield
        return

    fd = sys.stdout.fileno()

    # THIS IS IMPORTANT. OVERWRITING SYS.STDOUT IS NOT ENOUGH
    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            # with open(f"{mp.current_process().name}.log", "a") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different
