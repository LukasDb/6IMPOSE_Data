import multiprocessing as mp
import multiprocessing.queues
import queue
import time
from typing import Any
import simpose as sp


class RemoteSemaphore:
    """A semaphore that can be acquired and released from a different process with Timeout."""

    RELEASED = "released"
    IDLE = "idle"
    ACQUIRED = "acquired"

    def __init__(
        self,
        comm: multiprocessing.queues.Queue,
        value: int = 1,
        timeout: float | None = None,
    ):
        self._comm: multiprocessing.queues.Queue[Any] = comm
        self._timeout = timeout
        self._value = value
        self._timers: dict[str, float] = {}

    def start(self) -> None:
        for _ in range(self._value):
            self._comm.put((self.IDLE, mp.current_process().name))

    def __enter__(self) -> "RemoteSemaphore":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.release()

    def __str__(self) -> str:
        return f"RemoteSemaphore({self._comm})"

    # in the worker:
    def acquire(self) -> None:
        """blocking acquire semaphore"""
        sp.logger.debug(f"{mp.current_process().name} wants to acquire {self}")

        while True:
            signal, name = self._comm.get()  # blocking
            if signal == self.IDLE:
                self._comm.put((self.ACQUIRED, mp.current_process().name))
                sp.logger.debug(f"{mp.current_process().name} acquired {self}")
                break
            else:
                # put back "acquired"|"released"
                self._comm.put((signal, name))
                time.sleep(0.1)

    def release(self, process_name: str | None = None) -> None:
        """if this is not called, eventually the semaphore will time out
        optionally, can release the semaphore of another process"""
        if process_name is None:
            process_name = mp.current_process().name
        self._comm.put((self.RELEASED, process_name))
        sp.logger.debug(f"{process_name} released {self}")

    def is_acquired(self, process_name: str | None = None) -> bool:
        """non blocking check if semaphore is acquired (by any other process, or a specific one)"""
        acquired_processes = self._timers.keys()
        if process_name is None:
            return len(acquired_processes) > 0
        else:
            return process_name in acquired_processes

    def run(self) -> list[str]:
        """non blocking call to operate the semaphore. If a timeout is
        detected the respective worker is terminated. A list of the
        terminated mp.Process.name is returned"""

        # 1) capture pending signals
        captured = []
        while True:
            try:
                captured.append(self._comm.get(block=False))  # non blocking
            except queue.Empty:
                break

        # 2) process captured signals
        for signal, name in captured:
            if signal == self.IDLE:
                # retrieved own idle signal -> put back
                self._comm.put((self.IDLE, name))

            elif signal == self.ACQUIRED:
                # start timer
                self._timers[name] = time.time()

            elif signal == self.RELEASED:
                # retrieved release signal -> allow new acquire
                self._timers.pop(name)  # remove timer
                self._comm.put((self.IDLE, mp.current_process().name))

        # 3) check if timed out and kill process
        terminated: list[str] = []
        for name, t_started in self._timers.items():
            if self._timeout is not None and (time.time() - t_started) > self._timeout:
                # semaphore timed out -> allow new acquire
                proc = [p for p in mp.active_children() if p.name == name]
                sp.logger.error(f"Semaphore timed out for {name}. Terminating process.")
                if len(proc) == 1:
                    # kill if found (otherwise it's already dead)
                    proc[0].kill()

                self._comm.put((self.IDLE, mp.current_process().name))
                terminated.append(name)

        for name in terminated:
            self._timers.pop(name)  # remove timer

        return terminated
