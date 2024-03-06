from math import e
import multiprocessing as mp
import multiprocessing.queues as mpq
from queue import Queue
import queue
from random import shuffle
import time
from typing import Any, Tuple
import simpose as sp
from enum import Enum


class SignalType(Enum):
    REQUEST_RELEASE = "r_rls"
    REQUEST_ACQUIRE = "r_acq"
    GRANTED_ACQUIRE = "g_acq"
    CLOSED = "closed"


class MainSemaphore:
    def __init__(self, value: int = 1, timeout: float | None = None):
        self._comms: list[Queue[Tuple[str, SignalType]]] = []
        self._timeout = timeout
        self._start_value = value
        self._value = value
        self._pending: list[SignalType] = []
        self._timers: dict[str, float] = {}

    def get_comm(self) -> Queue[(str, SignalType)]:
        """called from main to get a new comm for a worker"""
        comm: Queue[Tuple[str, SignalType]] = mp.Queue()  # type: ignore
        self._comms.append(comm)
        return comm

    def release_for(self, process_name: str) -> None:
        """non blocking release semaphore for a specific process"""
        self.grant_release(process_name)

    def is_acquired(self, process_name: str | None = None) -> bool:
        """non blocking check if semaphore is acquired (by any other process, or a specific one)"""
        acquired_processes = self._timers.keys()
        if process_name is None:
            return len(acquired_processes) > 0
        else:
            return process_name in acquired_processes

    def grant_acquire(self, comm: Queue[tuple[str, SignalType]], process_name: str) -> None:
        comm.put(("", SignalType.GRANTED_ACQUIRE))
        self._timers[process_name] = time.time()  #  timer
        self._value -= 1
        sp.logger.debug(f"ACQUIRED for {process_name} now: {self._value}")

    def grant_release(self, process_name: str) -> None:
        self._timers.pop(process_name)
        if self._value < self._start_value:
            self._value += 1
        sp.logger.debug(f"RELEASED for {process_name} now: {self._value}")

    def run(self) -> list[str]:
        """non blocking call to operate the semaphore. If a timeout is
        detected the respective worker is terminated. A list of the
        terminated mp.Process.name is returned"""

        # process the signals (put back in private queue if not ready yet)
        to_be_removed = []

        for comm in self._comms:
            try:
                process_name, signal = comm.get(block=False)
            except queue.Empty:
                continue

            # if signal == SignalType.REQUEST_ACQUIRE and self._value == 0:
            #     # ignore request for now, but remember
            #     sp.logger.debug(f"Ignoring request from {process_name} for now")

            if signal == SignalType.REQUEST_ACQUIRE and self._value > 0:
                self.grant_acquire(comm, process_name)

            elif signal == SignalType.REQUEST_RELEASE:
                self.grant_release(process_name)

            elif signal == SignalType.CLOSED:
                to_be_removed.append(comm)
                sp.logger.debug(f"Removed comm for {process_name}")
            else:
                # just put back
                comm.put((process_name, signal))

        for c in to_be_removed:
            self._comms.remove(c)

        # 3) check if timed out and kill process
        terminated: list[str] = []
        for name, t_started in dict(self._timers).items():
            if self._timeout is None or (time.time() - t_started) < self._timeout:
                continue

            # semaphore timed out -> allow new acquire
            proc = [p for p in mp.active_children() if p.name == name]
            sp.logger.error(f"Semaphore timed out for {name}. Terminating process.")
            if len(proc) == 1:
                # kill if found (otherwise it's already dead)
                proc[0].kill()

            if self._value < self._start_value:
                self._value += 1
            try:
                self._timers.pop(name)  # remove timer
            except KeyError:
                pass
            terminated.append(name)

        return terminated


class RemoteSemaphore:
    """A semaphore that can be acquired and released from a different process with Timeout."""

    def __init__(self, comm: Queue[tuple[str, SignalType]]) -> None:
        self._comm: Queue[tuple[str, SignalType]] = comm
        self.name = mp.current_process().name

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
        sp.logger.debug(f"{self.name} wants to acquire {self}")
        self._request(SignalType.REQUEST_ACQUIRE)
        self._confirm(SignalType.GRANTED_ACQUIRE)

    def release(self) -> None:
        """if this is not called, eventually the semaphore will time out"""
        sp.logger.debug(f"{self.name} wants to release {self}")
        self._request(SignalType.REQUEST_RELEASE)

    def _request(self, request_type: SignalType) -> None:
        self._comm.put((self.name, request_type))

    def _confirm(self, grant_type: SignalType) -> None:
        while True:
            name, signal = self._comm.get()  # just wait for a token
            if signal == grant_type:
                return
            else:
                self._comm.put((name, signal))
                time.sleep(0.5)

    def close(self) -> None:
        self._comm.put((self.name, SignalType.CLOSED))
