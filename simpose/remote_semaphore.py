import multiprocessing as mp
from queue import Queue
import queue
import time
from typing import Any, NamedTuple, Tuple
import simpose as sp
from enum import Enum


class SignalType(Enum):
    REQUEST_RELEASE = "r_rls"
    REQUEST_ACQUIRE = "r_acq"
    GRANTED_ACQUIRE = "g_acq"
    CLOSED = "closed"


class Comm(NamedTuple):
    """The pair of one-way queues connecting the main process to one worker.

    Two *directed* queues rather than one shared queue is load-bearing, not
    stylistic: with a single queue, a worker blocking in `_confirm` reads back
    its own REQUEST_ACQUIRE (it is the only reader besides the main process,
    and it starts reading immediately after writing). It then has to re-queue
    the message, which lands it behind the main process's own re-queued
    "not grantable yet" token -- whose `break` stops the drain loop before the
    request is ever seen again. Both sides then wait forever. Splitting the
    directions means neither side can ever consume a message meant for itself.
    """

    requests: "Queue[Tuple[str, SignalType]]"  # worker -> main
    grants: "Queue[Tuple[str, SignalType]]"  # main -> worker


class MainSemaphore:
    def __init__(self, value: int = 1, timeout: float | None = None):
        self._comms: list[Comm] = []
        self._timeout = timeout
        self._start_value = value
        self._value = value
        # name -> time the worker started waiting for / holding the semaphore.
        # A worker is timed whether it is *waiting* or *holding*, so a worker
        # that never gets granted is still covered by the watchdog below.
        self._timers: dict[str, float] = {}
        # names that are waiting for a grant, oldest first (FIFO, so a worker
        # cannot be starved by later arrivals)
        self._waiting: list[tuple[str, Comm]] = []
        self._held: set[str] = set()

    def get_comm(self) -> Comm:
        """called from main to get a new comm for a worker"""
        comm = Comm(requests=mp.Queue(), grants=mp.Queue())  # type: ignore[arg-type]
        self._comms.append(comm)
        return comm

    def remove_comm(self, comm: Comm) -> None:
        """called from main to drop a dead worker's comm, so it is no longer polled"""
        self._comms = [c for c in self._comms if c is not comm]
        self._waiting = [w for w in self._waiting if w[1] is not comm]

    def release_for(self, process_name: str) -> None:
        """non blocking release semaphore for a specific process"""
        self.grant_release(process_name)

    def is_acquired(self, process_name: str | None = None) -> bool:
        """non blocking check if semaphore is acquired (by any other process, or a specific one)"""
        if process_name is None:
            return len(self._held) > 0
        else:
            return process_name in self._held

    def grant_acquire(self, comm: Comm, process_name: str) -> None:
        comm.grants.put((process_name, SignalType.GRANTED_ACQUIRE))
        self._timers[process_name] = time.time()  # restart timer: now holding
        self._held.add(process_name)
        self._value -= 1
        sp.logger.debug(f"ACQUIRED for {process_name} now: {self._value}")

    def grant_release(self, process_name: str) -> None:
        # a worker that is still only *waiting* can also be dropped (crash /
        # timeout), so clear it from both sets regardless of which one it is in
        self._waiting = [w for w in self._waiting if w[0] != process_name]
        self._timers.pop(process_name, None)
        if process_name not in self._held:
            return
        self._held.discard(process_name)
        if self._value < self._start_value:
            self._value += 1
        sp.logger.debug(f"RELEASED for {process_name} now: {self._value}")

    def _pump_waiting(self) -> None:
        """hand the semaphore to waiting workers, oldest first, while slots last"""
        while self._value > 0 and len(self._waiting) > 0:
            process_name, comm = self._waiting.pop(0)
            self.grant_acquire(comm, process_name)

    def run(self) -> list[str]:
        """non blocking call to operate the semaphore. If a timeout is
        detected the respective worker is terminated. A list of the
        terminated mp.Process.name is returned"""

        # drain each request queue completely -- nothing is ever put back, so
        # the loop cannot spin on its own message and a worker never waits a
        # full poll interval per message
        to_be_removed = []

        for comm in self._comms:
            while True:
                try:
                    process_name, signal = comm.requests.get(block=False)
                except queue.Empty:
                    break

                if signal == SignalType.REQUEST_ACQUIRE:
                    # queue it; the pump below grants it when a slot is free.
                    # start the clock now so a worker that never gets granted
                    # still trips the watchdog instead of hanging forever
                    self._timers.setdefault(process_name, time.time())
                    self._waiting.append((process_name, comm))

                elif signal == SignalType.REQUEST_RELEASE:
                    self.grant_release(process_name)

                elif signal == SignalType.CLOSED:
                    self.grant_release(process_name)
                    to_be_removed.append(comm)
                    sp.logger.debug(f"Removed comm for {process_name}")
                    break

        self._comms = [c for c in self._comms if c not in to_be_removed]
        self._waiting = [w for w in self._waiting if w[1] not in to_be_removed]

        self._pump_waiting()

        # 3) check if timed out and kill process
        to_be_terminated = []
        for name, t_started in dict(self._timers).items():
            if self._timeout is None or (time.time() - t_started) < self._timeout:
                continue
            # timeout occured
            to_be_terminated.append(name)

        terminated: list[str] = []
        for name in to_be_terminated:
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

    def __init__(self, comm: Comm) -> None:
        self._comm = comm
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
        self._comm.requests.put((self.name, request_type))

    def _confirm(self, grant_type: SignalType) -> None:
        # only the main process writes to `grants`, and only this worker reads
        # it, so a blocking get cannot swallow our own message: whatever
        # arrives is a grant addressed to us
        while True:
            _, signal = self._comm.grants.get()
            if signal == grant_type:
                return
            sp.logger.debug(f"{self.name} ignoring unexpected signal {signal}")

    def close(self) -> None:
        self._comm.requests.put((self.name, SignalType.CLOSED))
