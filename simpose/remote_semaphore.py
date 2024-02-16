from math import e
import multiprocessing as mp
import multiprocessing.queues as mpq
import queue
import time
from typing import Any
import simpose as sp


class SignalType:
    REQUEST_RELEASE = "r_rls"

    REQUEST_ACQUIRE = "r_acq"
    GRANTED_ACQUIRE = "g_acq"


class RemoteSemaphore:
    """A semaphore that can be acquired and released from a different process with Timeout."""

    MAIN = "main"

    def __init__(
        self,
        comm: mpq.Queue,
        value: int = 1,
        timeout: float | None = None,
    ):
        self._comm: mpq.Queue[dict] = comm
        self._pending: list[dict] = []
        self._timeout = timeout
        self._start_value = value
        self._value = value
        self._timers: dict[str, float] = {}
        self.name = mp.current_process().name

    def start(self) -> None:
        return

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
        self._request(SignalType.REQUEST_ACQUIRE, self.name)
        self._confirm(SignalType.GRANTED_ACQUIRE)
        sp.logger.debug(f"{self.name} acquired {self}")

    def release(self, process_name: str | None = None) -> None:
        """if this is not called, eventually the semaphore will time out
        optionally, can release the semaphore of another process"""
        name = self.name if process_name is None else process_name

        self._request(SignalType.REQUEST_RELEASE, name)
        sp.logger.debug(f"{name} released {self}")

    def _request(self, request_type: str, sender: str) -> None:
        self._comm.put({"type": request_type, "sender": sender, "receiver": self.MAIN})

    def _confirm(self, grant_type: str) -> None:
        while True:
            signal = self._comm.get()
            if signal["receiver"] == self.name and signal["type"] == grant_type:
                return
            self._comm.put(signal)
            time.sleep(0.2)  # important to prevent deadlocks

    def is_acquired(self, process_name: str | None = None) -> bool:
        """non blocking check if semaphore is acquired (by any other process, or a specific one)"""
        acquired_processes = self._timers.keys()
        if process_name is None:
            return len(acquired_processes) > 0
        else:
            return process_name in acquired_processes

    def grant_acquire(self, process_name: str) -> None:
        self._comm.put(
            {
                "type": SignalType.GRANTED_ACQUIRE,
                "sender": self.name,
                "receiver": process_name,
            }
        )
        self._timers[process_name] = time.time()  # start timer
        self._value -= 1
        sp.logger.debug(f"GRANTED ACQURE for {process_name} now: {self._value}")

    def grant_release(self, process_name: str) -> None:
        self._value += 1
        self._timers.pop(process_name)
        sp.logger.debug(f"GOT RELEASE for {process_name} now: {self._value}")

    def run(self) -> list[str]:
        """non blocking call to operate the semaphore. If a timeout is
        detected the respective worker is terminated. A list of the
        terminated mp.Process.name is returned"""

        # drain signals into private queue
        signals = []
        while True:
            try:
                time.sleep(0.05)
                signal = self._comm.get(block=False)
                signals.append(signal)
            except queue.Empty:
                break


        for s in self._pending:
            if self._value > 0:
                self.grant_acquire(s["sender"])
                self._pending.remove(s)

        # process the signals (put back in private queue if not ready yet)
        for signal in signals:
            if signal["receiver"] != self.MAIN:
                self._comm.put(signal)
                continue

            if signal["type"] == SignalType.REQUEST_ACQUIRE and self._value == 0:
                # ignore request for now, but remember
                if signal not in self._pending:
                    self._pending.append(signal)
                sp.logger.debug(f"Ignoring request from {signal['sender']} for now")

            elif signal["type"] == SignalType.REQUEST_ACQUIRE and self._value > 0:
                self.grant_acquire(signal["sender"])

            elif signal["type"] == SignalType.REQUEST_RELEASE:
                self.grant_release(signal["sender"])

            else:
                raise AssertionError(f"Unknown signal type for main: {signal['type']}")

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

            self._value += 1
            self._timers.pop(name)  # remove timer

            terminated.append(name)

        return terminated
