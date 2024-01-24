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

    def release(self, process_name: str | None = None) -> None:
        """if this is not called, eventually the semaphore will time out
        optionally, can release the semaphore of another process"""
        name = self.name if process_name is None else process_name

        self._request(SignalType.REQUEST_RELEASE, name)

        sp.logger.debug(f"{process_name} released {self}")

    def _request(self, request_type: str, sender: str) -> None:
        self._comm.put({"type": request_type, "sender": sender, "receiver": self.MAIN})

    def _confirm(self, grant_type: str) -> None:
        while True:
            signal = self._comm.get()
            if signal["receiver"] == self.name and signal["type"] == grant_type:
                break
            else:
                # put back
                self._comm.put(signal)
                time.sleep(0.1)

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

        # 1) capture all pending signals
        captured_signal: list[dict] = []
        while True:
            try:
                captured_signal.append(self._comm.get(block=False))  # non blocking
            except queue.Empty:
                break

        # 2) process captured signals
        for signal in captured_signal:
            if signal["receiver"] != self.MAIN:
                # put back
                self._comm.put(signal)
                continue

            if signal["type"] == SignalType.REQUEST_ACQUIRE and self._value > 0:
                self._value -= 1
                self._comm.put(
                    {
                        "type": SignalType.GRANTED_ACQUIRE,
                        "sender": self.name,
                        "receiver": signal["sender"],
                    }
                )
                self._timers[signal["sender"]] = time.time()  # start timer

            elif signal["type"] == SignalType.REQUEST_RELEASE:
                self._value += 1
                self._timers.pop(signal["sender"])  # remove timer
            else:
                raise AssertionError(f"Unknown signal type: {signal['type']} for main process!")

        # 3) check if timed out and kill process
        terminated: list[str] = []
        for name, t_started in dict(self._timers).items():
            if self._timeout is not None and (time.time() - t_started) > self._timeout:
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
