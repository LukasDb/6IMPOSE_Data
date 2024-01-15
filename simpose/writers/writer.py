from abc import ABC, abstractmethod
import numpy as np
from typing import Any
import simpose as sp
from pathlib import Path
import signal
import contextlib
from simpose import base_config
import multiprocessing as mp
import yaml
from typing import Callable


class DelayedKeyboardInterrupt:
    def __init__(self, index: int, on_term: None | Callable[[], None] = None) -> None:
        self.index = index
        self._signals = [signal.SIGINT, signal.SIGTERM]
        self._old_handlers = {s: signal.getsignal(s) for s in self._signals}
        self.signal_received: None | int = None
        self._on_term = on_term

    def __enter__(self) -> None:
        for s in self._signals:
            signal.signal(s, self.handler)

    def handler(self, sig: int, frame: Any) -> None:
        if self.signal_received is None:
            self.signal_received = sig
            sp.logger.warn(f"{sig} received. Finishing rendering {self.index}...")

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        if self.signal_received is not None and self._on_term is not None:
            self._on_term()

        for s in self._signals:
            signal.signal(s, self._old_handlers[s])

        if self.signal_received is not None:
            signal.raise_signal(self.signal_received)


class WriterConfig(base_config.BaseConfig):
    output_dir: Path = Path("path/to/output_dir")
    overwrite: bool = False
    start_index: int = 0
    end_index: int = 9
    _gpu_semaphore_comm: None | mp.queues.Queue = None
    _q_rendered: None | mp.queues.Queue = None
    _active_gpu: int = 0

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "output_dir": "Path to the output directory",
            "overwrite": "If True, overwrite existing files",
            "start_index": "Start index of the generated data",
            "end_index": "End index of the generated data (including)",
        }


class Writer(ABC):
    def __init__(self, params: WriterConfig, comm: Any) -> None:
        # self.scene = scene
        self.output_dir = params.output_dir.expanduser()
        self.overwrite = params.overwrite
        self.start_index = params.start_index
        self.end_index = params.end_index
        # self.gpu_semaphore = device_setup.get("gpu_semaphore", contextlib.nullcontext())
        # comm = device_setup["gpu_semaphore"]

        self.gpu_semaphore: sp.RemoteSemaphore = sp.RemoteSemaphore(comm)

        # self.gpu_semaphore: sp.RemoteSemaphore = sp.RemoteSemaphore(
        #    comm=params._gpu_semaphore_comm
        # )
        self.q_rendered: mp.Queue | None = params._q_rendered

        # self.q_rendered: mp.Queue | None = device_setup.get("q_rendered", None)

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, type: None, value: None, traceback: None) -> None:
        pass

    def dump_config(self, config: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(yaml.dump(config))

    @abstractmethod
    def get_pending_indices(self) -> np.ndarray:
        """determine which indices to generate according the current config"""
        pass

    def write_data(self, scene: sp.Scene, dataset_index: int) -> None:
        """dont allow CTRl+C during data generation"""
        scene.set_output_path(self.output_dir)

        with DelayedKeyboardInterrupt(dataset_index, on_term=lambda: self._cleanup(dataset_index)):
            try:
                self._write_data(scene, dataset_index)
                if self.q_rendered is not None:
                    self.q_rendered.put(1)
            except Exception as e:
                # clean up possibly corrupted data
                sp.logger.critical(f"Error while generating data no. {dataset_index}")
                sp.logger.critical(e)
                self._cleanup(dataset_index)
                raise e

    def post_process(self) -> None:
        pass

    @abstractmethod
    def _write_data(self, scene: sp.Scene, dataset_index: int) -> None:
        pass

    @abstractmethod
    def _cleanup(self, dataset_index: int) -> None:
        pass
