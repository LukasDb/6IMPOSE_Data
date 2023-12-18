from abc import ABC, abstractmethod
import simpose as sp
from pathlib import Path
import signal
import contextlib
from simpose import base_config
import multiprocessing as mp


class DelayedKeyboardInterrupt:
    def __init__(self, index) -> None:
        self.index = index

    def __enter__(self) -> None:
        self.signal_received = False
        self.termsignal_received = False
        self.old_handler: signal._HANDLER = signal.signal(signal.SIGINT, self.handler)

        self.oldterm_handler: signal._HANDLER = signal.signal(signal.SIGTERM, self.term_handler)

    def handler(self, sig, frame) -> None:
        self.signal_received = (sig, frame)
        sp.logger.warn(f"SIGINT received. Finishing rendering {self.index}...")

    def term_handler(self, sig, frame) -> None:
        self.termsignal_received = (sig, frame)
        sp.logger.warn(f"SIGTERM received. Finishing rendering {self.index}...")

    def __exit__(self, type, value, traceback) -> None:
        signal.signal(signal.SIGINT, self.old_handler)
        signal.signal(signal.SIGTERM, self.oldterm_handler)

        if self.signal_received:
            self.old_handler(*self.signal_received)  # type: ignore

        if self.termsignal_received:
            self.oldterm_handler(*self.termsignal_received)  # type: ignore


class WriterConfig(base_config.BaseConfig):
    output_dir: Path = Path("path/to/output_dir")
    overwrite: bool = False
    start_index: int = 0
    end_index: int = 9

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "output_dir": "Path to the output directory",
            "overwrite": "If True, overwrite existing files",
            "start_index": "Start index of the generated data",
            "end_index": "End index of the generated data (including)",
        }


class Writer(ABC):
    def __init__(self, params: WriterConfig, gpu_semaphore=None):
        # self.scene = scene
        self.output_dir = params.output_dir.expanduser()
        self.overwrite = params.overwrite
        self.start_index = params.start_index
        self.end_index = params.end_index
        self.gpu_semaphore = contextlib.nullcontext() if gpu_semaphore is None else gpu_semaphore

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    @abstractmethod
    def get_pending_indices(self) -> list[int]:
        """determine which indices to generate according the current config"""
        pass

    def write_data(self, scene: sp.Scene, dataset_index: int):
        """dont allow CTRl+C during data generation"""
        scene.set_output_path(self.output_dir)

        with DelayedKeyboardInterrupt(dataset_index):
            try:
                self._write_data(scene, dataset_index)
            except Exception as e:
                # clean up possibly corrupted data
                sp.logger.error(f"Error while generating data no. {dataset_index}")
                sp.logger.error(e)
                self._cleanup(dataset_index)
                raise e

    def post_process(self):
        pass

    @abstractmethod
    def _write_data(self, scene: sp.Scene, dataset_index: int):
        pass

    @abstractmethod
    def _cleanup(self, dataset_index: int):
        pass
