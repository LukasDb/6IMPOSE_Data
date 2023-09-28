from abc import ABC, abstractmethod
import simpose as sp
from pathlib import Path
import logging
import signal
from pydantic import BaseModel


class DelayedKeyboardInterrupt:
    def __init__(self, index) -> None:
        self.index = index

    def __enter__(self) -> None:
        self.signal_received = False
        self.old_handler: signal._HANDLER = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame) -> None:
        self.signal_received = (sig, frame)
        sp.logger.warn(f"SIGINT received. Finishing rendering {self.index}...")

    def __exit__(self, type, value, traceback) -> None:
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)  # type: ignore


class WriterParams(BaseModel, extra="forbid"):
    output_dir: Path
    overwrite: bool
    start_index: int
    end_index: int


class Writer(ABC):
    def __init__(self, params: WriterParams):
        # self.scene = scene
        self.output_dir = params.output_dir.expanduser()
        # self.scene.set_output_path(self.output_dir)
        self.overwrite = params.overwrite
        self.start_index = params.start_index
        self.end_index = params.end_index

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

    @abstractmethod
    def _write_data(self, scene: sp.Scene, dataset_index: int):
        pass

    @abstractmethod
    def _cleanup(self, dataset_index: int):
        pass
