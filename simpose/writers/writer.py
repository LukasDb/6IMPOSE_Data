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
    def __init__(self, params: WriterConfig, comm: Any | None = None) -> None:
        # self.scene = scene
        self.output_dir = params.output_dir.expanduser()
        self.overwrite = params.overwrite
        self.start_index = params.start_index
        self.end_index = params.end_index
        self.gpu_semaphore: sp.RemoteSemaphore | None = (
            sp.RemoteSemaphore(comm) if comm is not None else None
        )
        self.q_rendered: mp.Queue | None = params._q_rendered

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, type: None, value: None, traceback: None) -> None:
        if self.gpu_semaphore is not None:
            self.gpu_semaphore.close()

    def dump_config(self, config: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "config.yaml", "w") as f:
            f.write(yaml.dump(config))

    @abstractmethod
    def get_pending_indices(self) -> np.ndarray:
        """determine which indices to generate according the current config"""
        pass

    def write_data(
        self,
        dataset_index: int,
        *,
        scene: sp.Scene | None = None,
        render_product: sp.RenderProduct | None = None,
    ) -> None:
        """dont allow CTRl+C during data generation"""
        if scene is not None:
            scene.set_output_path(self.output_dir)

        # with DelayedKeyboardInterrupt(dataset_index, on_term=lambda: self._cleanup(dataset_index)):
        try:
            self._write_data(dataset_index, scene=scene, render_product=render_product)
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
    def _write_data(
        self,
        dataset_index: int,
        *,
        scene: sp.Scene | None,
        render_product: sp.RenderProduct | None,
    ) -> None:
        pass

    @abstractmethod
    def _cleanup(self, dataset_index: int) -> None:
        pass
