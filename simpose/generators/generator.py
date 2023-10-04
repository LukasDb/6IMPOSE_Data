import signal
import simpose as sp
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np


from simpose import base_config


class GeneratorParams(ABC, base_config.BaseConfig):
    n_workers: int


class Generator(ABC):
    writer: sp.writers.Writer
    randomizers: dict[str, sp.random.Randomizer]
    params: GeneratorParams

    def __init__(
        self,
        writer: sp.writers.Writer,
        randomizers: dict[str, sp.random.Randomizer],
        params: GeneratorParams,
    ):
        self.writer = writer
        self.randomizers = randomizers
        self.params = params

        # if debug:
        #     sp.logger.setLevel(0)
        #     set to temp dir
        #     scene = generate_data([0])
        #     scene.export_blend()
        #     scene.run_simulation()
        #     exit()

    def start(self):
        mp.set_start_method("spawn")

        pending_indices = self.writer.get_pending_indices()
        if len(pending_indices) == 0:
            sp.logger.info("No images to render.")
            return

        n_workers = min(self.params.n_workers, len(pending_indices))

        if n_workers == 1:
            sp.logger.debug("Using single process.")
            self.generate_data(pending_indices)
            return

        sp.logger.info("Rendering %d images with %d workers.", len(pending_indices), n_workers)

        # indices = np.array_split(pending_indices, n_workers)
        chunk_size = 200
        indices = np.array_split(pending_indices, len(pending_indices) // chunk_size)

        current_procs = []
        for indlist in indices:
            if len(current_procs) == n_workers:
                for p in current_procs:
                    p.join()
                current_procs = []

            proc = mp.Process(target=self.generate_data, args=(indlist,), daemon=True)
            current_procs.append(proc)
            proc.start()

    def process(self, queue: mp.Queue):
        np.random.seed(mp.current_process().pid)
        import importlib

        importlib.reload(sp)

        while True:
            indices = queue.get()
            if indices is None:
                break
            self.generate_data(indices)

    @abstractmethod
    def generate_data(self, indices: list[int]):
        """run the routine to generate the data for the given indices"""
        pass

    @staticmethod
    @abstractmethod
    def generate_template_config() -> str:
        pass
