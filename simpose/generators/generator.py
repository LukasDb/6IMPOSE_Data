import signal
import simpose as sp
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np
from pydantic import BaseModel
import sys


class GeneratorParams(BaseModel, extra="forbid"):
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
        indices = np.array_split(pending_indices, n_workers)

        queue = mp.Queue()
        processes = [
            mp.Process(target=self.process, args=(queue,), daemon=True) for _ in range(n_workers)
        ]
        for p in processes:
            p.start()

        # load jobs
        for i in range(n_workers):
            queue.put(indices[i])
        # send stop signal
        for _ in range(n_workers):
            queue.put(None)

        # intercept SIGINT
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for p in processes:
            p.join()

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
