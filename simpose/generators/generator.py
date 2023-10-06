import signal
import simpose as sp
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np
import subprocess


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

    def start(self, direct_start: bool, main_kwargs: dict):
        pending_indices = self.writer.get_pending_indices()
        if len(pending_indices) == 0:
            sp.logger.info("No images to render.")
            return

        if not direct_start:
            """launch mode"""
            n_workers = min(self.params.n_workers, len(pending_indices))
            if n_workers == 1:
                sp.logger.debug("Using single process.")
                self.generate_data(pending_indices)
                return

            sp.logger.info("Rendering %d images with %d workers.", len(pending_indices), n_workers)

            indices = np.array_split(pending_indices, n_workers)
            # TODO for now do start end end indices

            processes: list[subprocess.Popen] = []
            for ind_list in indices:
                start_index = ind_list[0]
                end_index = ind_list[-1]

                config_file = str(main_kwargs["config_file"])
                cmd = [
                    "simpose",
                    "generate",
                ]
                if main_kwargs["verbose"] > 0:
                    cmd.append("-" + "v" * main_kwargs["verbose"])

                cmd.extend(
                    [
                        "--direct_launch",
                        "--start_index",
                        f"{start_index}",
                        "--end_index",
                        f"{end_index}",
                        config_file,
                    ]
                )
                print(cmd)

                proc = subprocess.Popen(cmd)
                processes.append(proc)

            for proc in processes:
                proc.wait()

            return

        # print(self.writer.get_pending_indices())
        self.generate_data(pending_indices)
        return

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
