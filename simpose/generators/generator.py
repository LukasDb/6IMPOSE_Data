import time
import signal
import simpose as sp
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np
import subprocess
import os


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

        n_workers = min(self.params.n_workers, len(pending_indices))

        if n_workers == 1:
            sp.logger.debug("Using single process.")
            self.generate_data(pending_indices)
            return
        queue = mp.Queue()
        gpu_semaphore = mp.Semaphore(4)

        workers = [
            mp.Process(target=self.process, args=(queue, gpu_semaphore)) for _ in range(n_workers)
        ]
        for w in workers:
            w.start()

        indices = np.array_split(pending_indices, n_workers)
        for ind_list in indices:
            queue.put(ind_list)
        for _ in range(n_workers):
            queue.put(None)
        for w in workers:
            w.join()

        self.writer.post_process()

    def start_mpi(self, direct_start: bool, main_kwargs: dict):
        pending_indices = self.writer.get_pending_indices()
        if len(pending_indices) == 0:
            sp.logger.info("No images to render.")
            return

        n_workers = min(self.params.n_workers, len(pending_indices))

        if not direct_start:
            #     """launch mode"""
            if n_workers == 1:
                sp.logger.debug("Using single process.")
                self.generate_data(pending_indices)
                return

            sp.logger.info("Rendering %d images with %d workers.", len(pending_indices), n_workers)
            proc = subprocess.Popen(
                [
                    "mpiexec",
                    "-n",
                    f"{n_workers}",
                    "simpose",
                    "generate",
                    "--direct_launch",
                    main_kwargs["config_file"],
                ],
                env=os.environ,
            )
            proc.wait()

            self.post_process()
            return

        indices = np.array_split(pending_indices, n_workers)
        job_indices = indices[MPI.COMM_WORLD.Get_rank()].tolist()

        print(f"GOT RANK: {MPI.COMM_WORLD.Get_rank()} and start {self.writer.start_index}")
        time.sleep(MPI.COMM_WORLD.Get_rank() * 2)
        self.generate_data(job_indices)

        # processes: list[subprocess.Popen] = []
        # for ind_list in indices:
        #     start_index = ind_list[0]
        #     end_index = ind_list[-1]

        #     config_file = str(main_kwargs["config_file"])
        #     cmd = [
        #         "simpose",
        #         "generate",
        #     ]
        #     if main_kwargs["verbose"] > 0:
        #         cmd.append("-" + "v" * main_kwargs["verbose"])

        #     cmd.extend(
        #         [
        #             "--direct_launch",
        #             "--start_index",
        #             f"{start_index}",
        #             "--end_index",
        #             f"{end_index}",
        #             config_file,
        #         ]
        #     )
        #     print(cmd)

        #     proc = subprocess.Popen(cmd)
        #     processes.append(proc)
        # # ignore ctrl+c in parent process
        # for proc in processes:
        #     proc.wait()

        # return
        # print(self.writer.get_pending_indices())

        indices = np.array_split(pending_indices, n_workers)

        job_indices = indices[MPI.COMM_WORLD.Get_rank()].tolist()

        self.writer.start_index = job_indices[0]
        self.writer.end_index = job_indices[-1]

        print(f"GOT RANK: {MPI.COMM_WORLD.Get_rank()} and start {self.writer.start_index}")
        time.sleep(MPI.COMM_WORLD.Get_rank() * 2)
        self.generate_data(job_indices)

    def process(self, queue: mp.Queue, gpu_semaphore=None):
        np.random.seed(mp.current_process().pid)
        import importlib

        importlib.reload(sp)

        while True:
            indices = queue.get()
            self.writer.start_index = indices[0]
            self.writer.end_index = indices[-1]
            if indices is None:
                break
            self.generate_data(indices, gpu_semaphore=gpu_semaphore)

    def post_process(self):
        self.writer.post_process()

    @abstractmethod
    def generate_data(self, indices: list[int], gpu_semaphore=None):
        """run the routine to generate the data for the given indices"""
        pass

    @staticmethod
    @abstractmethod
    def generate_template_config() -> str:
        pass
