import simpose as sp
import signal
import itertools as it
from abc import ABC, abstractmethod
import multiprocessing as mp
import multiprocessing.queues as mpq
import numpy as np
from pydantic import validate_call
import functools
import time
from tqdm import tqdm
import queue
from typing import Any


class GeneratorParams(ABC, sp.base_config.BaseConfig):
    n_workers: int
    n_parallel_on_gpu: int
    gpus: None | list[int] = None
    worker_shards: int = 100

    @classmethod
    def get_description(cls) -> dict[str, str]:
        description = super().get_description()
        description.update(
            {
                "n_workers": "Number of worker processes",
                "n_parallel_on_gpu": "Number of parallel processes per GPU",
                "gpus": "List of GPUs to use",
                "worker_shards": "Number of shards to split the work into",
            }
        )
        return description


class Generator(ABC):
    def __init__(self, config: dict):
        self.config = config

        writer_config = config["Writer"]
        writer_name = writer_config["type"]

        generator_params_model: type[GeneratorParams] = getattr(
            sp.generators, self.config["Generator"]["type"] + "Config"
        )
        assert issubclass(generator_params_model, GeneratorParams)
        self.generator_params: GeneratorParams = generator_params_model.model_validate(
            self.config["Generator"]["params"]
        )

        self.randomizer_configs = self.config["Randomizers"]

        self.Writer: type[sp.writers.Writer] = getattr(sp.writers, writer_name)
        self.writer_params = lambda: sp.writers.WriterConfig.model_validate(
            writer_config["params"]
        )

    def start(self) -> None:
        mp.set_start_method("spawn")
        writer = self.Writer(self.writer_params(), comm=mp.Queue())
        writer.dump_config(self.config)

        all_pending_indices = writer.get_pending_indices()
        n_datapoints = len(all_pending_indices)
        if n_datapoints == 0:
            sp.logger.info("No images to render.")
            return

        sp.logger.info(f"Rendering {n_datapoints} images.")

        n_workers = min(
            self.generator_params.n_workers, n_datapoints // self.generator_params.worker_shards
        )

        # # split work into chunks of ~100 images
        if n_datapoints > self.generator_params.worker_shards:
            pending_indices = np.array_split(
                all_pending_indices, n_datapoints // self.generator_params.worker_shards
            )
        else:
            pending_indices = [np.array(all_pending_indices)]

        self._launch_parallel(self.generator_params, pending_indices, n_workers, n_datapoints)

        writer.post_process()

    def _launch_parallel(
        self,
        params: GeneratorParams,
        pending_indices: list[np.ndarray],
        n_workers: int,
        n_datapoints: int,
    ) -> None:
        active_gpus = it.cycle(params.gpus or [0])
        num_rendered_accumulator: mp.Queue[int] = mp.Queue()
        current_jobs: dict[str, np.ndarray] = {}
        job_queue: queue.Queue[
            tuple[
                np.ndarray,
                sp.writers.WriterConfig,
                type[sp.writers.Writer],
                GeneratorParams,  # generator config
                Any,  # randomizer configs
                mpq.Queue,
            ]
        ] = queue.Queue()

        semaphores = {
            gpu: sp.RemoteSemaphore(timeout=30.0, comm=mp.Queue(params.n_parallel_on_gpu))
            for gpu in params.gpus or [0]
        }

        def schedule_job(index_list: np.ndarray) -> None:
            active_gpu = next(active_gpus)

            # configure for this job
            writer_config = self.writer_params()
            writer_config.start_index = min(index_list)
            writer_config.end_index = max(index_list)
            # writer_config._gpu_semaphore_comm = gpu_semaphore._comm
            writer_config._active_gpu = active_gpu
            writer_config._q_rendered = num_rendered_accumulator

            job_queue.put(
                (
                    index_list,
                    writer_config,
                    self.Writer,
                    self.generator_params,
                    self.randomizer_configs,
                    semaphores[active_gpu]._comm,
                )
            )

        def launch_worker() -> bool:
            try:
                new_job = job_queue.get(block=False)
            except queue.Empty:
                return False
            proc = mp.Process(target=self.process, args=new_job)
            proc.start()
            current_jobs[proc.name] = new_job[0]
            return True

        # initialize job queue with all jobs
        for index_list in pending_indices:
            schedule_job(index_list)

        # start timeout semaphores
        for s in semaphores.values():
            s.start()

        # create and start workers, wait for started
        for _ in range(n_workers):
            launch_worker()
        time.sleep(1.0)

        finished = False
        bar = tqdm(total=n_datapoints, disable=False)

        try:
            while not finished:
                # print(f"Current jobs: {[x for x in current_jobs.keys()]}")
                # check for timeouts and reschedule job
                for remote_semaphore in semaphores.values():
                    terminated_workers = remote_semaphore.run()
                    for terminated_worker in terminated_workers:
                        # put job back into queue
                        schedule_job(current_jobs[terminated_worker])

                # check workers
                finished = True
                for proc_name in list(current_jobs.keys()):
                    if proc_name not in [p.name for p in mp.active_children()]:
                        # worker is done, start new one if job is available
                        current_jobs.pop(proc_name)

                        if not launch_worker():
                            # no job available -> did not launch new worker -> finished?
                            continue
                        finished = False  # started new worker -> not finished
                    else:
                        # some worker is still working -> not finished
                        finished = False

                # empty accumulator and update progress
                while True:
                    try:
                        bar.update(num_rendered_accumulator.get(block=False))
                    except queue.Empty:
                        break
                # time.sleep(1.0)

        except KeyboardInterrupt:
            # operate semaphores until all workers are done
            sp.logger.error("KeyboardInterrupt received. Please wait for workers to finish.")
            signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore from here on
            for p in mp.active_children():
                p.terminate()
            while any([p.is_alive() for p in mp.active_children()]):
                for remote_semaphore in semaphores.values():
                    remote_semaphore.run()

        sp.logger.info("All workers finished.")
        bar.close()

    @classmethod
    def process(
        cls,
        indices: np.ndarray,
        writer_config: sp.writers.WriterConfig,
        Writer: type[sp.writers.Writer],
        generator_config: GeneratorParams,
        randomizer_configs: dict[str, dict[str, str | dict]],
        comm: mp.Queue,
    ) -> int:
        import silence_tensorflow.auto
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(writer_config._active_gpu)

        np.random.seed(mp.current_process().pid)
        for handler in sp.logger.handlers:
            handler.setLevel(sp.logger.level)

        randomizers: dict[str, sp.random.Randomizer] = {}
        for rand_name, rand_config in randomizer_configs.items():
            randomizers[rand_name] = Generator.get_randomizer(rand_config)

        with Writer(writer_config, comm=comm) as writer:
            cls.generate_data(
                config=generator_config,
                writer=writer,
                randomizers=randomizers,
                indices=indices,
            )
        return len(indices)

    @staticmethod
    @abstractmethod
    def generate_data(
        config: GeneratorParams,
        writer: sp.writers.Writer,
        randomizers: dict[str, sp.random.Randomizer],
        indices: np.ndarray,
    ) -> None:
        """run the routine to generate the data for the given indices"""
        pass

    @staticmethod
    @abstractmethod
    def generate_template_config() -> str:
        pass

    @staticmethod
    @validate_call
    def get_randomizer(randomizer_config: dict[str, str | dict]) -> sp.random.Randomizer:
        name = randomizer_config["type"]
        config = randomizer_config["params"]

        if name == "Join":
            assert not isinstance(config, str)
            rands = [Generator.get_randomizer(rand) for _, rand in config.items()]
            assert all(isinstance(x, sp.random.JoinableRandomizer) for x in rands)
            randomizer = functools.reduce(lambda x, y: x + y, rands)  # type: ignore

            return randomizer

        assert isinstance(config, dict)
        assert isinstance(name, str)

        class_randomizer: type[sp.random.Randomizer] = getattr(sp.random, name)
        class_config: type[sp.random.RandomizerConfig] = getattr(sp.random, name + "Config")
        params = class_config.model_validate(config)
        return class_randomizer(params)
