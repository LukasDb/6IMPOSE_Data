from re import T
import simpose as sp
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np
from pydantic import validate_call
import functools
import time
from tqdm import tqdm
from simpose import base_config


class GeneratorParams(ABC, base_config.BaseConfig):
    n_workers: int
    n_parallel_on_gpu: int
    gpus: None | list[int] = None
    worker_shards: int = 100

    @classmethod
    def get_description(cls):
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

    def start(self):
        writer = self._build_writer(self.config["Writer"])
        writer.dump_config(self.config)

        params = self._get_generator_config(self.config)

        pending_indices = writer.get_pending_indices()
        n_datapoints = len(pending_indices)
        if n_datapoints == 0:
            sp.logger.info("No images to render.")
            return

        sp.logger.info(f"Rendering {n_datapoints} images.")

        n_workers = min(params.n_workers, n_datapoints)

        # # split work into chunks of ~100 images
        if n_datapoints > params.worker_shards:
            pending_indices = np.array_split(pending_indices, n_datapoints // params.worker_shards)
        else:
            pending_indices = [pending_indices]

        n_gpus = len(params.gpus or [0])
        active_gpus = params.gpus or [0]

        mp_context = mp.get_context("spawn")  # consistent in linux and macos

        # manager for use with pool!
        with mp_context.Manager() as manager, mp_context.Pool(
            n_workers, maxtasksperchild=1
        ) as pool, tqdm(total=n_datapoints) as bar:
            semaphores = {
                gpu: manager.BoundedSemaphore(params.n_parallel_on_gpu) for gpu in active_gpus
            }
            q_rendered = manager.Queue()

            def jobs():
                for job_index, ind_list in enumerate(pending_indices):
                    active_gpu = active_gpus[job_index % n_gpus]
                    gpu_semaphore = semaphores[active_gpu]
                    device_setup = {
                        "gpu_semaphore": gpu_semaphore,
                        "active_gpu": active_gpu,
                        "q_rendered": q_rendered,
                    }
                    yield ind_list, self.config, device_setup

            result = pool.starmap_async(self.init_and_launch, jobs(), chunksize=1)
            while not result.ready():
                while not q_rendered.empty():
                    bar.update(q_rendered.get())
                time.sleep(1.0)

        writer.post_process()

    @classmethod
    def init_and_launch(cls, indices, config, device_setup):
        import signal
        import silence_tensorflow.auto
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_setup["active_gpu"])
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        np.random.seed(mp.current_process().pid)
        for handler in sp.logger.handlers:
            handler.setLevel(sp.logger.level)
        return cls.process(indices, config, device_setup)

    @classmethod
    def process(cls, indices, config, device_setup):
        randomizers: dict[str, sp.random.Randomizer] = {}
        for rand_name, rand_initializer in config["Randomizers"].items():
            randomizers[rand_name] = Generator.get_randomizer(rand_initializer)

        gen_config = cls._get_generator_config(config)

        writer_config = config["Writer"]
        writer_name = writer_config["type"]
        writer_config = sp.writers.WriterConfig.model_validate(writer_config["params"])
        writer_config.start_index = min(indices)
        writer_config.end_index = max(indices)
        Writer: type[sp.writers.Writer] = getattr(sp.writers, writer_name)

        with Writer(writer_config, device_setup) as writer:
            cls.generate_data(
                config=gen_config,
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
        indices: list[int],
    ):
        """run the routine to generate the data for the given indices"""
        pass

    @staticmethod
    @abstractmethod
    def generate_template_config() -> str:
        pass

    @staticmethod
    def _build_writer(writer_config):
        writer_name = writer_config["type"]
        writer_config = sp.writers.WriterConfig.model_validate(writer_config["params"])
        writer: sp.writers.Writer = getattr(sp.writers, writer_name)(writer_config, {})
        return writer

    @staticmethod
    def _get_generator_config(config) -> GeneratorParams:
        generator_params_model: type[GeneratorParams] = getattr(
            sp.generators, config["Generator"]["type"] + "Config"
        )
        return generator_params_model.model_validate(config["Generator"]["params"])

    @staticmethod
    @validate_call
    def get_randomizer(initializer: dict[str, str | dict]) -> sp.random.Randomizer:
        name = initializer["type"]
        config = initializer["params"]

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
        cnf = class_config.model_validate(config)
        return class_randomizer(cnf)
