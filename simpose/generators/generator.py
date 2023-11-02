import simpose as sp
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np
from pydantic import validate_call
import functools
import time
from simpose import base_config


class GeneratorParams(ABC, base_config.BaseConfig):
    n_workers: int
    n_parallel_on_gpu: int


class Generator(ABC):
    def __init__(self, config: dict):
        self.config = config

    def start(self):
        writer = self._build_writer(self.config["Writer"])
        params = self._get_generator_config(self.config)

        pending_indices = writer.get_pending_indices()
        if len(pending_indices) == 0:
            sp.logger.info("No images to render.")
            return

        sp.logger.info(f"Rendering {len(pending_indices)} images.")

        n_workers = min(params.n_workers, len(pending_indices))

        # if n_workers == 1:
        #     sp.logger.debug("Using single process.")
        #     randomizers: dict[str, sp.random.Randomizer] = {}
        #     for rand_name, rand_initializer in self.config["Randomizers"].items():
        #         randomizers[rand_name] = Generator.get_randomizer(rand_initializer)
        #     self.generate_data(params, writer, randomizers, pending_indices)
        #     writer.post_process()
        #     return

        # # split work into chunks of ~100 images
        if len(pending_indices) > 100:
            pending_indices = np.array_split(pending_indices, len(pending_indices) // 10)

        def init_worker():
            import signal

            signal.signal(signal.SIGINT, signal.SIG_IGN)

        with mp.Manager() as manager, mp.Pool(n_workers, init_worker, maxtasksperchild=1) as pool:
            gpu_semaphore = manager.Semaphore(params.n_parallel_on_gpu)

            def jobs():
                for ind_list in pending_indices:
                    yield ind_list, self.config, gpu_semaphore

            for _ in pool.imap_unordered(self.process, jobs(), chunksize=1):
                pass

            pool.join()

        writer.post_process()

    @classmethod
    def process(cls, args):
        indices, config, gpu_semaphore = args
        print(f"fired up ({mp.current_process().name})")

        np.random.seed(mp.current_process().pid)
        import importlib

        importlib.reload(sp)

        randomizers: dict[str, sp.random.Randomizer] = {}
        for rand_name, rand_initializer in config["Randomizers"].items():
            randomizers[rand_name] = Generator.get_randomizer(rand_initializer)

        gen_config = cls._get_generator_config(config)

        print(f"Got index list and starting now ({mp.current_process().name})")

        writer_config = config["Writer"]
        writer_name = writer_config["type"]
        writer_config = sp.writers.WriterConfig.model_validate(writer_config["params"])
        writer_config.start_index = min(indices)
        writer_config.end_index = max(indices)
        writer: sp.writers.Writer = getattr(sp.writers, writer_name)(writer_config)

        cls.generate_data(
            config=gen_config,
            writer=writer,
            randomizers=randomizers,
            indices=indices,
            gpu_semaphore=gpu_semaphore,
        )

    # old version with queue
    # @classmethod
    # def process(cls, config, queue: mp.Queue, gpu_semaphore=None):
    #     print(f"fired up ({mp.current_process().name})")

    #     np.random.seed(mp.current_process().pid)
    #     import importlib

    #     importlib.reload(sp)

    #     randomizers: dict[str, sp.random.Randomizer] = {}
    #     for rand_name, rand_initializer in config["Randomizers"].items():
    #         randomizers[rand_name] = Generator.get_randomizer(rand_initializer)

    #     gen_config = cls._get_generator_config(config)

    #     while True:
    #         indices = queue.get()
    #         if indices is None:
    #             break

    #         print(f"Got index list and starting now ({mp.current_process().name})")

    #         writer_config = config["Writer"]
    #         writer_name = writer_config["type"]
    #         writer_config = sp.writers.WriterConfig.model_validate(writer_config["params"])
    #         writer_config.start_index = min(indices)
    #         writer_config.end_index = max(indices)
    #         writer: sp.writers.Writer = getattr(sp.writers, writer_name)(writer_config)

    #         cls.generate_data(
    #             config=gen_config,
    #             writer=writer,
    #             randomizers=randomizers,
    #             indices=indices,
    #             gpu_semaphore=gpu_semaphore,
    #         )

    @staticmethod
    @abstractmethod
    def generate_data(
        config: GeneratorParams,
        writer: sp.writers.Writer,
        randomizers: dict[str, sp.random.Randomizer],
        indices: list[int],
        gpu_semaphore=None,
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
        writer: sp.writers.Writer = getattr(sp.writers, writer_name)(writer_config)
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
