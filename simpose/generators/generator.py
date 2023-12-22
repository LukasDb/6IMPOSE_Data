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

        with mp.Manager() as manager:
            n_gpus = len(params.gpus or [0])
            active_gpus = params.gpus or [0]
            semaphores = {
                gpu: manager.BoundedSemaphore(params.n_parallel_on_gpu) for gpu in active_gpus
            }
            rendered = manager.dict()

            def jobs():
                for job_index, ind_list in enumerate(pending_indices):
                    active_gpu = active_gpus[job_index % n_gpus]
                    gpu_semaphore = semaphores[active_gpu]
                    yield ind_list, self.config, gpu_semaphore, active_gpu, rendered

            with mp.Pool(n_workers, maxtasksperchild=1) as pool, tqdm(total=n_datapoints) as bar:
                # for n_rendered in pool.imap_unordered(self.process, jobs(), chunksize=1):
                #     bar.update(n_rendered)
                # pool.map(self.process, jobs(), chunksize=1)
                result = pool.map_async(self.process, jobs(), chunksize=1)
                while result.ready() is False:
                    total_rendered = sum(rendered.values())
                    bar.update(total_rendered - bar.n)
                    time.sleep(1.0)

        writer.post_process()

    @classmethod
    def init_worker(cls, gpu: int):
        import signal
        import silence_tensorflow.auto
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        np.random.seed(mp.current_process().pid)
        import importlib

        importlib.reload(sp)

    @classmethod
    def mock(cls, args):
        indices, config, gpu_semaphore, gpu = args
        cls.init_worker(gpu)

        import time

        import tensorflow as tf

        import bpy

        pref = bpy.context.preferences.addons["cycles"].preferences
        pref.get_devices()  # type: ignore
        available_devices = list(pref.devices)
        with gpu_semaphore:
            time.sleep(2)
        output = f"Worker {mp.current_process().name} started:"
        # output += f"{len(indices)} images, {gpu_semaphore}"
        output += f", gpus: [{tf.config.list_physical_devices('GPU')}]]"
        # output += f", blender: {available_devices}"
        print(output)
        return len(indices)

    @classmethod
    def process(cls, args):
        indices, config, gpu_semaphore, gpu, rendered_dict = args
        cls.init_worker(gpu)

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
        # writer: sp.writers.Writer = getattr(sp.writers, writer_name)(writer_config)

        with Writer(writer_config, gpu_semaphore, rendered_dict) as writer:
            cls.generate_data(
                config=gen_config,
                writer=writer,
                randomizers=randomizers,
                indices=indices,
            )
        # TODO make writer update the counter, with each image
        # rendered_dict[mp.current_process().name] = len(indices)
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
