import simpose as sp
from abc import ABC, abstractmethod
import multiprocessing as mp
import numpy as np
from pydantic import validate_call
import functools
import time
from tqdm import tqdm
import queue


class GeneratorParams(ABC, sp.base_config.BaseConfig):
    n_workers: int
    n_parallel_on_gpu: int
    gpus: None | list[int] = None
    worker_shards: int = 100
    worker_stall_timeout: float = 1200.0
    worker_progress_timeout: float = 7200.0

    @classmethod
    def get_description(cls) -> dict[str, str]:
        description = super().get_description()
        description.update(
            {
                "n_workers": "Number of worker processes",
                "n_parallel_on_gpu": "Number of parallel processes per GPU",
                "gpus": "List of GPUs to use",
                "worker_shards": "Number of shards to split the work into",
                "worker_stall_timeout": (
                    "Seconds without a worker heartbeat before it is dumped, killed and"
                    " its shard rescheduled (0 disables)"
                ),
                "worker_progress_timeout": (
                    "Seconds without a written datapoint before a worker is dumped,"
                    " killed and its shard rescheduled (0 disables)"
                ),
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
        mp.set_start_method("spawn", force=True)

        # fail fast: construct the randomizers once in the parent, so bad paths /
        # empty model dirs abort with a clear error instead of crash-looping workers
        for rand_name, rand_config in self.randomizer_configs.items():
            try:
                Generator.get_randomizer(rand_config)
            except Exception as e:
                raise RuntimeError(f"Invalid config for randomizer '{rand_name}': {e}") from e

        writer = self.Writer(self.writer_params(), comm=mp.Queue())
        # validate the writer state (e.g. overwrite refusal) BEFORE dump_config
        # overwrites the existing dataset's config.yaml
        pending_indices = writer.get_pending_indices()
        writer.dump_config(self.config)

        n_datapoints = len(pending_indices)
        if n_datapoints == 0:
            sp.logger.info("No images to render.")
            return

        sp.logger.info(f"Rendering {n_datapoints} images.")

        n_workers = min(
            self.generator_params.n_workers, n_datapoints // self.generator_params.worker_shards
        )
        n_workers = max(1, n_workers)

        self._launch_parallel(self.generator_params, pending_indices, n_workers, n_datapoints)

        writer.post_process()

    def _launch_parallel(
        self,
        params: GeneratorParams,
        pending_indices: np.ndarray,
        n_workers: int,
        n_datapoints: int,
    ) -> None:
        MAX_SHARD_ATTEMPTS = 3

        num_rendered_accumulator: mp.Queue[tuple[str, int]] = mp.Queue()
        success_accumulator: mp.Queue[int] = mp.Queue()
        heartbeat_accumulator: mp.Queue[str] = mp.Queue()
        # Two independent liveness signals per worker, because the GPU semaphore's
        # own timeout only covers a worker while it waits for or holds the
        # semaphore: scene setup, model loading, geometry upload and writing all
        # happen outside it, and a worker wedged there was invisible to this loop
        # forever (observed: 16 workers idle for 7 h, nothing timed out).
        #   heartbeat: is it still running python at all? goes silent when a C
        #              extension deadlocks with the GIL held, which is what a hung
        #              driver call looks like from here.
        #   progress:  is it still turning out datapoints? catches the worker that
        #              stays responsive but no longer gets anywhere.
        last_heartbeat: dict[mp.Process, float] = {}
        last_progress: dict[mp.Process, float] = {}
        stall_timeout = params.worker_stall_timeout
        progress_timeout = params.worker_progress_timeout
        job_queue: queue.Queue[tuple[np.ndarray, sp.writers.WriterConfig]] = queue.Queue()
        current_jobs: dict[mp.Process, np.ndarray] = {}
        current_comms: dict[mp.Process, tuple[sp.MainSemaphore, sp.Comm]] = {}
        current_gpu: dict[mp.Process, int] = {}
        failed_attempts: dict[tuple[int, int], int] = {}
        gpu_list = params.gpus or [0]
        # live (launched, not-yet-reaped) worker count per GPU: used to pick the
        # least-loaded GPU at *launch* time instead of baking a GPU into a job at
        # *schedule* time. Schedule-time round-robin bakes an assignment over the
        # whole backlog up front; since jobs are dequeued FIFO as workers die (in
        # scene-render-time order, not GPU order), the job that back-fills a GPU
        # that just freed up is essentially a random pick from that round-robin
        # sequence - it can easily land on an already-busy GPU while the one that
        # just freed gets nothing new queued for it, and sits idle for the rest
        # of the run once its own backlog of pre-baked jobs runs out.
        gpu_worker_count = {gpu: 0 for gpu in gpu_list}
        semaphores = {
            gpu: sp.MainSemaphore(value=params.n_parallel_on_gpu, timeout=600.0)
            for gpu in gpu_list
        }

        def schedule_job(index_list: np.ndarray) -> None:
            writer_config = self.writer_params()
            writer_config.start_index = min(index_list)
            writer_config.end_index = max(index_list)
            writer_config._q_rendered = num_rendered_accumulator

            job_queue.put((index_list, writer_config))

        def launch_worker() -> bool:
            try:
                index_list, writer_config = job_queue.get(block=False)
            except queue.Empty:
                return False
            # assign the GPU now, to whichever one currently has the fewest live
            # workers - naturally routes new work to a GPU that just freed up
            active_gpu = min(gpu_list, key=lambda g: gpu_worker_count[g])
            writer_config._active_gpu = active_gpu
            gpu_worker_count[active_gpu] += 1

            # create the semaphore comm only when the worker actually launches:
            # one queue (2 fds) per live worker instead of per scheduled shard
            semaphore = semaphores[active_gpu]
            comm = semaphore.get_comm()
            proc = mp.Process(
                target=self.process,
                args=(
                    index_list,
                    writer_config,
                    self.Writer,
                    self.generator_params,
                    self.randomizer_configs,
                    comm,
                    success_accumulator,
                    heartbeat_accumulator,
                ),
                daemon=True,
            )
            proc.start()
            last_heartbeat[proc] = last_progress[proc] = time.time()
            current_jobs[proc] = index_list
            current_comms[proc] = (semaphore, comm)
            current_gpu[proc] = active_gpu
            sp.logger.info(f"Launched new worker: {proc.name} on GPU {active_gpu}")
            return True

        # initialize job queue with all jobs; split at index gaps first, so every
        # shard is a contiguous range (resume derives a shard's expected record
        # count from the {start}_{end} filename). fixed-size chunks, never longer
        # than worker_shards: a shard must not span more than one scene, because
        # object ids are only consistent within a scene
        ws = self.generator_params.worker_shards
        contiguous_runs = np.split(
            pending_indices, np.where(np.diff(pending_indices) != 1)[0] + 1
        )
        for run in contiguous_runs:
            for chunk_start in range(0, len(run), ws):
                index_list = run[chunk_start : chunk_start + ws]
                if len(index_list) > 0:
                    schedule_job(index_list)

        # create and start workers, wait for started
        for _ in range(n_workers):
            launch_worker()
        time.sleep(1.0)

        finished = False
        successful_workers = []
        # average rate, no smoothing
        bar = tqdm(total=n_datapoints, disable=False, smoothing=0, dynamic_ncols=True)
        while not finished:
            # operate semaphores, reschedule job on timeout
            time.sleep(0.05)
            for s in semaphores.values():
                s.run()

            while True:
                try:
                    successful_workers.append(success_accumulator.get(block=False))
                except queue.Empty:
                    break

            # note who is still alive in python
            while True:
                try:
                    beating = heartbeat_accumulator.get(block=False)
                except queue.Empty:
                    break
                for proc in current_jobs:
                    if proc.name == beating:
                        last_heartbeat[proc] = time.time()

            # check alive workers
            for proc in list([x for x in current_jobs.keys() if x.is_alive()]):
                if proc.pid in successful_workers:
                    # worker was successful but did not exit
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    continue

                # a wedged worker is killed, which routes it into the dead-worker
                # branch below: semaphore released, shard rescheduled. Its stacks are
                # dumped first, since killing it destroys the only evidence of where
                # it hung (and it may well be unattachable by py-spy/gdb under the
                # default kernel.yama.ptrace_scope=1)
                now = time.time()
                silent_for = now - last_heartbeat.get(proc, now)
                idle_for = now - last_progress.get(proc, now)
                if stall_timeout > 0 and silent_for >= stall_timeout:
                    reason = f"stopped running python {silent_for:.0f}s ago"
                elif progress_timeout > 0 and idle_for >= progress_timeout:
                    reason = f"has not written a datapoint in {idle_for:.0f}s"
                else:
                    continue

                dump = sp.diagnostics.request_stack_dump(proc.pid)
                sp.logger.error(
                    f"{proc.name} {reason}. Stacks written to {dump}; "
                    "killing it and restarting its shard."
                )
                try:
                    proc.kill()
                except Exception:
                    pass

            time.sleep(0.05)

            # check dead workers
            dead_procs = list([x for x in current_jobs.keys() if not x.is_alive()])
            if len(dead_procs) > 0:
                # a worker's success message can arrive after its process death is
                # visible; give the queue a grace period before treating it as crashed
                time.sleep(0.5)
                while True:
                    try:
                        successful_workers.append(success_accumulator.get(block=False))
                    except queue.Empty:
                        break

            for proc in dead_procs:
                if proc.pid not in successful_workers:
                    sp.logger.error(f"{proc.name} died. Releasing its semaphore and rescheduling.")
                    try:
                        proc.kill()
                    except Exception:
                        pass

                    # release the semaphore this worker actually held
                    for sem in semaphores.values():
                        if sem.is_acquired(proc.name):
                            sem.release_for(proc.name)

                    # reschedule job (since worker did not release -> likely crashed),
                    # but give up after repeated failures instead of looping forever
                    index_list = current_jobs[proc]
                    shard_key = (int(min(index_list)), int(max(index_list)))
                    failed_attempts[shard_key] = failed_attempts.get(shard_key, 0) + 1
                    if failed_attempts[shard_key] >= MAX_SHARD_ATTEMPTS:
                        for p_ in current_jobs:
                            try:
                                p_.kill()
                            except Exception:
                                pass
                        bar.close()
                        raise RuntimeError(
                            f"Shard {shard_key[0]}..{shard_key[1]} failed "
                            f"{MAX_SHARD_ATTEMPTS} times; aborting the run. Check the "
                            "worker error output above for the underlying failure."
                        )
                    schedule_job(index_list)
                else:
                    successful_workers.remove(proc.pid)

                # drop the dead worker's semaphore comm so it is not polled anymore
                # and its fds are freed
                semaphore_comm = current_comms.pop(proc, None)
                if semaphore_comm is not None:
                    semaphore_comm[0].remove_comm(semaphore_comm[1])

                # free up its GPU slot before picking the next worker's GPU, so
                # the replacement is routed back to the GPU that just freed up
                gpu = current_gpu.pop(proc, None)
                if gpu is not None:
                    gpu_worker_count[gpu] -= 1

                # remove from current jobs
                current_jobs.pop(proc)
                last_heartbeat.pop(proc, None)
                last_progress.pop(proc, None)

                # in any case try to launch a new worker for each dead worker
                launch_worker()

            # finished when:
            # 1) all semaphores released
            # 2) no more scheduled jobs in queue
            # 3) all workers are done

            # 1) check if all semaphores are released ==  # all are not acquired

            finished = (
                all([not s.is_acquired() for s in semaphores.values()])
                and job_queue.empty()
                and all(not n.is_alive() for n in current_jobs.keys())
            )

            # empty accumulator and update progress
            while True:
                try:
                    rendered_by, n_rendered = num_rendered_accumulator.get(block=False)
                except queue.Empty:
                    break
                bar.update(n_rendered)
                for proc in current_jobs:
                    if proc.name == rendered_by:
                        last_progress[proc] = time.time()
            bar.set_postfix_str(
                f"| queued: {job_queue.qsize()} jobs | current: {[x.name for x in current_jobs.keys()]}"
            )
        bar.close()

        sp.logger.info("All workers finished.")

    @classmethod
    def process(
        cls,
        indices: np.ndarray,
        writer_config: sp.writers.WriterConfig,
        Writer: type[sp.writers.Writer],
        generator_config: GeneratorParams,
        randomizer_configs: dict[str, dict[str, str | dict]],
        comm: mp.Queue,
        success_accumulator: mp.Queue,
        heartbeat: "mp.Queue | None" = None,
    ) -> None:
        import silence_tensorflow.auto
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(writer_config._active_gpu)
        # CUDA_VISIBLE_DEVICES does not constrain EGL device enumeration, so the
        # EGL rasterizer would otherwise put every worker's label passes on
        # physical GPU 0 regardless of this worker's assigned GPU
        os.environ["SIMPOSE_EGL_DEVICE"] = str(writer_config._active_gpu)

        # started before anything heavy, so a hang in scene setup / model loading
        # is covered too, not just the rendering. A thread rather than a per-frame
        # tick because whole scenes pass between two frames being written; and it
        # goes quiet exactly when it should -- a C extension deadlocking with the
        # GIL held stops this thread as dead as the main one
        if heartbeat is not None:
            cls._start_heartbeat(
                heartbeat, cls.heartbeat_interval(generator_config.worker_stall_timeout)
            )

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
            sp.logger.error(f"generate data finished {mp.current_process().name}")
        sp.logger.error(f"exited writer, now exiting proc {mp.current_process().name}")
        success_accumulator.put(mp.current_process().pid)
        exit(0)

    MAX_HEARTBEAT_INTERVAL = 15.0

    @classmethod
    def heartbeat_interval(cls, stall_timeout: float) -> float:
        """several beats per timeout window, so no single late beat can trip the
        watchdog on a healthy worker"""
        return max(1.0, min(cls.MAX_HEARTBEAT_INTERVAL, stall_timeout / 4.0))

    @classmethod
    def _start_heartbeat(cls, heartbeat: mp.Queue, interval: float) -> None:
        """tell the parent this worker is still executing python, until it isn't"""
        import threading

        name = mp.current_process().name

        def beat() -> None:
            while True:
                try:
                    heartbeat.put(name)
                except Exception:
                    return  # parent gone / queue closed -- the run is over anyway
                time.sleep(interval)

        threading.Thread(target=beat, daemon=True, name="sp-heartbeat").start()

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
