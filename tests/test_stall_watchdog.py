"""A worker that wedges *outside* the GPU semaphore must still be caught.

The GPU semaphore's own timeout only runs while a worker waits for or holds the
semaphore. Scene setup, model loading, geometry upload and writing all happen
outside it, so a worker hanging there used to keep its shard forever without the
parent noticing (observed: 16 workers idle for 7 h, nothing timed out).

Two watchdogs, exercised separately below, because they catch different wedges:
the heartbeat one only fires when the worker stops running python at all, and a
worker can also stay perfectly responsive while getting nowhere.
"""

import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

import simpose as sp
from simpose.generators.generator import Generator, GeneratorParams

# long enough that a worker's spawn + tensorflow import fits inside it, since the
# watchdogs start counting at launch
TIMEOUT = 20.0
NEVER = 10 * TIMEOUT


class _StallParams(GeneratorParams):
    pass


class _NoopWriter(sp.writers.Writer):
    def get_pending_indices(self) -> np.ndarray:
        return np.arange(self.start_index, self.end_index + 1)

    def _write_data(self, dataset_index, *, scene=None, render_product=None) -> None:  # type: ignore[no-untyped-def]
        pass

    def _cleanup(self, dataset_index: int) -> None:
        pass


class _WedgedGenerator(Generator):
    """holds the GIL in a C call and never gives it back -- what a hung driver
    call looks like from the outside: every thread of the process stops, the
    heartbeat included"""

    @staticmethod
    def generate_data(config, writer, randomizers, indices) -> None:  # type: ignore[no-untyped-def]
        re.match(r"(a+)+$", "a" * 64 + "b")  # catastrophic backtracking

    @staticmethod
    def generate_template_config() -> str:
        return ""


class _IdleGenerator(Generator):
    """responsive (so it keeps heartbeating) but never produces a datapoint"""

    @staticmethod
    def generate_data(config, writer, randomizers, indices) -> None:  # type: ignore[no-untyped-def]
        time.sleep(NEVER)

    @staticmethod
    def generate_template_config() -> str:
        return ""


class _SlowGenerator(Generator):
    """takes far longer per datapoint than the poll interval, but keeps going"""

    @staticmethod
    def generate_data(config, writer, randomizers, indices) -> None:  # type: ignore[no-untyped-def]
        for idx in indices:
            time.sleep(0.4 * TIMEOUT)
            writer.write_data(int(idx), render_product=None)

    @staticmethod
    def generate_template_config() -> str:
        return ""


def _make_generator(
    cls: type[Generator], tmp_path: Path, params: GeneratorParams, n: int
) -> Generator:
    """assemble what _launch_parallel needs, bypassing config-file resolution"""
    gen = cls.__new__(cls)
    gen.config = {}
    gen.Writer = _NoopWriter  # type: ignore[assignment]
    gen.generator_params = params
    gen.randomizer_configs = {}
    gen.writer_params = lambda: sp.writers.WriterConfig(  # type: ignore[assignment]
        output_dir=tmp_path, start_index=0, end_index=n - 1
    )
    return gen


def _run(
    cls: type[Generator], tmp_path: Path, *, stall: float, progress: float, n: int = 10
) -> None:
    mp.set_start_method("spawn", force=True)
    os.environ["SIMPOSE_DIAG_DIR"] = str(tmp_path)  # inherited by the spawned workers

    params = _StallParams(
        n_workers=1,
        n_parallel_on_gpu=1,
        gpus=[0],
        worker_shards=n,
        worker_stall_timeout=stall,
        worker_progress_timeout=progress,
    )
    gen = _make_generator(cls, tmp_path, params, n)
    gen._launch_parallel(params, np.arange(n), n_workers=1, n_datapoints=n)


def _dumps(tmp_path: Path) -> list[Path]:
    """dumps that actually contain a traceback: every process creates its (empty)
    dump file at import, and only a killed one leaves something in it"""
    return sorted(
        p for p in (tmp_path / "simpose_stacks").glob("*.txt") if p.stat().st_size > 0
    )


def test_wedged_worker_is_dumped_killed_and_its_shard_retried(tmp_path: Path) -> None:
    # the shard can never finish, so it is retried MAX_SHARD_ATTEMPTS times and
    # then the run aborts -- rather than hanging forever, as it did before
    with pytest.raises(RuntimeError, match="failed 3 times"):
        _run(_WedgedGenerator, tmp_path, stall=TIMEOUT, progress=0.0)

    assert len(_dumps(tmp_path)) == 3, "expected one stack dump per killed worker"
    # a stack dump taken while the GIL is held is the whole point of using
    # faulthandler here, and it has to name the frame the worker died in
    assert any("generate_data" in d.read_text() for d in _dumps(tmp_path))


def test_idle_worker_is_caught_even_though_it_still_responds(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="failed 3 times"):
        _run(_IdleGenerator, tmp_path, stall=NEVER, progress=TIMEOUT)

    assert len(_dumps(tmp_path)) == 3


def test_slow_worker_is_left_alone(tmp_path: Path) -> None:
    """the timeouts are stall detectors, not a wall clock on the work: a worker
    that needs several timeout windows for its shard must survive"""
    _run(_SlowGenerator, tmp_path, stall=TIMEOUT, progress=TIMEOUT, n=6)

    assert not _dumps(tmp_path), "a healthy worker was killed"
