"""Resume round trip: a partially generated dataset plus a resumed run must yield
exactly one record per index, and corrupt shards must be detected and regenerated."""
import json
import os

import numpy as np
import pytest

import simpose as sp
from simpose.data.dataset import Dataset
from simpose.writers.shard_index import INVALID, count_records
from simpose.writers.tfrecord_writer import TFRecordWriter
from simpose.writers.writer import WriterConfig

from helpers import write_shard


def _pending(output_dir, start, end, overwrite=False):
    cfg = WriterConfig(
        output_dir=output_dir, overwrite=overwrite, start_index=start, end_index=end
    )
    return list(TFRecordWriter(cfg, None).get_pending_indices())


def _read_indices(root):
    ds = sp.data.TFRecordDataset(root, num_parallel_files=1, pre_shuffle=False)
    return [int(item[Dataset.CAM_LOCATION][0].numpy()) for item in ds]


def test_interrupted_run_resumes_and_completes(tmp_path):
    # "interrupted" run: only the first of two shards was written
    write_shard(tmp_path, 0, 2)

    pending = _pending(tmp_path, 0, 5)
    assert pending == [3, 4, 5]

    # the resumed run generates exactly the pending shard
    write_shard(tmp_path, 3, 5)
    assert _pending(tmp_path, 0, 5) == []

    # every index present exactly once
    assert sorted(_read_indices(tmp_path)) == [0, 1, 2, 3, 4, 5]


def test_corrupt_shard_is_detected_and_regenerated(tmp_path):
    write_shard(tmp_path, 0, 2)
    write_shard(tmp_path, 3, 5)

    shard = tmp_path / "data" / "000000_000002.tfrecord"
    payload = shard.read_bytes()
    shard.write_bytes(payload[: len(payload) // 2])  # truncate, like a SIGKILL

    assert _pending(tmp_path, 0, 5) == [0, 1, 2]
    assert not shard.exists(), "corrupt shard must be removed so it gets regenerated"

    write_shard(tmp_path, 0, 2)
    assert _pending(tmp_path, 0, 5) == []
    assert sorted(_read_indices(tmp_path)) == [0, 1, 2, 3, 4, 5]


def test_incomplete_shard_is_detected(tmp_path):
    # shard file claims 0..5 but only 3 records were written (worker died mid-shard)
    cfg = WriterConfig(output_dir=tmp_path, overwrite=True, start_index=0, end_index=5)
    from helpers import make_render_product

    with TFRecordWriter(cfg, None) as writer:
        for i in range(3):
            writer.write_data(i, render_product=make_render_product(2, seed=i, index=i))

    assert _pending(tmp_path, 0, 5) == [0, 1, 2, 3, 4, 5]


def test_overwrite_ignores_existing_data(tmp_path):
    write_shard(tmp_path, 0, 5)
    assert _pending(tmp_path, 0, 5, overwrite=True) == [0, 1, 2, 3, 4, 5]


def test_fresh_dataset_is_fully_pending(tmp_path):
    assert _pending(tmp_path / "new", 0, 9) == list(range(10))


def test_index_is_written_and_reused(tmp_path):
    """A second resume must answer from shard_index.json without re-inflating."""
    write_shard(tmp_path, 0, 2)
    assert _pending(tmp_path, 0, 5) == [3, 4, 5]

    index = tmp_path / "shard_index.json"
    assert index.exists(), "the scan must persist its counts"

    # corrupt the payload while preserving size+mtime: a real re-count would now
    # fail, so an unchanged answer proves the index served it
    shard = tmp_path / "data" / "000000_000002.tfrecord"
    stat = shard.stat()
    raw = bytearray(shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    shard.write_bytes(bytes(raw))
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert _pending(tmp_path, 0, 5) == [3, 4, 5]


def test_index_entry_is_invalidated_when_shard_changes(tmp_path):
    """A stale index must never mask a shard that has since been truncated."""
    write_shard(tmp_path, 0, 2)
    assert _pending(tmp_path, 0, 5) == [3, 4, 5]

    shard = tmp_path / "data" / "000000_000002.tfrecord"
    payload = shard.read_bytes()
    shard.write_bytes(payload[: len(payload) // 2])  # size changes -> fingerprint voids

    assert _pending(tmp_path, 0, 5) == [0, 1, 2, 3, 4, 5]
    assert not shard.exists()


def test_worker_sidecar_avoids_a_rescan(tmp_path):
    """The count a worker leaves behind is adopted instead of re-inflating."""
    write_shard(tmp_path, 0, 2)
    sidecar = tmp_path / "data" / "000000_000002.count"
    assert sidecar.exists(), "the writer must record its own record count"

    # corrupt the shard in place (size+mtime preserved) so that inflating it
    # would report garbage -- with only the sidecar to go on, a correct answer
    # proves the count came from there and the payload was never read
    shard = tmp_path / "data" / "000000_000002.tfrecord"
    stat = shard.stat()
    raw = bytearray(shard.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    shard.write_bytes(bytes(raw))
    os.utime(shard, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert not (tmp_path / "shard_index.json").exists(), "no index yet, only the sidecar"
    assert _pending(tmp_path, 0, 5) == [3, 4, 5]


def test_stale_sidecar_is_not_believed(tmp_path):
    """A sidecar from a longer previous shard must not vouch for a shorter one."""
    from helpers import make_render_product

    write_shard(tmp_path, 0, 5)
    sidecar = tmp_path / "data" / "000000_000005.count"
    assert json.loads(sidecar.read_text())["n_records"] == 6

    # same shard name, rewritten with fewer records (worker died mid-shard)
    cfg = WriterConfig(output_dir=tmp_path, overwrite=True, start_index=0, end_index=5)
    with TFRecordWriter(cfg, None) as writer:
        for i in range(2):
            writer.write_data(i, render_product=make_render_product(2, seed=i, index=i))

    assert _pending(tmp_path, 0, 5) == [0, 1, 2, 3, 4, 5]


def test_deleted_shard_leaves_no_sidecar(tmp_path):
    write_shard(tmp_path, 0, 2)
    shard = tmp_path / "data" / "000000_000002.tfrecord"
    payload = shard.read_bytes()
    shard.write_bytes(payload[: len(payload) // 2])

    assert _pending(tmp_path, 0, 5) == [0, 1, 2, 3, 4, 5]
    assert not (tmp_path / "data" / "000000_000002.count").exists()


@pytest.mark.parametrize("n_records", [0, 1, 3])
def test_raw_count_matches_tf_data(tmp_path, n_records):
    """The framing walk must agree with tf.data, which defines the format."""
    import tensorflow as tf

    from helpers import make_render_product

    cfg = WriterConfig(output_dir=tmp_path, overwrite=True, start_index=0, end_index=9)
    with TFRecordWriter(cfg, None) as writer:
        for i in range(n_records):
            writer.write_data(i, render_product=make_render_product(2, seed=i, index=i))

    shard = tmp_path / "data" / "000000_000009.tfrecord"
    expected = sum(1 for _ in tf.data.TFRecordDataset(str(shard), compression_type="ZLIB"))
    assert count_records(shard) == expected == n_records


def test_garbage_shard_counts_as_invalid(tmp_path):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    garbage = tmp_path / "data" / "000000_000009.tfrecord"
    garbage.write_bytes(b"this is not a zlib stream")
    assert count_records(garbage) == INVALID
