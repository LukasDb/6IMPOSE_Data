"""Completeness bookkeeping for the sharded TFRecord output.

`TFRecordWriter.get_pending_indices` needs to know which shards already hold a
full set of records. Counting them the obvious way -- iterating each shard with
`tf.data.TFRecordDataset` -- costs a full ZLIB inflate of the whole dataset in
the *parent* process before a single worker spawns (~31 min and 313 GB of inflate
on dropjects_lvis), which reads as a hung run and additionally drags CUDA into
the parent.

Three layers, cheapest first:

1. `shard_index.json` next to the data dir -- record counts keyed by shard name.
   Written incrementally as shards complete, so a resume is a few-KB read.
2. For anything the index doesn't vouch for, a parallel raw-zlib count that
   walks the record framing without materializing payloads (~7x faster than the
   serial tf.data scan, and it never touches the GPU).
3. The result of (2) is folded back into the index, so the expensive path runs
   at most once per shard.

Why the index is never trusted on its own: a stale "complete" entry silently
bakes missing datapoints into a dataset that looks finished. Each entry
therefore carries the shard's `size` and `mtime_ns`; if either has moved, the
entry is void and the shard is re-counted. A complete shard is written once and
never reopened, so that fingerprint is sound. The expected record count is
*never* stored -- it is derived from the filename (`023400_023449` -> 50), so a
changed index range can't be mistaken for a complete shard.
"""

import json
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, cast

import simpose as sp
import tqdm

INDEX_NAME = "shard_index.json"

# raw-zlib counting sentinel: unreadable, truncated, or otherwise not trustworthy
INVALID = -1


def shard_range(record_file: Path) -> tuple[int, int] | None:
    """`023400_023449.tfrecord` -> (23400, 23449); None if the name isn't a shard."""
    try:
        start, end = (int(x) for x in record_file.stem.split("_"))
    except ValueError:
        return None
    return start, end


def count_records(record_file: Path) -> int:
    """Number of complete records in a ZLIB TFRecord, or INVALID.

    A ZLIB-compressed TFRecord is a *single* zlib stream of concatenated
    `uint64 length | uint32 crc | payload | uint32 crc` frames, so the lengths
    are interleaved with the payloads and there is no way to reach the framing
    without inflating. What this avoids is *parsing* -- payload bytes are
    counted and dropped, never copied into Python objects or protos.

    Leftover bytes at EOF (a partial header, or a frame whose payload is short)
    mean the file was truncated mid-record -> INVALID.
    """
    decompressor = zlib.decompressobj()
    n_records = 0
    header = b""
    remaining = 0  # payload+crc bytes still to skip for the current record

    try:
        with record_file.open("rb") as f:
            while chunk := f.read(1 << 22):
                buf = decompressor.decompress(chunk)
                pos, size = 0, len(buf)
                while pos < size:
                    if remaining:
                        step = min(remaining, size - pos)
                        pos += step
                        remaining -= step
                        if remaining == 0:
                            n_records += 1
                        continue
                    take = min(8 - len(header), size - pos)
                    header += buf[pos : pos + take]
                    pos += take
                    if len(header) == 8:
                        (length,) = struct.unpack("<Q", header)
                        header = b""
                        remaining = 4 + length + 4  # crc + payload + crc
    except (zlib.error, OSError):
        return INVALID

    if remaining or header or not decompressor.eof:
        return INVALID  # truncated mid-record or mid-stream
    return n_records


def write_sidecar(record_file: Path, n_records: int) -> None:
    """Note a just-finished shard's record count next to the shard itself.

    Called by the worker that produced the shard. Workers run in separate
    processes and finish independently, so they must not read-modify-write the
    shared index -- 16 of them racing on one JSON would silently drop entries.
    Each writes its own `<shard>.count` instead, and the parent folds these into
    the index on the next scan (`ShardIndex.counts`). Best-effort: a missing
    sidecar just means that shard gets counted the expensive way once.
    """
    try:
        sidecar = record_file.with_suffix(".count")
        stat = record_file.stat()
        with sidecar.open("w") as f:
            json.dump(
                {"n_records": n_records, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}, f
            )
    except OSError as e:
        sp.logger.debug(f"Could not write shard sidecar for {record_file.name}: {e}")


class ShardIndex:
    """Record counts for the shards in `<output_dir>/data`, cached on disk."""

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir / INDEX_NAME
        self._entries: dict[str, dict[str, int]] = {}
        if self._path.exists():
            try:
                with self._path.open() as f:
                    index_file: object = json.load(f)
                if isinstance(index_file, dict):
                    shards: object = cast(dict[str, Any], index_file).get("shards", {})
                    if isinstance(shards, dict):
                        self._entries = cast(dict[str, dict[str, int]], shards)
            except (json.JSONDecodeError, OSError):
                sp.logger.warning(f"Could not read {INDEX_NAME}; rebuilding it from the shards.")

    def _validate(self, entry: dict[str, int], record_file: Path) -> int | None:
        """The entry's count, if it still describes the shard on disk."""
        try:
            stat = record_file.stat()
        except OSError:
            return None
        if entry.get("size") != stat.st_size or entry.get("mtime_ns") != stat.st_mtime_ns:
            return None  # shard changed since it was indexed
        count = entry.get("n_records")
        return count if isinstance(count, int) else None

    def _fresh(self, record_file: Path) -> int | None:
        """Indexed count for `record_file`, or None if not vouched for.

        Checks the index proper, then any sidecar left by the worker that wrote
        the shard (adopting it into the index so the sidecar is only read once).
        """
        entry = self._entries.get(record_file.name)
        if entry is not None:
            count = self._validate(entry, record_file)
            if count is not None:
                return count

        sidecar = record_file.with_suffix(".count")
        if not sidecar.exists():
            return None
        try:
            with sidecar.open() as f:
                loaded: object = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(loaded, dict):
            return None
        sidecar_entry = cast(dict[str, int], loaded)
        count = self._validate(sidecar_entry, record_file)
        if count is not None:
            self._entries[record_file.name] = sidecar_entry
        return count

    def update(self, record_file: Path, n_records: int) -> None:
        """Record `n_records` for a shard, fingerprinted against its current stat."""
        try:
            stat = record_file.stat()
        except OSError:
            return
        self._entries[record_file.name] = {
            "n_records": n_records,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def drop(self, record_file: Path) -> None:
        """Forget a shard, including any sidecar, so nothing stale outlives it."""
        self._entries.pop(record_file.name, None)
        record_file.with_suffix(".count").unlink(missing_ok=True)

    def save(self) -> None:
        """Persist atomically -- a torn index would force a full rescan next run."""
        tmp = self._path.with_suffix(".json.tmp")
        try:
            with tmp.open("w") as f:
                json.dump({"version": 1, "shards": self._entries}, f, indent=2)
            tmp.replace(self._path)
        except OSError as e:
            sp.logger.warning(f"Could not write {INDEX_NAME}: {e}")

    def counts(self, record_files: Iterable[Path], n_threads: int = 32) -> dict[Path, int]:
        """Record count per shard, using the index and counting only the rest.

        Counting is threaded because `zlib.decompress` drops the GIL, so this
        scales with cores despite being CPU-bound. Progress is reported over the
        *bytes* to inflate rather than the file count, since that is what the
        time is actually proportional to.
        """
        record_files = list(record_files)
        counts: dict[Path, int] = {}
        todo: list[Path] = []

        for record_file in record_files:
            cached = self._fresh(record_file)
            if cached is None:
                todo.append(record_file)
            else:
                counts[record_file] = cached

        if counts:
            sp.logger.info(f"Shard index: {len(counts)}/{len(record_files)} shards known.")
        if not todo:
            # counts may include entries just adopted from worker sidecars, so
            # persist even when nothing had to be inflated
            self.save()
            return counts

        total_bytes = sum(f.stat().st_size for f in todo)
        sp.logger.info(
            f"Counting records in {len(todo)} shard(s) "
            f"({total_bytes / 1e9:.1f} GB to inflate) using {n_threads} threads."
        )

        with tqdm.tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Verifying shards",
            dynamic_ncols=True,
        ) as bar:
            with ThreadPoolExecutor(n_threads) as pool:
                for record_file, n_records in zip(todo, pool.map(count_records, todo)):
                    counts[record_file] = n_records
                    self.update(record_file, n_records)
                    bar.update(record_file.stat().st_size)

        self.save()
        return counts
