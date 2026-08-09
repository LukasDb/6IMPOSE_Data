from typing import Any
import json
import simpose as sp
import numpy as np
from .writer import Writer, WriterConfig
from .shard_index import ShardIndex, shard_range, write_sidecar

import silence_tensorflow.auto
import tensorflow as tf

class TFRecordWriter(Writer):
    def __init__(self, params: WriterConfig, comm: "sp.Comm | None" = None) -> None:
        super().__init__(params, comm)

    def __enter__(self) -> "TFRecordWriter":
        super().__enter__()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.output_dir.joinpath("metadata.json").open("w") as f:
            json.dump({"version": 4.0}, f, indent=2)

        self._data_dir = self.output_dir / "gt"
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir.joinpath("data").mkdir(parents=True, exist_ok=True)

        self._shard_path = (
            self.output_dir.joinpath("data")
            / f"{self.start_index:06}_{self.end_index:06}.tfrecord"
        )
        # counted here rather than re-derived from the finished file, so a
        # completed shard never has to be inflated to be known complete
        self._n_written = 0

        self._writer = tf.io.TFRecordWriter(
            str(self._shard_path),
            # level 1: ~3-5x faster than the default for ~10% larger files; the
            # compression runs serialized between renders in the worker
            options=tf.io.TFRecordOptions(compression_type="ZLIB", compression_level=1),
        )

        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        self._writer.close()
        # after close(), so the sidecar's size/mtime fingerprint matches the
        # final file. Lets the next resume skip inflating this shard entirely.
        write_sidecar(self._shard_path, self._n_written)
        return super().__exit__(type, value, traceback)

    def get_pending_indices(self) -> np.ndarray:
        indices = np.arange(self.start_index, self.end_index + 1)
        if self.overwrite:
            return indices

        # resume: keep shards that are complete and readable, regenerate the rest
        data_dir = self.output_dir / "data"
        if not data_dir.exists():
            return indices

        # shard completeness comes from shard_index.json when it can be trusted,
        # and from a parallel raw-zlib record count otherwise (which then updates
        # the index). Deliberately not tf.data here: that pulls CUDA into the
        # parent process, which then competes with the worker on that GPU.
        index = ShardIndex(self.output_dir)
        record_files = [f for f in sorted(data_dir.glob("*.tfrecord")) if shard_range(f)]
        for unnamed in sorted(set(data_dir.glob("*.tfrecord")) - set(record_files)):
            sp.logger.warning(f"Ignoring unrecognized record file {unnamed}")

        completed: set[int] = set()
        dropped = False
        for record_file, n_found in index.counts(record_files).items():
            bounds = shard_range(record_file)
            if bounds is None:
                continue  # filtered out above; keeps the type checker happy
            start, end = bounds
            n_expected = end - start + 1

            if n_found == n_expected:
                completed.update(range(start, end + 1))
            else:
                sp.logger.warning(
                    f"Shard {record_file.name} is incomplete or corrupt "
                    f"({n_found}/{n_expected} records) and will be regenerated."
                )
                record_file.unlink()
                index.drop(record_file)
                dropped = True
        if dropped:
            index.save()

        pending = np.array([i for i in indices if i not in completed], dtype=indices.dtype)
        if len(pending) < len(indices):
            sp.logger.info(
                f"Resuming: {len(indices) - len(pending)} datapoints already present, "
                f"{len(pending)} to generate."
            )
        return pending

    def _write_data(
        self,
        dataset_index: int,
        *,
        scene: sp.Scene | None,
        render_product: sp.RenderProduct | None = None,
    ) -> None:
        if scene is not None and render_product is None:
            sp.logger.debug(f"Generating data for {dataset_index}")
            scene.frame_set(dataset_index)
            render_product = scene.render(self.gpu_semaphore)
            sp.logger.debug("Raw data rendered.")
        elif scene is None and render_product is not None:
            sp.logger.debug(f"Writing data for {dataset_index}")
        else:
            raise ValueError("Either scene or render_product must be given.")

        rp = render_product

        sp_keys = sp.data.Dataset

        data: dict[str, np.ndarray] = {}
        if rp.intrinsics is not None:
            data[sp_keys.CAM_MATRIX] = rp.intrinsics.astype(np.float32)
        if rp.cam_position is not None:
            data[sp_keys.CAM_LOCATION] = np.array(rp.cam_position).astype(np.float32)
        if rp.cam_quat_xyzw is not None:
            data[sp_keys.CAM_ROTATION] = rp.cam_quat_xyzw.astype(np.float32)
        if rp.stereo_baseline is not None:
            data[sp_keys.STEREO_BASELINE] = np.array(rp.stereo_baseline).astype(np.float32)
        if rp.objs is not None and len(rp.objs) == 0:
            # frame without annotated objects: write empty tensors so the record
            # keeps the same keys (and per-object shapes) as its shard-mates
            data[sp_keys.OBJ_CLASSES] = np.zeros((0,), dtype=np.str_)
            data[sp_keys.OBJ_IDS] = np.zeros((0,), dtype=np.int32)
            data[sp_keys.OBJ_POS] = np.zeros((0, 3), dtype=np.float32)
            data[sp_keys.OBJ_ROT] = np.zeros((0, 4), dtype=np.float32)
            data[sp_keys.OBJ_BBOX_VISIB] = np.zeros((0, 4), dtype=np.int32)
            data[sp_keys.OBJ_BBOX_OBJ] = np.zeros((0, 4), dtype=np.int32)
            data[sp_keys.OBJ_PX_COUNT_VISIB] = np.zeros((0,), dtype=np.int32)
            data[sp_keys.OBJ_PX_COUNT_VALID] = np.zeros((0,), dtype=np.int32)
            data[sp_keys.OBJ_PX_COUNT_ALL] = np.zeros((0,), dtype=np.int32)
            data[sp_keys.OBJ_VISIB_FRACT] = np.zeros((0,), dtype=np.float32)
        elif rp.objs is not None:
            if rp.objs[0].cls is not None:
                data[sp_keys.OBJ_CLASSES] = np.array([obj.cls for obj in rp.objs])
            if rp.objs[0].object_id is not None:
                data[sp_keys.OBJ_IDS] = np.array([obj.object_id for obj in rp.objs]).astype(
                    np.int32
                )
            if rp.objs[0].position is not None:
                data[sp_keys.OBJ_POS] = np.array([obj.position for obj in rp.objs]).astype(
                    np.float32
                )
            if rp.objs[0].quat_xyzw is not None:
                data[sp_keys.OBJ_ROT] = np.array([obj.quat_xyzw for obj in rp.objs]).astype(
                    np.float32
                )
            if rp.objs[0].bbox_visib is not None:
                data[sp_keys.OBJ_BBOX_VISIB] = np.array(
                    [obj.bbox_visib for obj in rp.objs]
                ).astype(np.int32)
            if rp.objs[0].bbox_obj is not None:
                data[sp_keys.OBJ_BBOX_OBJ] = np.array([obj.bbox_obj for obj in rp.objs]).astype(
                    np.int32
                )
            if rp.objs[0].px_count_visib is not None:
                data[sp_keys.OBJ_PX_COUNT_VISIB] = np.array(
                    [obj.px_count_visib for obj in rp.objs]
                ).astype(np.int32)
            if rp.objs[0].px_count_valid is not None:
                data[sp_keys.OBJ_PX_COUNT_VALID] = np.array(
                    [obj.px_count_valid for obj in rp.objs]
                ).astype(np.int32)
            if rp.objs[0].px_count_all is not None:
                data[sp_keys.OBJ_PX_COUNT_ALL] = np.array(
                    [obj.px_count_all for obj in rp.objs]
                ).astype(np.int32)
            if rp.objs[0].visib_fract is not None:
                data[sp_keys.OBJ_VISIB_FRACT] = np.array(
                    [obj.visib_fract for obj in rp.objs]
                ).astype(np.float32)

        if rp.rgb is not None:
            data[sp_keys.RGB] = rp.rgb.astype(np.uint8)
        if rp.rgb_R is not None:
            data[sp_keys.RGB_R] = rp.rgb_R.astype(np.uint8)

        # v4: depth is float16 (was float32). ~1 mm quantization over the 0.5-2.5 m
        # working range, and roughly halves the datapoint size -- float32 depth both
        # doubled the bytes and compressed worse (28.6 MB vs 11.6 MB per datapoint).
        if rp.depth is not None:
            data[sp_keys.DEPTH] = rp.depth.astype(np.float16)
        if rp.depth_R is not None:
            data[sp_keys.DEPTH_R] = rp.depth_R.astype(np.float16)
        if rp.depth_GT is not None:
            data[sp_keys.DEPTH_GT] = rp.depth_GT.astype(np.float16)
        if rp.depth_GT_R is not None:
            data[sp_keys.DEPTH_GT_R] = rp.depth_GT_R.astype(np.float16)

        if rp.mask is not None:
            if rp.mask.max() > 255:
                raise ValueError(
                    "Object ids exceed 255 and cannot be stored in the uint8 mask."
                )
            data[sp_keys.MASK] = rp.mask.astype(np.uint8)

        with tf.device("/cpu:0"):  # type: ignore
            serialized_data = self._serialize_data(**data)
            self._writer.write(serialized_data)
            self._n_written += 1

        sp.logger.debug("Written to tfrecord.")

    def _serialize_data(self, **data: np.ndarray) -> Any:
        to_feature = lambda x: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])  # type: ignore
        )
        serialized_features = {k: to_feature(v) for k, v in data.items()}
        example_proto = tf.train.Example(features=tf.train.Features(feature=serialized_features))
        return example_proto.SerializeToString()

    def _cleanup(self, dataset_index: int) -> None:
        # records are written atomically per datapoint, so the shard file still
        # contains only valid, complete records: keep it. incomplete shards are
        # detected by their record count (get_pending_indices) or overwritten
        # when the shard is rescheduled.
        sp.logger.warning(
            f"Datapoint {dataset_index} failed. Keeping shard "
            f"{self.start_index:06}_{self.end_index:06}.tfrecord with the "
            "previously written records."
        )
