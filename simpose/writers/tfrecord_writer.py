from typing import Any
import simpose as sp
import numpy as np
from .writer import Writer, WriterConfig
import multiprocessing as mp

import silence_tensorflow.auto
import tensorflow as tf

tf.config.set_soft_device_placement(False)


class TFRecordWriter(Writer):
    def __init__(self, params: WriterConfig, comm: mp.queues.Queue) -> None:
        super().__init__(params, comm)

    def __enter__(self) -> "TFRecordWriter":
        super().__enter__()
        self._data_dir = self.output_dir / "gt"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._writers = {}
        for name in ["rgb", "gt", "depth", "mask"]:
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
            self._writers[name] = tf.io.TFRecordWriter(
                str(
                    self.output_dir / name / f"{self.start_index:06}_{self.end_index:06}.tfrecord"
                ),
                options=tf.io.TFRecordOptions(compression_type="ZLIB"),
            )

        return self

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        for writer in self._writers.values():
            writer.close()
        return super().__exit__(type, value, traceback)

    def get_pending_indices(self) -> np.ndarray:
        if not self.overwrite:
            raise NotImplementedError(
                "'Not overwrite' mode not implemented yet. Please set 'overwrite' to True."
            )

        else:
            indices = np.arange(self.start_index, self.end_index + 1)
        return indices

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

        gt_data: dict[str, np.ndarray] = {}
        if rp.intrinsics is not None:
            gt_data["cam_matrix"] = rp.intrinsics.astype(np.float32)
        if rp.cam_position is not None:
            gt_data["cam_location"] = np.array(rp.cam_position).astype(np.float32)
        if rp.cam_quat_xyzw is not None:
            gt_data["cam_rotation"] = rp.cam_quat_xyzw.astype(np.float32)
        if rp.stereo_baseline is not None:
            gt_data["stereo_baseline"] = np.array(rp.stereo_baseline).astype(np.float32)
        if rp.objs is not None:
            if rp.objs[0].cls is not None:
                gt_data["obj_classes"] = np.array([obj.cls for obj in rp.objs])
            if rp.objs[0].object_id is not None:
                gt_data["obj_ids"] = np.array([obj.object_id for obj in rp.objs]).astype(np.int32)
            if rp.objs[0].position is not None:
                gt_data["obj_pos"] = np.array([obj.position for obj in rp.objs]).astype(np.float32)
            if rp.objs[0].quat_xyzw is not None:
                gt_data["obj_rot"] = np.array([obj.quat_xyzw for obj in rp.objs]).astype(
                    np.float32
                )
            if rp.objs[0].bbox_visib is not None:
                gt_data["obj_bbox_visib"] = np.array([obj.bbox_visib for obj in rp.objs]).astype(
                    np.int32
                )
            if rp.objs[0].bbox_obj is not None:
                gt_data["obj_bbox_obj"] = np.array([obj.bbox_obj for obj in rp.objs]).astype(
                    np.int32
                )
            if rp.objs[0].px_count_visib is not None:
                gt_data["obj_px_count_visib"] = np.array(
                    [obj.px_count_visib for obj in rp.objs]
                ).astype(np.int32)
            if rp.objs[0].px_count_valid is not None:
                gt_data["obj_px_count_valid"] = np.array(
                    [obj.px_count_valid for obj in rp.objs]
                ).astype(np.int32)
            if rp.objs[0].px_count_all is not None:
                gt_data["obj_px_count_all"] = np.array(
                    [obj.px_count_all for obj in rp.objs]
                ).astype(np.int32)
            if rp.objs[0].visib_fract is not None:
                gt_data["obj_visib_fract"] = np.array([obj.visib_fract for obj in rp.objs]).astype(
                    np.float32
                )

        rgb = {}
        if rp.rgb is not None:
            rgb["rgb"] = rp.rgb.astype(np.uint8)
        if rp.rgb_R is not None:
            rgb["rgb_R"] = rp.rgb_R.astype(np.uint8)

        depth = {}
        if rp.depth is not None:
            depth["depth"] = rp.depth.astype(np.float32)
        if rp.depth_R is not None:
            depth["depth_R"] = rp.depth_R.astype(np.float32)

        with tf.device("/cpu:0"):  # type: ignore
            serialized_RGBs = self._serialize_data(**rgb)
            self._writers["rgb"].write(serialized_RGBs)

            serialized_gt = self._serialize_data(**gt_data)
            self._writers["gt"].write(serialized_gt)

            if rp.mask is not None:
                serialized_mask = self._serialize_data(mask=rp.mask.astype(np.uint8))
                self._writers["mask"].write(serialized_mask)

            serialized_depths = self._serialize_data(**depth)
            self._writers["depth"].write(serialized_depths)
        sp.logger.debug("Written to tfrecord.")

        # self.remove_temporary_files(dataset_index)
        sp.logger.debug("Temporary files removed.")

    def _serialize_data(self, **data: np.ndarray) -> Any:
        to_feature = lambda x: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])  # type: ignore
        )
        serialized_features = {k: to_feature(v) for k, v in data.items()}
        example_proto = tf.train.Example(features=tf.train.Features(feature=serialized_features))
        return example_proto.SerializeToString()

    def _cleanup(self, dataset_index: int) -> None:
        # self.remove_temporary_files(dataset_index)

        # delete incomplete tfrecord files
        for name in ["rgb", "gt", "depth", "mask"]:
            record_path = (
                self.output_dir / name / f"{self.start_index:06}_{self.end_index:06}.tfrecord"
            )
            if record_path.exists():
                sp.logger.debug(f"Removing {record_path}")
                record_path.unlink()
