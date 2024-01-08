import json
import simpose as sp
import numpy as np
from pathlib import Path
import cv2
import logging
from ..exr import EXR
from .writer import Writer, WriterConfig
import multiprocessing as mp
from PIL import Image

import silence_tensorflow.auto
import tensorflow as tf

tf.config.set_soft_device_placement(False)


class TFRecordWriter(Writer):
    def __init__(
        self, params: WriterConfig, gpu_semaphore=None, rendered_dict: dict | None = None
    ):
        super().__init__(params, gpu_semaphore, rendered_dict)

    def __enter__(self):
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

        return super().__enter__()

    def __exit__(self, type, value, traceback):
        for writer in self._writers.values():
            writer.close()
        return super().__exit__(type, value, traceback)

    def get_pending_indices(self):
        if not self.overwrite:
            raise NotImplementedError("Not implemented yet")

        else:
            indices = np.arange(self.start_index, self.end_index + 1)
        return indices

    def _write_data(self, scene: sp.Scene, dataset_index: int):
        sp.logger.debug(f"Generating data for {dataset_index}")
        scene.frame_set(dataset_index)  # this sets the suffix for file names

        scene.render(self.gpu_semaphore)

        depth = np.array(
            cv2.imread(
                str(Path(self.output_dir, "depth", f"depth_{dataset_index:04}.exr")),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
        )[..., 0].astype(np.float32)
        depth[depth > 100.0] = 0.0

        mask_path = Path(self.output_dir / f"mask/mask_{dataset_index:04}.exr")
        mask = EXR(mask_path).read("visib.R")
        # for each object, deactivate all but one and render mask
        # only for labelled objects we have a mask
        objs = scene.get_labelled_objects()
        obj_list = []
        for obj in objs:
            px_count_visib = np.count_nonzero(mask == obj.object_id)
            bbox_visib = self._get_bbox(mask, obj.object_id)
            bbox_obj = [0, 0, 0, 0]
            px_count_all = 0.0
            px_count_valid = 0.0
            visib_fract = 0.0

            obj_mask = EXR(mask_path).read(f"{obj.object_id:04}.R")

            bbox_obj = self._get_bbox(obj_mask, 1)
            px_count_all = np.count_nonzero(obj_mask == 1)
            px_count_valid = np.count_nonzero(depth[mask == obj.object_id])
            if px_count_all != 0:
                visib_fract = px_count_visib / px_count_all

            obj_list.append(
                {
                    "class": obj.get_class(),
                    "object id": obj.object_id,
                    "pos": list(obj.location),
                    "rotation": list(obj.rotation.as_quat(canonical=False)),
                    "bbox_visib": bbox_visib,
                    "bbox_obj": bbox_obj,
                    "px_count_visib": px_count_visib,
                    "px_count_valid": px_count_valid,
                    "px_count_all": px_count_all,
                    "visib_fract": visib_fract,
                }
            )

        cam = scene.get_cameras()[0]  # TODO in the future save all cameras
        cam_pos = cam.location
        cam_rot = cam.rotation
        cam_matrix = cam.calculate_intrinsics(scene.resolution_x, scene.resolution_y)

        rgb = np.array(
            Image.open(str(Path(self.output_dir, "rgb", f"rgb_{dataset_index:04}.png")))
        )
        rgb_R = np.array(
            Image.open(str(Path(self.output_dir, "rgb", f"rgb_{dataset_index:04}_R.png")))
        )

        depth_R = np.array(
            cv2.imread(
                str(Path(self.output_dir, "depth", f"depth_{dataset_index:04}_R.exr")),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
        )[..., 0].astype(np.float32)
        depth[depth > 100.0] = 0.0

        # possible shape of gt:
        # class: [n, n_object, 1] # < but this is string? problem?
        # ID:    [n, n_object, 1]
        # pos:   [n, n_object, 3]
        # rot:   [n, n_object, 4]
        # and so on
        gt_data = {
            "cam_matrix": cam_matrix.astype(np.float32),
            "cam_location": np.array(cam_pos).astype(np.float32),
            "cam_rotation": cam_rot.as_quat(False).astype(np.float32),
            "stereo_baseline": np.array(cam.baseline).astype(np.float32),
            "obj_classes": np.array([obj["class"] for obj in obj_list]),
            "obj_ids": np.array([obj["object id"] for obj in obj_list]).astype(np.int32),
            "obj_pos": np.array([obj["pos"] for obj in obj_list]).astype(np.float32),
            "obj_rot": np.array([obj["rotation"] for obj in obj_list]).astype(np.float32),
            "obj_bbox_visib": np.array([obj["bbox_visib"] for obj in obj_list]).astype(np.int32),
            "obj_bbox_obj": np.array([obj["bbox_obj"] for obj in obj_list]).astype(np.int32),
            "obj_px_count_visib": np.array([obj["px_count_visib"] for obj in obj_list]).astype(
                np.int32
            ),
            "obj_px_count_valid": np.array([obj["px_count_valid"] for obj in obj_list]).astype(
                np.int32
            ),
            "obj_px_count_all": np.array([obj["px_count_all"] for obj in obj_list]).astype(
                np.int32
            ),
            "obj_visib_fract": np.array([obj["visib_fract"] for obj in obj_list]).astype(
                np.float32
            ),
        }

        with tf.device("/cpu:0"):  # type: ignore
            serialized_rgbs = self._serizalize_data(
                rgb=rgb.astype(np.uint8), rgb_R=rgb_R.astype(np.uint8)
            )
            self._writers["rgb"].write(serialized_rgbs)

            serialized_gt = self._serizalize_data(**gt_data)
            self._writers["gt"].write(serialized_gt)

            serialized_mask = self._serizalize_data(mask=mask.astype(np.uint8))
            self._writers["mask"].write(serialized_mask)

            serialized_depths = self._serizalize_data(
                depth=depth.astype(np.float32), depth_R=depth_R.astype(np.float32)
            )
            self._writers["depth"].write(serialized_depths)

        # here I could clean up the temporary files
        self.remove_temporary_files(dataset_index)

    def _serizalize_data(self, **data):
        to_feature = lambda x: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])
        )
        serialized_features = {k: to_feature(v) for k, v in data.items()}
        example_proto = tf.train.Example(features=tf.train.Features(feature=serialized_features))
        return example_proto.SerializeToString()

    def _get_bbox(self, mask, object_id):
        y, x = np.where(mask == object_id)
        if len(y) == 0:
            return [0, 0, 0, 0]
        x1 = np.min(x).tolist()
        x2 = np.max(x).tolist()
        y1 = np.min(y).tolist()
        y2 = np.max(y).tolist()
        return [x1, y1, x2, y2]

    def _cleanup(self, dataset_index: int):
        self.remove_temporary_files(dataset_index)

        # delete incomplete tfrecord files
        for name in ["rgb", "gt", "depth", "mask"]:
            record_path = (
                self.output_dir / name / f"{self.start_index:06}_{self.end_index:06}.tfrecord"
            )
            if record_path.exists():
                sp.logger.debug(f"Removing {record_path}")
                record_path.unlink()

    def remove_temporary_files(self, dataset_index):
        gt_path = self._data_dir / f"gt_{dataset_index:05}.json"
        if gt_path.exists():
            sp.logger.debug(f"Removing {gt_path}")
            gt_path.unlink()

        rgb_path = self.output_dir / "rgb" / f"rgb_{dataset_index:04}.png"
        if rgb_path.exists():
            sp.logger.debug(f"Removing {rgb_path}")
            rgb_path.unlink()

        rgb_R_path = self.output_dir / "rgb" / f"rgb_{dataset_index:04}_R.png"
        if rgb_R_path.exists():
            sp.logger.debug(f"Removing {rgb_R_path}")
            rgb_R_path.unlink()

        mask_path = self.output_dir / "mask" / f"mask_{dataset_index:04}.exr"
        if mask_path.exists():
            sp.logger.debug(f"Removing {mask_path}")
            mask_path.unlink()

        depth_path = self.output_dir / "depth" / f"depth_{dataset_index:04}.exr"
        if depth_path.exists():
            sp.logger.debug(f"Removing {depth_path}")
            depth_path.unlink()

        depth_R_path = self.output_dir / "depth" / f"depth_{dataset_index:04}_R.exr"
        if depth_R_path.exists():
            sp.logger.debug(f"Removing {depth_R_path}")
            depth_R_path.unlink()

        mask_paths = (self.output_dir / "mask").glob(f"mask_*_{dataset_index:04}.exr")
        for mask_path in mask_paths:
            sp.logger.debug(f"Removing {mask_path}")
            mask_path.unlink()
