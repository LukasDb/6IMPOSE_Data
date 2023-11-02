import contextlib
import json
import simpose as sp
import numpy as np
from pathlib import Path
import cv2
import logging
from ..exr import EXR
from .writer import Writer, WriterConfig

# import tensorflow and set gpu growth
# import tensorflow as tf

# for dev in tf.config.list_physical_devices("GPU"):
#     tf.config.experimental.set_memory_growth(dev, True)


class TFRecordWriter(Writer):
    def __init__(
        self,
        params: WriterConfig,
    ):
        super().__init__(params)
        self._data_dir = self.output_dir / "gt"
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def get_pending_indices(self):
        if not self.overwrite:
            # existing_files = self.output_dir.joinpath("gt").glob("*.json")
            # existing_ids = [int(x.stem.split("_")[-1]) for x in existing_files]
            # indices = np.setdiff1d(np.arange(self.start_index, self.end_index + 1), existing_ids)
            # for tfrecords, indices do not matter

            def get_length(file):
                return sum(1 for _ in tf.data.TFRecordDataset(file, compression_type="ZLIB"))

            files = tf.io.matching_files(str(self.output_dir / "*.tfrecord"))

            total_length = (
                tf.data.Dataset.from_tensor_slices(files)
                .map(get_length, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                .reduce(0, lambda x, y: x + y)
                .numpy()
            )

        else:
            indices = np.arange(self.start_index, self.end_index + 1)
        return indices

    def _write_data(self, scene: sp.Scene, dataset_index: int, gpu_semaphore=None):
        sp.logger.debug(f"Generating data for {dataset_index}")
        scene.frame_set(dataset_index)  # this sets the suffix for file names

        # for each object, deactivate all but one and render mask
        objs = scene.get_labelled_objects()
        if gpu_semaphore is None:
            gpu_semaphore = contextlib.nullcontext()
        with gpu_semaphore:
            scene.render()

        depth = np.array(
            cv2.imread(
                str(Path(self.output_dir, "depth", f"depth_{dataset_index:04}.exr")),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
        )[..., 0].astype(np.float32)
        depth[depth > 100.0] = 0.0

        mask_path = Path(self.output_dir / f"mask/mask_{dataset_index:04}.exr")
        mask = EXR(mask_path).read("visib.R")
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

        meta_dict = {
            "cam_rotation": list(cam_rot.as_quat(canonical=True)),
            "cam_location": list(cam_pos),
            "cam_matrix": np.array(cam_matrix).tolist(),
            "stereo_baseline": "none" if not cam.is_stereo_camera() else cam.baseline,
            "objs": list(obj_list),
        }

        with (self._data_dir / f"gt_{dataset_index:05}.json").open("w") as F:
            json.dump(meta_dict, F, indent=2)

    def _get_bbox(self, mask, object_id):
        y, x = np.where(mask == object_id)
        if len(y) == 0:
            return [0, 0, 0, 0]
        x1 = np.min(x).tolist()
        x2 = np.max(x).tolist()
        y1 = np.min(y).tolist()
        y2 = np.max(y).tolist()
        return [x1, y1, x2, y2]

    def _cleanup(self, dataset_index):
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

        mask_paths = (self.output_dir / "mask").glob(f"mask_*_{dataset_index:04}.exr")
        for mask_path in mask_paths:
            if mask_path.exists():
                sp.logger.debug(f"Removing {mask_path}")
                mask_path.unlink()
