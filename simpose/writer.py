import bpy
import json
import simpose
import numpy as np
from pathlib import Path
from .redirect_stdout import redirect_stdout
import cv2
import logging

import signal


class DelayedKeyboardInterrupt:
    def __init__(self, index) -> None:
        self.index = index

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.warn(f"SIGINT received. Finishing rendering {self.index}...")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


class Writer:
    def __init__(self, scene: simpose.Scene, output_dir: Path, render_object_masks: bool):
        self._output_dir = output_dir
        self._data_dir = output_dir / "gt"
        self._scene = scene
        self._render_object_masks = render_object_masks
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._scene.set_output_path(self._output_dir)

    def generate_data(self, dataset_index: int):
        """dont allow CTRl+C during data generation"""
        with DelayedKeyboardInterrupt(dataset_index):
            try:
                self._generate_data(dataset_index)
            except Exception as e:
                # clean up possibly corrupted data
                logging.error(f"Error while generating data no. {dataset_index}")
                logging.error(e)
                self._cleanup(dataset_index)
                raise e

    def _generate_data(self, dataset_index: int):
        logging.debug(f"Generating data for {dataset_index}")
        self._scene.frame_set(dataset_index)  # this sets the suffix for file names

        # for each object, deactivate all but one and render mask
        objs = self._scene.get_labelled_objects()
        self._scene.render(render_object_masks=self._render_object_masks)

        mask = cv2.imread(
            str(Path(self._output_dir, "mask", f"mask_{dataset_index:04}.exr")),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )[..., 0]

        depth = cv2.imread(
            str(Path(self._output_dir, "depth", f"depth_{dataset_index:04}.exr")),
            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )[..., 0]
        depth[depth > 100.0] = 0.0

        obj_list = []
        for obj in objs:
            px_count_visib = np.count_nonzero(mask == obj.object_id)
            bbox_visib = self._get_bbox(mask, obj.object_id)
            bbox_obj = [0, 0, 0, 0]
            px_count_all = 0.0
            px_count_valid = 0.0
            visib_fract = 0.0

            if self._render_object_masks:
                obj_mask = cv2.imread(
                    str(
                        Path(
                            self._output_dir,
                            "mask",
                            f"mask_{obj.object_id:04}_{dataset_index:04}.exr",
                        )
                    ),
                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                )[..., 0]

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
                    "rotation": list(obj.rotation.as_quat()),
                    "bbox_visib": bbox_visib,
                    "bbox_obj": bbox_obj,
                    "px_count_visib": px_count_visib,
                    "px_count_valid": px_count_valid,
                    "px_count_all": px_count_all,
                    "visib_fract": visib_fract,
                }
            )

        cam = self._scene.get_cameras()[0]  # TODO in the future save all cameras
        cam_pos = cam.location
        cam_rot = cam.rotation
        cam_matrix = cam.get_calibration_matrix_K_from_blender()

        meta_dict = {
            "cam_rotation": list(cam_rot.as_quat()),
            "cam_location": list(cam_pos),
            "cam_matrix": np.array(cam_matrix).tolist(),
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
        # writer generates:
        # - mask/mask_{dataset_index:04}.exr
        # - depth/depth_{dataset_index:04}.exr
        # - mask/mask_{object_id:04}_{dataset_index:04}.exr
        # - gt/gt_{dataset_index:05}.json
        # - rgb/rgb_{dataset_index:04}.png
        # possibly for stereo also

        gt_path = self._data_dir / f"gt_{dataset_index:05}.json"
        if gt_path.exists():
            logging.debug(f"Removing {gt_path}")
            gt_path.unlink()

        rgb_path = self._output_dir / "rgb" / f"rgb_{dataset_index:04}.png"
        if rgb_path.exists():
            logging.debug(f"Removing {rgb_path}")
            rgb_path.unlink()

        mask_path = self._output_dir / "mask" / f"mask_{dataset_index:04}.exr"
        if mask_path.exists():
            logging.debug(f"Removing {mask_path}")
            mask_path.unlink()

        depth_path = self._output_dir / "depth" / f"depth_{dataset_index:04}.exr"
        if depth_path.exists():
            logging.debug(f"Removing {depth_path}")
            depth_path.unlink()

        if self._render_object_masks:
            mask_paths = (self._output_dir / "mask").glob(f"mask_*_{dataset_index:04}.exr")
            for mask_path in mask_paths:
                if mask_path.exists():
                    logging.debug(f"Removing {mask_path}")
                    mask_path.unlink()
