import bpy
import json
import simpose
import numpy as np
from pathlib import Path
from .redirect_stdout import redirect_stdout
import cv2
import logging


class Writer:
    def __init__(self, scene: simpose.Scene, output_dir: Path, render_object_masks: bool):
        self._output_dir = output_dir
        self._data_dir = output_dir / "gt"
        self._scene = scene
        self._render_object_masks = render_object_masks
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._scene.set_output_path(self._output_dir)

    """ contains logic to write the dataset """

    def generate_data(self, dataset_index: int):
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
