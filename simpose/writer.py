import bpy
import json
import simpose
import numpy as np
from pathlib import Path
from .redirect_stdout import redirect_stdout
import cv2


class Writer:
    def __init__(self, scene: simpose.Scene, output_dir: Path):
        self._output_dir = output_dir
        self._data_dir = output_dir / "gt"
        self._scene = scene
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._scene.set_output_path(self._output_dir)

    """ contains logic to write the dataset """

    def generate_data(self, dataset_index: int):
        self._scene.frame_set(dataset_index)  # this sets the suffix for file names

        # for each object, deactivate all but one and render mask
        objs = self._scene.get_objects()
        self._scene.render_rgb_and_depth()
        self._scene.render_masks()
        
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

            bbox_visib = self._get_bbox(mask, obj.object_id)
            bbox_obj = self._get_bbox(obj_mask, 1)

            px_count_all = np.count_nonzero(obj_mask == 1)
            px_count_visib = np.count_nonzero(mask == obj.object_id)
            px_count_valid = np.count_nonzero(depth[mask == obj.object_id])
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
