import time
import json
import simpose as sp
import numpy as np
from pathlib import Path
import cv2
import logging
from ..exr import EXR
import h5py
from PIL import Image
from .writer import Writer, WriterConfig


class H5Writer(Writer):
    def __init__(
        self,
        params: WriterConfig,
    ):
        super().__init__(params)
        self._data_dir = self.output_dir / "gt"
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def get_pending_indices(self):
        if not self.overwrite:
            existing_files = self.output_dir.joinpath("gt").glob("*.json")
            existing_ids = [int(x.stem.split("_")[-1]) for x in existing_files]
            indices = np.setdiff1d(np.arange(self.start_index, self.end_index + 1), existing_ids)
        else:
            indices = np.arange(self.start_index, self.end_index + 1)
        return indices

    def _write_data(self, scene: sp.Scene, dataset_index: int):
        sp.logger.debug(f"Generating data for {dataset_index}")
        scene.frame_set(dataset_index)  # this sets the suffix for file names

        # for each object, deactivate all but one and render mask
        objs = scene.get_labelled_objects()
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
        cam_pos = np.array(cam.location)
        cam_rot = cam.rotation
        cam_matrix = cam.calculate_intrinsics(scene.resolution_x, scene.resolution_y)

        # TODO DONT write unnecessary gt
        meta_dict = {
            "cam_rotation": list(cam_rot.as_quat(canonical=True)),
            "cam_location": list(cam_pos),
            "cam_matrix": np.array(cam_matrix).tolist(),
            "stereo_baseline": "none" if not cam.is_stereo_camera() else cam.baseline,
            "objs": list(obj_list),
        }

        with (self._data_dir / f"gt_{dataset_index:05}.json").open("w") as F:
            json.dump(meta_dict, F, indent=2)

        # --- WRITE TO H5 ---
        # h5_index = dataset_index - self.start_index
        h5_index = dataset_index
        rgb = np.array(Image.open(Path(self.output_dir, "rgb", f"rgb_{dataset_index:04}.png")))
        matrices = {
            "rgb": rgb,
            "depth": depth,
            "mask": mask,  # we only save the visible mask *not* the object masks
            "cam_matrix": cam_matrix,
            "cam_pos": cam_pos,
            "cam_rot": cam_rot.as_quat(canonical=True),
            "stereo_baseline": np.array(cam.baseline),
        }

        rgb_R_path = self.output_dir / "rgb" / f"rgb_{dataset_index:04}_R.png"
        depth_R_path = self.output_dir / "depth" / f"depth_{dataset_index:04}_R.exr"
        if rgb_R_path.exists():
            rgb_R = np.array(Image.open(rgb_R_path))
            matrices["rgb_R"] = rgb_R
        if depth_R_path.exists():
            depth_R = np.array(
                cv2.imread(
                    str(depth_R_path),
                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                )
            )[..., 0].astype(np.float32)
            matrices["depth_R"] = depth_R

        h5file = None
        while h5file is None:
            try:
                h5file = h5py.File(self.output_dir / "data.h5", "a")
            except OSError:
                sp.logger.debug("Waiting for h5 file to be free...")
                time.sleep(1.0)
                continue

        for key, value in matrices.items():
            if key not in h5file.keys():
                ds_shape = (self.end_index + 1, *value.shape)  # create max length dataset
                h5file.create_dataset(
                    key,
                    shape=ds_shape,
                    dtype=value.dtype,
                )
            dataset = h5file[key]
            assert isinstance(dataset, h5py.Dataset)

            if len(dataset) < self.end_index + 1:
                dataset.resize(self.end_index + 1, axis=0)

            dataset[h5_index] = value
            # dataset.attrs["length"] = h5_index + 1

        # how to do obj list?
        if "objs" not in h5file.keys():
            h5file.create_group("objs")
        all_objs_group = h5file["objs"]
        assert isinstance(all_objs_group, h5py.Group)
        # if overwrite then delete the group
        if f"{h5_index:06}" not in all_objs_group.keys():
            obj_group_for_img = all_objs_group.create_group(f"{h5_index:06}")

        obj_group_for_img = all_objs_group[f"{h5_index:06}"]
        assert isinstance(obj_group_for_img, h5py.Group)
        for key in obj_list[0].keys():
            if key in obj_group_for_img.keys():
                del obj_group_for_img[key]
            obj_group_for_img.create_dataset(key, data=[x[key] for x in obj_list])

        h5file.close()
        # then delete the files since everything is in h5 now
        # self._cleanup(dataset_index)

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

        depth_R_path = self.output_dir / "depth" / f"depth_{dataset_index:04}_R.exr"
        if depth_R_path.exists():
            sp.logger.debug(f"Removing {depth_R_path}")
            depth_R_path.unlink()

        mask_paths = (self.output_dir / "mask").glob(f"mask_*_{dataset_index:04}.exr")
        for mask_path in mask_paths:
            if mask_path.exists():
                sp.logger.debug(f"Removing {mask_path}")
                mask_path.unlink()
