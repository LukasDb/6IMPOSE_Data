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
from mpi4py import MPI

# "rgb": rgb,
# "depth": depth,
# "mask": mask,  # we only save the visible mask *not* the object masks
# "cam_matrix": cam_matrix,
# "cam_pos": cam_pos,
# "cam_rot": cam_rot.as_quat(canonical=True),
# "stereo_baseline": np.array(cam.baseline),


class H5Writer(Writer):
    def __init__(
        self,
        params: WriterConfig,
    ):
        super().__init__(params)
        self._data_dir = self.output_dir / "gt"
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def post_process(self):
        """gets called by a single worker after all workers are done"""
        files = sorted(list(self.output_dir.glob("*.h5")))
        vds_len = 0
        for file in files:
            with h5py.File(file, "r") as F:
                indices = F["indices"]
                assert isinstance(indices, h5py.Dataset)
                vds_len += len(np.unique(indices))

        with h5py.File(files[0], "r") as F:
            ds_keys = list([x for x in F.keys() if isinstance(F[x], h5py.Dataset)])
            ds_shapes = {key: F[key].shape for key in ds_keys}
            ds_dtypes = {key: F[key].dtype for key in ds_keys}

        for key in ds_keys:
            vds_shape = (vds_len, *ds_shapes[key][1:])
            layout = h5py.VirtualLayout(shape=vds_shape, dtype=ds_dtypes[key])

            for file in files:
                start_ind = int(file.stem.split("_")[-1])
                with h5py.File(file, "r") as F:
                    ds_shape = F[key].shape
                vsource = h5py.VirtualSource(file, key, shape=ds_shape)
                layout[start_ind : h5py.h5s.UNLIMITED] = vsource[: h5py.h5s.UNLIMITED]

            with h5py.File(self.output_dir / f"data.h5", "a") as F:
                F.create_virtual_dataset(key, layout, fillvalue=0)

        with h5py.File(self.output_dir / f"data.h5", "a") as target_F:
            for file in files:
                # link each obj into the target file
                with h5py.File(file, "r") as source_F:
                    objs_group = source_F["objs"]
                    assert isinstance(objs_group, h5py.Group)
                    source_objs = list(objs_group.keys())

                for source_key in source_objs:
                    target_key = int(source_key) + int(file.stem.split("_")[-1])
                    target_F.require_group("objs")[f"{target_key:06}"] = h5py.ExternalLink(
                        file, f"/objs/{source_key}"
                    )

    def get_pending_indices(self):
        # TODO
        if not self.overwrite and (self.output_dir / f"data.h5").exists():
            # wait for available file
            F = None
            while F is None:
                try:
                    F = h5py.File(self.output_dir / f"data.h5", "r")
                except OSError:
                    time.sleep(0.1)
                    continue
            existing_ids = F["indices"]
            assert isinstance(existing_ids, h5py.Dataset)
            existing_ids = np.unique(existing_ids)
            F.close()

            assert isinstance(existing_ids, np.ndarray)
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

        meta_dict = {
            "cam_rotation": list(cam_rot.as_quat(canonical=True)),
            "cam_location": list(cam_pos),
            "cam_matrix": np.array(cam_matrix).tolist(),
            "stereo_baseline": "none" if not cam.is_stereo_camera() else cam.baseline,
            "objs": list(obj_list),
        }

        # --- WRITE TO H5 ---
        # h5_index = dataset_index # if one file for all data
        h5_index = dataset_index - self.start_index  # if one file per worker
        h5_name = f"data_{self.start_index}.h5"
        # h5_length = self.end_index + 1 # if one file for all data
        h5_length = self.end_index - self.start_index + 1  # if one file per worker

        rgb = np.array(Image.open(Path(self.output_dir, "rgb", f"rgb_{dataset_index:04}.png")))
        matrices: dict[str, np.ndarray] = {
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

        # --- write to worker h5
        h5file = None
        # writing to H5 takes longer than rendering! -> do it in parallel
        while h5file is None:
            try:
                h5file = h5py.File(self.output_dir / h5_name, "a")
            except OSError:
                # sp.logger.debug("Waiting for h5 file to be free...")
                time.sleep(0.1)
                continue

        create_kwargs = {"compression": "lzf", "chunks": True}  # fast

        logging.debug(f"Opened {h5_name} and writing images...")
        for key, value in matrices.items():
            ds_shape = (h5_length, *value.shape)  # create max length dataset
            maxshape = (None, *value.shape)
            dataset = h5file.require_dataset(
                key,
                shape=ds_shape,
                dtype=value.dtype,
                maxshape=maxshape,
                exact=True,
                **create_kwargs,
            )
            if len(dataset) < h5_length:
                logging.debug(f"Resizing {key} dataset to {h5_length}")
                dataset.resize(h5_length, axis=0)
            dataset[h5_index] = value

        logging.debug(f"Writing metadata...")
        obj_group_for_img = h5file.require_group("objs").require_group(f"{h5_index:06}")
        for key in obj_list[0].keys():
            if key in obj_group_for_img.keys():
                del obj_group_for_img[key]
            obj_group_for_img.create_dataset(key, data=[x[key] for x in obj_list], **create_kwargs)

        inds_ds = h5file.require_dataset(
            "indices",
            shape=(h5_length,),
            dtype=np.int32,
            maxshape=(None,),
            exact=True,
            **create_kwargs,
        )
        if len(inds_ds) < h5_length:
            logging.debug(f"Resizing 'indices' dataset to {h5_length}")
            inds_ds.resize(h5_length, axis=0)
        inds_ds[h5_index] = dataset_index

        h5file.close()

        if False:  # Keep old simpose dataset?
            with (self._data_dir / f"gt_{dataset_index:05}.json").open("w") as F:
                json.dump(meta_dict, F, indent=2)
        else:
            self._cleanup(dataset_index)

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
