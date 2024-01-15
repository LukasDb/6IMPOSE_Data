import simpose as sp
from .dataset import Dataset
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image
from typing import Generator


class SimposeDataset(Dataset):
    _gt_keys = [
        Dataset.CAM_MATRIX,
        Dataset.CAM_LOCATION,
        Dataset.CAM_ROTATION,
        Dataset.STEREO_BASELINE,
        Dataset.OBJ_CLASSES,
        Dataset.OBJ_IDS,
        Dataset.OBJ_POS,
        Dataset.OBJ_ROT,
        Dataset.OBJ_BBOX_VISIB,
        Dataset.OBJ_VISIB_FRACT,
        Dataset.OBJ_PX_COUNT_VISIB,
        Dataset.OBJ_PX_COUNT_VALID,
        Dataset.OBJ_PX_COUNT_ALL,
        Dataset.OBJ_BBOX_OBJ,
    ]

    @staticmethod
    def get(root_dir: Path, get_keys: None | list[str] = None) -> tf.data.Dataset:
        def generator() -> Generator:
            indices = [int(f.name[3:8]) for f in (root_dir / "gt").iterdir() if f.is_file()]

            for idx in indices:
                data = {}
                if get_keys is None or any([k for k in get_keys if k in SimposeDataset._gt_keys]):
                    with (root_dir / "gt" / f"gt_{idx:05}.json").open("r") as F:
                        shot = json.load(F)

                    if get_keys is None or Dataset.CAM_MATRIX in get_keys:
                        data[Dataset.CAM_MATRIX] = np.array(shot["cam_matrix"])

                    if get_keys is None or Dataset.CAM_ROTATION in get_keys:
                        data[Dataset.CAM_ROTATION] = np.array(shot["cam_rotation"])

                    if get_keys is None or Dataset.CAM_LOCATION in get_keys:
                        data[Dataset.CAM_LOCATION] = np.array(shot["cam_location"])

                    if get_keys is None or Dataset.STEREO_BASELINE in get_keys:
                        data[Dataset.STEREO_BASELINE] = np.array(shot["stereo_baseline"])

                    if get_keys is None or Dataset.OBJ_CLASSES in get_keys:
                        data[Dataset.OBJ_CLASSES] = np.array([d["class"] for d in shot["objs"]])

                    if get_keys is None or Dataset.OBJ_IDS in get_keys:
                        data[Dataset.OBJ_IDS] = np.array([d["object id"] for d in shot["objs"]])

                    if get_keys is None or Dataset.OBJ_POS in get_keys:
                        data[Dataset.OBJ_POS] = np.array([d["pos"] for d in shot["objs"]])

                    if get_keys is None or Dataset.OBJ_ROT in get_keys:
                        data[Dataset.OBJ_ROT] = np.array([d["rotation"] for d in shot["objs"]])

                    if get_keys is None or Dataset.OBJ_BBOX_VISIB in get_keys:
                        data[Dataset.OBJ_BBOX_VISIB] = np.array(
                            [d["bbox_visib"] for d in shot["objs"]]
                        )

                    if get_keys is None or Dataset.OBJ_VISIB_FRACT in get_keys:
                        data[Dataset.OBJ_VISIB_FRACT] = np.array(
                            [d["visib_fract"] for d in shot["objs"]]
                        )

                    if get_keys is None or Dataset.OBJ_PX_COUNT_VISIB in get_keys:
                        data[Dataset.OBJ_PX_COUNT_VISIB] = np.array(
                            [d["px_count_visib"] for d in shot["objs"]]
                        )

                    if get_keys is None or Dataset.OBJ_PX_COUNT_VALID in get_keys:
                        data[Dataset.OBJ_PX_COUNT_VALID] = np.array(
                            [d["px_count_valid"] for d in shot["objs"]]
                        )

                    if get_keys is None or Dataset.OBJ_PX_COUNT_ALL in get_keys:
                        data[Dataset.OBJ_PX_COUNT_ALL] = np.array(
                            [d["px_count_all"] for d in shot["objs"]]
                        )

                    if get_keys is None or Dataset.OBJ_BBOX_OBJ in get_keys:
                        data[Dataset.OBJ_BBOX_OBJ] = np.array(
                            [d["bbox_obj"] for d in shot["objs"]]
                        )

                if get_keys is None or Dataset.RGB in get_keys:
                    data[Dataset.RGB] = np.array(
                        Image.open(str(root_dir / "rgb" / f"rgb_{idx:04}.png"))
                    )

                if get_keys is None or Dataset.RGB_R in get_keys:
                    try:
                        bgr_R = np.array(Image.open(str(root_dir / "rgb" / f"rgb_{idx:04}_R.png")))
                        data[Dataset.RGB_R] = bgr_R
                    except Exception:
                        bgr_R = None

                if get_keys is None or Dataset.DEPTH in get_keys:
                    depth = np.array(
                        cv2.imread(
                            str(root_dir / "depth" / f"depth_{idx:04}.exr"),
                            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                        )
                    ).astype(np.float32)
                    data[Dataset.DEPTH] = depth

                if get_keys is None or Dataset.DEPTH_R in get_keys:
                    depth_R = np.array(
                        cv2.imread(
                            str(root_dir / "depth" / f"depth_{idx:04}_R.exr"),
                            cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                        )
                    ).astype(np.float32)
                    data[Dataset.DEPTH_R] = depth_R

                if get_keys is None or Dataset.MASK in get_keys:
                    mask_path = root_dir.joinpath(f"mask/mask_{idx:04}.exr")
                    mask = sp.EXR(mask_path).read("visib.R").astype(np.uint8)
                    data[Dataset.MASK] = mask

                yield data

        signature = {}

        data = next(generator())

        signature = {
            k: tf.TensorSpec(shape=v.shape, dtype=v.dtype, name=k) for k, v in data.items()
        }

        return tf.data.Dataset.from_generator(generator, output_signature=signature)
