from abc import ABC, abstractmethod
from pathlib import Path
import tensorflow as tf


class Dataset(ABC):
    # name of available keys
    RGB = "rgb"
    RGB_R = "rgb_R"
    DEPTH = "depth"
    DEPTH_R = "depth_R"
    MASK = "mask"
    GT = "gt"
    CAM_MATRIX = "cam_matrix"
    CAM_LOCATION = "cam_location"
    CAM_ROTATION = "cam_rotation"
    STEREO_BASELINE = "stereo_baseline"
    OBJ_CLASSES = "obj_classes"
    OBJ_IDS = "obj_ids"
    OBJ_POS = "obj_pos"
    OBJ_ROT = "obj_rot"
    OBJ_BBOX_VISIB = "obj_bbox_visib"
    OBJ_VISIB_FRACT = "obj_visib_fract"
    OBJ_PX_COUNT_VISIB = "obj_px_count_visib"
    OBJ_PX_COUNT_VALID = "obj_px_count_valid"
    OBJ_PX_COUNT_ALL = "obj_px_count_all"
    OBJ_BBOX_OBJ = "obj_bbox_obj"

    @staticmethod
    @abstractmethod
    def get(root_dir: Path, get_keys: None | list[str] = None) -> tf.data.Dataset:
        """create a tf.data.Dataset from 6IMPOSE dataset. Returns a dict with the specified keys"""
        pass
