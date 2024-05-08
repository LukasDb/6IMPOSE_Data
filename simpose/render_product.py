from dataclasses import dataclass
import numpy as np


@dataclass
class ObjectAnnotation:
    cls: str | None = None
    object_id: int | None = None
    position: tuple[float, float, float] | None = None
    quat_xyzw: tuple[float, float, float, float] | None = None
    bbox_visib: tuple[int, int, int, int] | None = None
    bbox_obj: tuple[int, int, int, int] | None = None
    px_count_visib: int | None = None
    px_count_valid: int | None = None
    px_count_all: int | None = None
    visib_fract: float | None = None


@dataclass
class RenderProduct:
    rgb: None | np.ndarray = None
    rgb_R: None | np.ndarray = None
    depth: None | np.ndarray = None
    depth_R: None | np.ndarray = None
    depth_GT: None | np.ndarray = None
    depth_GT_R: None | np.ndarray = None
    mask: None | np.ndarray = None
    objs: list[ObjectAnnotation] | None = None
    intrinsics: None | np.ndarray = None
    cam_position: None | tuple[float, float, float] = None
    cam_quat_xyzw: None | np.ndarray = None
    stereo_baseline: None | float = None
