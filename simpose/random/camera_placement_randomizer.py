import simpose as sp
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import logging

from .randomizer import Randomizer, RandomizerConfig, register_operator


class CameraPlacementRandomizerConfig(RandomizerConfig):
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pitch_range: Tuple[float, float] = (10, 90)
    distance_range: Tuple[float, float] = (0.4, 1.2)
    roll_jitter: float = 20.0
    pitch_jitter: float = 5.0
    yaw_jitter: float = 5.0

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "origin": "Origin to place the camera around",
            "pitch_range": "Elevation of the camera to the origin (Z+ is up == 90°)",
            "distance_range": "Distance of the camera from the origin",
            "roll_jitter": "Maximum jitter applied to camera roll angle when looking at the origin",
            "pitch_jitter": "Maximum jitter applied to camera pitch angle when looking at the origin",
            "yaw_jitter": "Maximum jitter applied to camera yaw angle when looking at the origin",
        }


@register_operator(cls_params=CameraPlacementRandomizerConfig)
class CameraPlacementRandomizer(Randomizer):
    def __init__(self, params: CameraPlacementRandomizerConfig):
        super().__init__(params)
        self.params = params

    def call(self, scene: sp.Scene):
        p = self.params
        cam_view = np.array([0.0, 0.0, 1.0])
        radius = np.random.uniform(*p.distance_range)

        assert (p.pitch_range[1] - p.pitch_range[0]) >= 5, "Pitch range is too small"

        # rejection sample until found valid angle
        for _ in range(10000):
            rot = R.random()
            loc = rot.apply(cam_view) * radius + np.array(p.origin)
            d = np.linalg.norm(loc[:2])
            pitch = np.arctan2(loc[2], d) * 180 / np.pi
            if p.pitch_range[0] < pitch < p.pitch_range[1]:
                break

        cam = scene.get_cameras()[0]

        cam.set_location(loc)
        cam.point_at(np.array(p.origin))  # with z up

        cam.apply_local_rotation_offset(
            R.from_euler(
                "xyz",
                (
                    np.random.uniform(-p.pitch_jitter, p.pitch_jitter),
                    np.random.uniform(-p.yaw_jitter, p.yaw_jitter),
                    np.random.uniform(-p.roll_jitter, p.roll_jitter),
                ),
                degrees=True,
            )
        )
        sp.logger.debug(f"Placed camera at {loc}.")
