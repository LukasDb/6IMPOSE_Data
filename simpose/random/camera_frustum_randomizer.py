import simpose
import numpy as np
from typing import List
import logging
from scipy.spatial.transform import Rotation as R

from .randomizer import Randomizer, RandomizerConfig


class CameraFrustumRandomizerConfig(RandomizerConfig):
    r_range: tuple = (0.3, 1.2)
    yp_limit: tuple = (0.9, 0.9)

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "r_range": "Range of the distance from the camera",
            "yp_limit": "Limit horizontal and vertical position in FOV of camera",
        }


class CameraFrustumRandomizer(Randomizer):
    def __init__(
        self,
        params: CameraFrustumRandomizerConfig,
    ) -> None:
        super().__init__(params)
        self._r_range = params.r_range
        self._yp_limit = params.yp_limit
        self._subjects: List[simpose.Object] = []

    def add_object(self, object):
        self._subjects.append(object)

    def set_to_camera(self, camera: simpose.Camera):
        self._cam = camera

    def call(self, scene: simpose.Scene):
        aspect_ratio = scene.resolution[0] / scene.resolution[1]
        cam_data = self._cam.data
        hfov = cam_data.angle_x / 2.0
        vfov = cam_data.angle_x / 2.0 / aspect_ratio
        # min_fov = min(hfov, vfov)

        r_range = self._r_range
        yp_limit = self._yp_limit

        cam_origin = self._cam.location
        cam_rot = self._cam.rotation

        self._randomize_orientation()

        for subject in self._subjects:
            r = np.random.uniform(r_range[0], r_range[1])
            yaw = yp_limit[0] * np.random.uniform(-hfov, hfov)
            pitch = yp_limit[1] * np.random.uniform(-vfov, vfov)

            x = np.cos(pitch) * np.sin(yaw) * r
            y = np.sin(pitch) * r
            z = np.cos(pitch) * np.cos(yaw) * r

            pos = np.array([x, y, z]) @ cam_rot.as_matrix() + np.array(cam_origin)
            subject.set_location(pos)

            simpose.logger.debug(f"randomize_in_camera_frustum: {subject} randomized to {pos}")

    def _randomize_orientation(self):
        for subject in self._subjects:
            subject.set_rotation(R.random())
