import simpose
import numpy as np
from scipy.spatial.transform import Rotation as R
from simpose.observers import Observable
from .randomizer import Randomizer, RandomizerConfig, register_operator


class CameraFrustumRandomizerConfig(RandomizerConfig):
    r_range: tuple = (0.3, 1.2)
    yp_limit: tuple = (0.9, 0.9)

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "r_range": "Range of the distance from the camera",
            "yp_limit": "Limit horizontal and vertical position in FOV of camera",
        }


#@register_operator(cls_params=CameraFrustumRandomizerConfig)
class CameraFrustumRandomizer(Randomizer):
    def __init__(
        self,
        params: CameraFrustumRandomizerConfig,
    ) -> None:
        super().__init__(params)
        self._r_range = params.r_range
        self._yp_limit = params.yp_limit

    def call(self, caller: Observable) -> None:
        scene = caller
        assert isinstance(scene, simpose.Scene)
        aspect_ratio = scene.resolution[0] / scene.resolution[1]
        cam = scene.get_cameras()[0]
        cam_data = cam.data
        hfov = cam_data.angle_x / 2.0
        vfov = cam_data.angle_x / 2.0 / aspect_ratio
        # min_fov = min(hfov, vfov)

        r_range = self._r_range
        yp_limit = self._yp_limit

        cam_origin = cam.location
        cam_rot = cam.rotation

        for subject in scene.get_active_objects():
            r = np.random.uniform(r_range[0], r_range[1])
            yaw = yp_limit[0] * np.random.uniform(-hfov, hfov)
            pitch = yp_limit[1] * np.random.uniform(-vfov, vfov)

            x = np.cos(pitch) * np.sin(yaw) * r
            y = np.sin(pitch) * r
            z = np.cos(pitch) * np.cos(yaw) * r

            pos = np.array([x, y, z]) @ cam_rot.as_matrix() + np.array(cam_origin)
            subject.set_location(pos)
            subject.set_rotation(R.random())

            simpose.logger.debug(f"randomize_in_camera_frustum: {subject} randomized to {pos}")
