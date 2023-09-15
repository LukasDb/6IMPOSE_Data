import simpose
import numpy as np
from typing import List
import logging
from scipy.spatial.transform import Rotation as R


class CameraFrustumRandomizer(simpose.Callback):
    def __init__(
        self,
        scene: simpose.Scene,
        cam: simpose.Camera,
        cb_type: simpose.CallbackType,
        *,
        r_range,
        yp_limit=(0.9, 0.9),
    ) -> None:
        super().__init__(scene, cb_type)
        self._scene = scene
        self._cam = cam
        self._r_range = r_range
        self._yp_limit = yp_limit
        self._subjects: List[simpose.Object] = []

    def add(self, object):
        self._subjects.append(object)

    def callback(self):
        aspect_ratio = self._scene.resolution[0] / self._scene.resolution[1]
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

            logging.getLogger(__name__).debug(
                f"randomize_in_camera_frustum: {subject} randomized to {pos}"
            )

    def _randomize_orientation(self):
        for subject in self._subjects:
            subject.set_rotation(R.random())

    # def randomize_appearance(self, metallic_range: Tuple[float,float],roughness_range: Tuple[float,float]):
    #     for subject in self._subjects:
    #         metallic_value = np.random.uniform(metallic_range[0],metallic_range[1])
    #         subject.set_metallic_value(metallic_value)
    #         roughness_value = np.random.uniform(roughness_range[0],roughness_range[1])
    #         subject.set_roughness_value(roughness_value)

    # def randomize_geometry(self, scale_range: Tuple[float,float], rotation_range: Tuple[float,float]):
    #     for subject in self._subjects:
    #         obj = subject._bl_object

    #         # Access the object's mesh data
    #         mesh = obj.data

    #         # Apply random scale to the vertices
    #         scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    #         for vertex in mesh.vertices:
    #             vertex.co *= scale_factor

    #         # Apply random rotation to the object
    #         rotation_angle = np.random.uniform(rotation_range[0], rotation_range[1])
    #         obj.rotation_euler.z += rotation_angle

    #         # Update the object with the modified mesh data
    #         obj.data.update()
