import simpose
import numpy as np
from typing import Tuple
import bpy
from scipy.spatial.transform import Rotation as R
import logging


class LightRandomizer(simpose.Callback):
    def __init__(
        self,
        scene: simpose.Scene,
        cam: simpose.Camera,
        cb_type: simpose.CallbackType,
        *,
        no_of_lights_range: Tuple[int, int],
        energy_range: Tuple[int, int],
        color_range: Tuple[float, float],
        distance_range: Tuple[float, float],
        size_range: Tuple[float, float],
    ):
        super().__init__(scene, cb_type)
        self._scene = scene
        self._cam = cam
        self._no_of_lights_range = no_of_lights_range
        self._energy_range = energy_range
        self._color_range = color_range
        self._distance_range = distance_range
        self._size_range = size_range

    def callback(self):
        """generates random point lights arond cam"""
        for key in bpy.data.lights:
            bpy.data.lights.remove(key, do_unlink=True)

        n_lights = np.random.randint(*self._no_of_lights_range)

        for i in range(n_lights):
            energy = np.random.uniform(*self._energy_range)
            light = self._scene.create_light(
                f"Light_{i}", type=simpose.Light.TYPE_AREA, energy=energy
            )
            pos = self._get_random_position_rel_to_camera(self._cam)
            light.set_location(pos)
            light.point_at(np.array([0.0, 0.0, 0.0]))
            light.color = np.random.uniform(*self._color_range, size=(3,))
            light.size = np.random.uniform(*self._size_range)

        logging.getLogger("simpose").debug(f"Created {n_lights} lights")

    def _get_random_position_rel_to_camera(self, cam: simpose.Camera):
        dist = np.random.uniform(*self._distance_range)
        dir = R.random().as_matrix() @ np.array([0, 0, 1])
        offset = dist * dir
        cam_pos = cam.location
        pos = offset + cam_pos

        return pos
