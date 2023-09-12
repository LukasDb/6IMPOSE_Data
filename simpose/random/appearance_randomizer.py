import simpose
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging


class AppearanceRandomizer(simpose.Callback):
    def __init__(
        self,
        scene: simpose.Scene,
        cb_type: simpose.CallbackType,
        *,
        metallic_range: float = 0.3,  # standard deviation 68% <, 95% <<, 99.7% <<<
        roughness_range: float = 0.3,
        hue_range: float = 0.04,
        saturation_range: float = 0.24,
        value_range: float = 0.24,
    ):
        super().__init__(scene, cb_type)
        self._scene = scene
        self._subjects: set[simpose.Object] = set()
        app = simpose.Object.ObjectAppearance
        self._ranges = {
            app.METALLIC: metallic_range,
            app.ROUGHNESS: roughness_range,
            app.HUE: hue_range,
            app.SATURATION: saturation_range,
            app.VALUE: value_range,
        }

    def add(self, object: simpose.Object):
        self._subjects.add(object)

    def callback(self):
        for obj in self._subjects:
            for appearance in simpose.Object.ObjectAppearance:
                default = obj.get_default_appearance(appearance)
                r = self._ranges[appearance]
                random_value = np.random.uniform(default - r, default + r)

                logging.debug(f"{obj}.{appearance}: %f", random_value)
                obj.set_appearance(appearance, random_value, set_default=False)
