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
        metallic_scale: float = 0.3,
        roughness_scale: float = 0.3,
        hue_scale: float = 0.04,
        saturation_scale: float = 0.24,
        value_scale: float = 0.24,
    ):
        super().__init__(scene, cb_type)
        self._scene = scene
        self._subjects: set[simpose.Object] = set()
        app = simpose.Object.ObjectAppearance
        self._scales = {
            app.METALLIC: metallic_scale,
            app.ROUGHNESS: roughness_scale,
            app.HUE: hue_scale,
            app.SATURATION: saturation_scale,
            app.VALUE: value_scale,
        }

    def add(self, object: simpose.Object):
        self._subjects.add(object)

    def callback(self):
        for obj in self._subjects:
            for appearance in simpose.Object.ObjectAppearance:
                random_value = np.random.normal(
                    obj.get_appearance(appearance), self._scales[appearance]
                )
                logging.debug(f"{obj}.{appearance}: %f", random_value)
                obj.set_appearance(appearance, random_value, set_default=False)
