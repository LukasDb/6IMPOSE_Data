import simpose
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

from .randomizer import Randomizer, RandomizerConfig


class AppearanceRandomizerConfig(RandomizerConfig):
    metallic_range: float = 0.25  # standard deviation 68% <, 95% <<, 99.7% <<<
    roughness_range: float = 0.25
    hue_range: float = 0.04
    saturation_range: float = 0.24
    value_range: float = 0.24

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "metallic_range": "standard deviation of the metallic value",
            "roughness_range": "standard deviation of the roughness value",
            "hue_range": "standard deviation of the hue value",
            "saturation_range": "standard deviation of the saturation value",
            "value_range": "standard deviation of the value value",
        }


class AppearanceRandomizer(Randomizer):
    def __init__(
        self,
        params: AppearanceRandomizerConfig,
    ):
        super().__init__(params)

        self._subjects: set[simpose.Object] = set()
        app = simpose.Object.ObjectAppearance
        self._ranges = {
            app.METALLIC: params.metallic_range,
            app.ROUGHNESS: params.roughness_range,
            app.HUE: params.hue_range,
            app.SATURATION: params.saturation_range,
            app.VALUE: params.value_range,
        }

    def add(self, object: simpose.Object):
        self._subjects.add(object)

    def call(self, _: simpose.Scene):
        for obj in list(self._subjects)[:]:
            for appearance in simpose.Object.ObjectAppearance:
                try:
                    default = obj.get_default_appearance(appearance)
                except ReferenceError:
                    self._subjects.remove(obj)
                    break
                r = self._ranges[appearance]
                random_value = np.random.uniform(default - r, default + r)
                obj.set_appearance(appearance, random_value, set_default=False)
