import simpose as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

from .randomizer import Randomizer, RandomizerConfig, register_operator


class AppearanceRandomizerConfig(RandomizerConfig):
    metallic_range: float = 0.25  # standard deviation 68% <, 95% <<, 99.7% <<<
    roughness_range: float = 0.25
    hue_range: float = 0.01
    saturation_range: float = 0.1
    value_range: float = 0.1

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "metallic_range": "standard deviation of the metallic value",
            "roughness_range": "standard deviation of the roughness value",
            "hue_range": "standard deviation of the hue value",
            "saturation_range": "standard deviation of the saturation value",
            "value_range": "standard deviation of the value value",
        }


@register_operator(cls_params=AppearanceRandomizerConfig)
class AppearanceRandomizer(Randomizer):
    def __init__(
        self,
        params: AppearanceRandomizerConfig,
    ):
        super().__init__(params)
        app = sp.entities.ObjectAppearance
        self._ranges = {
            app.METALLIC: params.metallic_range,
            app.ROUGHNESS: params.roughness_range,
            app.HUE: params.hue_range,
            app.SATURATION: params.saturation_range,
            app.VALUE: params.value_range,
        }

    def call(self, caller: sp.observers.Observable) -> None:
        scene = caller
        assert isinstance(scene, sp.Scene)
        objects = scene.get_active_objects()
        for obj in objects:
            for appearance in sp.entities.ObjectAppearance:
                try:
                    default = obj.get_default_appearance(appearance)
                except ReferenceError:
                    break
                r = self._ranges[appearance]
                random_value = np.random.uniform(default - r, default + r)
                obj.set_appearance(appearance, random_value, set_default=False)
