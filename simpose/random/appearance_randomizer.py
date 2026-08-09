import simpose as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

from .randomizer import Randomizer, RandomizerConfig, register_operator


class AppearanceRandomizerConfig(RandomizerConfig):
    # each value is sampled uniformly from [default - range, default + range]
    metallic_range: float = 0.25
    roughness_range: float = 0.25
    hue_range: float = 0.01
    saturation_range: float = 0.1
    value_range: float = 0.1

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "metallic_range": "uniform half-range around the default metallic value",
            "roughness_range": "uniform half-range around the default roughness value",
            "hue_range": "uniform half-range around the default hue value",
            "saturation_range": "uniform half-range around the default saturation value",
            "value_range": "uniform half-range around the default value value",
        }


#@register_operator(cls_params=AppearanceRandomizerConfig)
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
                r = self._ranges[appearance]
                try:
                    default = obj.get_default_appearance(appearance)
                    random_value = np.random.uniform(default - r, default + r)
                    obj.set_appearance(appearance, random_value, set_default=False)
                except ReferenceError:
                    break  # stale blender object: skip it, keep randomizing the others
