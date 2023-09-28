import simpose
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import logging

from .randomizer import Randomizer, RandomizerConfig


class LightRandomizerConfig(RandomizerConfig):
    no_of_lights_range: Tuple[int, int]
    energy_range: Tuple[int, int]
    color_range: Tuple[float, float]
    distance_range: Tuple[float, float]
    size_range: Tuple[float, float]


class LightRandomizer(Randomizer):
    def __init__(self, params: LightRandomizerConfig):
        super().__init__(params)
        self._no_of_lights_range = params.no_of_lights_range
        self._energy_range = params.energy_range
        self._color_range = params.color_range
        self._distance_range = params.distance_range
        self._size_range = params.size_range
        self._lights: list[simpose.Light] = []

    def call(self, scene: simpose.Scene):
        """generates random point lights around origin"""
        for light in self._lights:
            light.remove()
        self._lights.clear()

        n_lights = np.random.randint(*self._no_of_lights_range)

        for i in range(n_lights):
            energy = np.random.uniform(*self._energy_range)
            light = scene.create_light(f"Light_{i}", type=simpose.Light.TYPE_AREA, energy=energy)
            dist = np.random.uniform(*self._distance_range)
            dir = R.random().as_matrix() @ np.array([0, 0, 1])
            pos = dist * dir

            light.set_location(pos)
            light.point_at(np.array([0.0, 0.0, 0.0]))
            light.color = np.random.uniform(*self._color_range, size=(3,))
            light.size = np.random.uniform(*self._size_range)

            self._lights.append(light)

        simpose.logger.debug(f"Created {n_lights} lights")
