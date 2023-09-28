import simpose
import numpy as np
from pathlib import Path
from typing import List
import logging
from pydantic import validator

from .randomizer import Randomizer, RandomizerConfig


class BackgroundRandomizerConfig(RandomizerConfig):
    backgrounds_dir: Path

    @validator("backgrounds_dir")
    def validate_path(cls, v):
        return Path(v)


class BackgroundRandomizer(Randomizer):
    def __init__(
        self,
        params: BackgroundRandomizerConfig,
    ) -> None:
        super().__init__(params)
        self._backgrounds_dir: Path = params.backgrounds_dir.expanduser()
        self._backgrounds: List = list(self._backgrounds_dir.glob("*.jpg"))
        simpose.logger.debug(
            f"Loaded {len(self._backgrounds)} backgrounds from {self._backgrounds_dir}"
        )

    def call(self, scene: simpose.Scene):
        bg = np.random.choice(self._backgrounds)
        scene.set_background(bg)
