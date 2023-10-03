import simpose
import numpy as np
from pathlib import Path

from .randomizer import Randomizer, RandomizerConfig, register_operator


class BackgroundRandomizerConfig(RandomizerConfig):
    backgrounds_dir: Path = Path("path/to/backgrounds_folder")

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "backgrounds_dir": "Path to the background directory",
        }


@register_operator(cls_params=BackgroundRandomizerConfig)
class BackgroundRandomizer(Randomizer):
    def __init__(
        self,
        params: BackgroundRandomizerConfig,
    ) -> None:
        super().__init__(params)
        self._backgrounds_dir: Path = params.backgrounds_dir.expanduser()
        self._backgrounds: list = list(self._backgrounds_dir.glob("*.jpg"))
        simpose.logger.debug(
            f"Loaded {len(self._backgrounds)} backgrounds from {self._backgrounds_dir}"
        )

    def call(self, scene: simpose.Scene):
        bg = np.random.choice(self._backgrounds)
        scene.set_background(bg)
