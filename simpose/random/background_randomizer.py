import simpose
import numpy as np
from pathlib import Path
from typing import List
import logging


class BackgroundRandomizer(simpose.Callback):
    def __init__(
        self,
        scene: simpose.Scene,
        cb_type: simpose.CallbackType,
        *,
        backgrounds_dir: Path = Path("backgrounds"),
    ) -> None:
        super().__init__(scene, cb_type)
        self._scene = scene
        self._backgrounds_dir: Path = backgrounds_dir
        self._backgrounds: List = list(self._backgrounds_dir.glob("*.jpg"))
        logging.getLogger("simpose").debug(f"Loaded {len(self._backgrounds)} backgrounds from {self._backgrounds_dir}")

    def callback(self):
        bg = np.random.choice(self._backgrounds)
        self._scene.set_background(bg)
