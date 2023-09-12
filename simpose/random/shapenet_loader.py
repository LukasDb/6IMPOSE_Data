import simpose
from pathlib import Path
import numpy as np
from typing import Tuple, List
import logging


class ShapenetLoader(simpose.Callback):
    """loads shapenet objects as distractors into scene"""

    def __init__(
        self,
        scene: simpose.Scene,
        cb_type: simpose.CallbackType,
        *,
        shapenet_root: Path,
        num_objects: int,
        scale_range: Tuple[float, float] = (0.08, 0.2),
    ):
        super().__init__(scene, cb_type)
        self._scene = scene

        # find .obj files in shapenet root
        # because folder structure at root defines type of object and not all types are equally represented choose type first, then random object
        self._num_objects = num_objects
        self._shapenet_root = shapenet_root
        self._scale_range = scale_range

        # get a list of all folders in shapenet root
        shapenet_contents = self._shapenet_root.iterdir()
        self._shapenet_types = list([x for x in shapenet_contents if x.is_dir()])

        self._shapenet_objects: List[simpose.Object] = []

    def get_objects(self, **kwargs) -> List[simpose.Object]:
        """renews the list of objects and returns it"""

        for obj in self._shapenet_objects:
            obj.remove()
        self._shapenet_objects.clear()

        obj_types_indices = np.random.choice(
            np.arange(len(self._shapenet_types)), replace=True, size=self._num_objects
        )

        for obj_type_index in obj_types_indices:
            obj_type = self._shapenet_types[obj_type_index]
            objs = list(obj_type.iterdir())

            obj_path = np.random.choice(list(objs), replace=False)
            obj = self._scene.create_object(
                obj_path / "models/model_normalized.obj",
                add_semantics=False,
                scale=np.random.uniform(*self._scale_range),
                **kwargs,
            )
            self._shapenet_objects.append(obj)
            logging.debug(f"Added Shapenet object: {obj_path}")

        return self._shapenet_objects

    def callback(self):
        logging.debug(f"Added objects: ")
