import simpose
import itertools as it
from pathlib import Path
import numpy as np
import logging
from enum import Enum, auto

logger = logging.getLogger(__name__)


class ModelSource(Enum):
    YCB = auto()
    SHAPENET = auto()
    GENERIC_OBJ = "obj"
    GENERIC_PLY = "ply"
    GENERIC_GLTF = "gltf"


class ModelLoader(simpose.Callback):
    """loads shapenet objects as distractors into scene"""

    def __init__(
        self,
        scene: simpose.Scene,
        cb_type: simpose.CallbackType,
        *,
        root: Path,
        num_objects: int,
        model_source: ModelSource = ModelSource.SHAPENET,
        scale_range: tuple[float, float] = (0.08, 0.3),
    ):
        super().__init__(scene, cb_type)
        self._scene = scene

        # find .obj files in shapenet root
        # because folder structure at root defines type of object and not all types are equally represented choose type first, then random object
        self._num_objects = num_objects
        self._root = root
        self._scale_range = scale_range
        self._objects: list[simpose.Object] = []

        if model_source == ModelSource.SHAPENET:
            raise NotImplementedError("Shapenet objects behave incorrectly at the moment.")
            shapenet_contents = self._root.iterdir()
            _shapenet_types = list([x for x in shapenet_contents if x.is_dir()])
            shapenet_objs = list(it.chain.from_iterable([x.iterdir() for x in _shapenet_types]))
            model_paths = [x / "models/model_normalized.gltf" for x in shapenet_objs if x.is_dir()]
            for model_path in model_paths:
                if not model_path.exists():
                    logger.warning(f"Model path {model_path} does not exist.")

        elif model_source == ModelSource.YCB:
            model_paths = list([x / "google_16k/textured.obj" for x in self._root.iterdir()])

        elif model_source in [
            ModelSource.GENERIC_OBJ,
            ModelSource.GENERIC_PLY,
            ModelSource.GENERIC_GLTF,
        ]:
            # will ignore generated _vhacd models, and converted objs for collision
            model_paths = list(
                [
                    x
                    for x in self._root.glob(f"**/*.{model_source.value}")
                    if not "_vhacd.obj" in x.name
                ]
            )
        else:
            raise NotImplementedError(f"ModelSource {model_source} not implemented.")
        self._model_paths = model_paths

        logger.debug(f"Found {len(self._model_paths)} models ({model_source}).")

    def get_objects(self, **kwargs) -> list[simpose.Object]:
        """renews the list of objects and returns it"""
        for obj in self._objects:
            obj.remove()
        self._objects.clear()

        for _ in range(self._num_objects):
            model_path = np.random.choice(self._model_paths, replace=False)  # type: ignore

            obj = self._scene.create_object(
                model_path,
                add_semantics=False,
                scale=np.random.uniform(*self._scale_range),
                **kwargs,
            )
            self._objects.append(obj)
            logger.debug(f"Added Shapenet object: {model_path}")

        return self._objects

    def callback(self):
        logger.debug(f"Added objects: ")
