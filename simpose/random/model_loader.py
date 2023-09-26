import simpose
import itertools as it
from pathlib import Path
import numpy as np
import logging
from enum import Enum, auto

logger = logging.getLogger("simpose")


class ModelSource(Enum):
    YCB = auto()
    SHAPENET = auto()
    SYNTHDET = auto()
    GENERIC_OBJ = "obj"
    GENERIC_PLY = "ply"
    GENERIC_GLTF = "gltf"
    GENERIC_FBX = "fbx"


class ModelLoader(simpose.Callback):
    """loads shapenet objects as distractors into scene"""

    def __init__(
        self,
        scene: simpose.Scene,
        cb_type: simpose.CallbackType,
        *,
        root: Path,
        model_source: ModelSource = ModelSource.SHAPENET,
        scale_range: tuple[float, float] = (0.5, 2),
    ):
        super().__init__(scene, cb_type)
        self._scene = scene

        # find .obj files in shapenet root
        # because folder structure at root defines type of object and not all types are equally represented choose type first, then random object
        self._root = root
        self._scale_range = scale_range
        self._additional_loaders: list["ModelLoader"] = []
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
            # shapenet scale is in weird "normalized range", so that diagonal = 1
            # we scale to 0.2m which is more in the range of objects for robotic grasping
            to_scale = 0.2
            self._scale_range = (scale_range[0] * to_scale, scale_range[1] * to_scale)

        elif model_source == ModelSource.YCB:
            model_paths = list([x / "google_16k/textured.obj" for x in self._root.iterdir()])
            # scale is in m

        elif model_source == ModelSource.SYNTHDET:
            model_paths = list((root / "Models").glob("*.fbx"))
            # units are in INCHES! apply conversion factor to m to scale_range
            to_scale = 0.0254
            self._scale_range = (scale_range[0] * to_scale, scale_range[1] * to_scale)

        elif model_source in [
            ModelSource.GENERIC_OBJ,
            ModelSource.GENERIC_PLY,
            ModelSource.GENERIC_GLTF,
            ModelSource.GENERIC_FBX,
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

    def get_objects(self, num_objects: int, **kwargs) -> list[simpose.Object]:
        """renews the list of objects and returns it"""
        for _ in range(num_objects):
            self.get_object(**kwargs)
        return self._objects

    def get_object(self, **kwargs) -> simpose.Object:
        if len(self._additional_loaders) > 1:
            i = np.random.randint(0, len(self._additional_loaders))
            loader = self._additional_loaders[i]
        else:
            loader = self
        return loader._get_object(**kwargs)

    def _get_object(self, **kwargs) -> simpose.Object:
        i = np.random.randint(0, len(self._model_paths))
        model_path = self._model_paths[i]
        obj = self._scene.create_object(
            model_path,
            add_semantics=False,
            scale=np.random.uniform(*self._scale_range),
            **kwargs,
        )
        self._objects.append(obj)
        logger.debug(f"Added object: {model_path}")
        return obj

    def reset(self):
        for obj in self._objects:
            obj.remove()
        self._objects.clear()

        for other in self._additional_loaders:
            other.reset()

    def callback(self):
        raise NotImplementedError

    def __add__(self, other):
        self._additional_loaders.append(other)
        return self
