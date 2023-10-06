import simpose
import itertools as it
from pathlib import Path
import numpy as np
import logging
import random
from enum import Enum, auto
from .randomizer import JoinableRandomizer, RandomizerConfig


class ModelSource(Enum):
    YCB = "YCB"
    SHAPENET = "SHAPENET"
    SYNTHDET = "SYNTHDET"
    GENERIC_OBJ = "obj"
    GENERIC_PLY = "ply"
    GENERIC_GLTF = "gltf"
    GENERIC_FBX = "fbx"


class ModelLoaderConfig(RandomizerConfig):
    root: Path = Path("path/to/models")
    source: ModelSource = ModelSource.GENERIC_OBJ
    scale_range: tuple[float, float] = (0.5, 2)
    exclude: list[str] = []

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "root": "Path to the root directory of the models",
            "source": "Type of the dataset source",
            "scale_range": "Range of the scale of the models",
            "exclude": "List of file names to exclude",
        }


class ModelLoader(JoinableRandomizer):
    """loads shapenet objects as distractors into scene"""

    def __init__(self, params: ModelLoaderConfig):
        super().__init__(params)
        # find .obj files in shapenet root
        # because folder structure at root defines type of object and not all types are equally represented choose type first, then random object
        self._root = root = params.root.expanduser()
        self._scale_range = scale_range = params.scale_range
        self._additional_loaders: list[ModelLoader] = []
        self._objects: list[simpose.Object] = []

        model_source = params.source

        if model_source == ModelSource.SHAPENET:
            raise NotImplementedError("Shapenet objects behave incorrectly at the moment.")
            shapenet_contents = self._root.iterdir()
            _shapenet_types = list([x for x in shapenet_contents if x.is_dir()])
            shapenet_objs = list(it.chain.from_iterable([x.iterdir() for x in _shapenet_types]))
            model_paths = [x / "models/model_normalized.gltf" for x in shapenet_objs if x.is_dir()]
            for model_path in model_paths:
                if not model_path.exists():
                    simpose.logger.warning(f"Model path {model_path} does not exist.")
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
                    if not "_vhacd.obj" in x.name and not x.name in params.exclude
                ]
            )
        else:
            raise NotImplementedError(f"ModelSource {model_source} not implemented.")
        self._model_paths = model_paths

        self.num_models = len(self._model_paths)

        simpose.logger.debug(f"Found {len(self._model_paths)} models ({model_source}).")

    def get_objects(
        self, scene: simpose.Scene, num_objects: int, **kwargs
    ) -> list[simpose.Object]:
        """renews the list of objects and returns it"""
        for _ in range(num_objects):
            self.get_object(scene, **kwargs)
        return self._objects

    def get_object(self, scene: simpose.Scene, **kwargs) -> simpose.Object:
        if len(self._additional_loaders) > 1:
            # i = np.random.randint(0, len(self._additional_loaders))
            # loader = self._additional_loaders[i]
            loader = random.choices(
                self._additional_loaders, weights=[x.num_models for x in self._additional_loaders]
            )[0]
        else:
            loader = self
        return loader._get_object(scene, **kwargs)

    def _get_object(self, scene: simpose.Scene, **kwargs) -> simpose.Object:
        i = np.random.randint(0, len(self._model_paths))
        model_path = self._model_paths[i]
        obj = scene.create_object(
            model_path,
            add_semantics=False,
            scale=np.random.uniform(*self._scale_range),
            **kwargs,
        )
        self._objects.append(obj)
        simpose.logger.debug(f"Added object: {model_path}")
        return obj

    def reset(self):
        for obj in self._objects:
            obj.remove()
        self._objects.clear()

        for other in self._additional_loaders:
            other.reset()

    def call(self, _: simpose.Scene):
        raise NotImplementedError

    def __add__(self, other: "ModelLoader"):
        self._additional_loaders.append(other)
        return self
