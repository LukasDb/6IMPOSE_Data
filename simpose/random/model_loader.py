import json
import simpose as sp
from .randomizer import Randomizer, RandomizerConfig, JoinableRandomizer
from pathlib import Path
import numpy as np
import random
from enum import Enum


class ModelSource(Enum):
    YCB = "YCB"
    SHAPENET = "SHAPENET"
    SYNTHDET = "SYNTHDET"
    OMNI3D = "OMNI3D"
    GENERIC_OBJ = "obj"
    GENERIC_PLY = "ply"
    GENERIC_GLTF = "gltf"
    GENERIC_FBX = "fbx"
    FILE = "file"


class ModelLoaderConfig(RandomizerConfig):
    root: Path = Path("path/to/models")
    source: ModelSource = ModelSource.GENERIC_OBJ
    scale_range: tuple[float, float] | float = 1.0  # (0.5, 2)
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
        self._objects: list[sp.Object] = []

        model_source = params.source

        if model_source == ModelSource.SHAPENET:
            raise NotImplementedError("Shapenet objects behave incorrectly at the moment.")

        elif model_source == ModelSource.YCB:
            path_and_name = list(
                [(x / "google_16k/textured.obj", x.name) for x in self._root.iterdir()]
            )
            # scale is in m

        elif model_source == ModelSource.SYNTHDET:
            path_and_name = list([(x, x.name) for x in root.joinpath("Models").glob("*.fbx")])
            # units are in INCHES! apply conversion factor to m to scale_range
            to_scale = 0.0254
            if isinstance(scale_range, tuple):
                self._scale_range = (scale_range[0] * to_scale, scale_range[1] * to_scale)
            else:
                self._scale_range = scale_range * to_scale

        elif model_source == ModelSource.OMNI3D:
            path_and_name = list(
                [
                    (x / "Scan/simplified.obj", x.name)
                    for x in self._root.joinpath("models").iterdir()
                ]
            )
            to_scale = 0.001
            if isinstance(scale_range, tuple):
                self._scale_range = (scale_range[0] * to_scale, scale_range[1] * to_scale)
            else:
                self._scale_range = scale_range * to_scale

        elif model_source in [
            ModelSource.GENERIC_OBJ,
            ModelSource.GENERIC_PLY,
            ModelSource.GENERIC_GLTF,
            ModelSource.GENERIC_FBX,
        ]:
            # will ignore generated _vhacd models, and converted objs for collision
            path_and_name = list(
                [
                    (x, x.name)
                    for x in self._root.glob(f"**/*.{model_source.value}")
                    if not "_vhacd.obj" in x.name
                    and not "_collision.obj" in x.name
                    and not x.name in params.exclude
                ]
            )
        elif model_source == ModelSource.FILE:
            with self._root.joinpath("objects.json") as f:
                name_to_path = json.load(f)
                # list[str, str] where first is name and second is path
            path_and_name = [(self._root.joinpath(Path(path)), name) for path, name in name_to_path.items()]

        else:
            raise NotImplementedError(f"ModelSource {model_source} not implemented.")
        self._model_paths = path_and_name

        self.num_models = len(self._model_paths)

        sp.logger.debug(f"Found {len(self._model_paths)} models ({model_source}).")

    def get_objects(
        self,
        scene: sp.Scene,
        num_objects: int,
        mass: float,
        friction: float,
        hide: bool,
        restitution: float,
    ) -> list[sp.entities.Object]:
        """renews the list of objects and returns it"""
        return [
            self.get_object(scene, mass, friction, hide, restitution) for _ in range(num_objects)
        ]

    def get_object(
        self, scene: sp.Scene, mass: float, friction: float, hide: bool, restitution: float
    ) -> sp.entities.Object:
        if len(self._additional_loaders) > 1:
            # i = np.random.randint(0, len(self._additional_loaders))
            # loader = self._additional_loaders[i]
            loader = random.choices(
                self._additional_loaders, weights=[x.num_models for x in self._additional_loaders]
            )[0]
        else:
            loader = self
        return loader._get_object(scene, mass, friction, hide, restitution)

    def _get_object(
        self, scene: sp.Scene, mass: float, friction: float, hide: bool, restitution: float
    ) -> sp.entities.Object:
        i = np.random.randint(0, len(self._model_paths))
        model_path, obj_name = self._model_paths[i]
        scale = (
            np.random.uniform(*self._scale_range)
            if isinstance(self._scale_range, tuple)
            else self._scale_range
        )
        obj = scene.create_object(
            model_path,
            obj_name=obj_name,
            add_semantics=False,
            scale=scale,
            mass=mass,
            friction=friction,
            hide=hide,
            restitution=restitution,
        )
        self._objects.append(obj)
        return obj

    def reset(self) -> None:
        for obj in self._objects:
            obj.remove()
        self._objects.clear()

        for other in self._additional_loaders:
            other.reset()

    def call(self, _: sp.observers.Observable) -> None:
        raise NotImplementedError

    def __add__(self, other: Randomizer) -> "ModelLoader":
        assert isinstance(other, ModelLoader)
        self._additional_loaders.append(other)
        return self
