import simpose as sp
from simpose.random import Randomizer, RandomizerConfig
from simpose.random.light_randomizer import LightRandomizer
from .generator import Generator, GeneratorParams

import multiprocessing as mp
from pathlib import Path
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import random


class RandomImagerPickerConfig(RandomizerConfig):
    img_dir: Path

    @classmethod
    def get_description(cls):
        pass


class RandomImagePicker(Randomizer):
    def __init__(self, params: RandomImagerPickerConfig) -> None:
        super().__init__(params)
        self._img_paths = [
            x for x in params.img_dir.expanduser().iterdir() if "norm" not in x.name
        ]

    def randomize_plane(self, plane: sp.Plane):
        self._plane = plane

    def call(self, _: sp.Scene):
        i = np.random.randint(0, len(self._img_paths))
        img_path = self._img_paths[i]
        sp.logger.debug(f"Set plane texture to {img_path.name}")
        self._plane.set_image(img_path)


class DroppedObjectsConfig(GeneratorParams):
    drop_height: float = 1.0
    drop_spread: float = 0.4
    time_step: float = 0.25
    num_time_steps: int = 10
    num_camera_locations: int = 10
    friction: float = 0.8
    use_stereo: bool = True
    cam_hfov: float = 70
    cam_baseline: float = 0.063
    img_w: int = 1920
    img_h: int = 1080

    floor_textures_dir: Path = Path("path/to/floor/textures")

    num_primary_distractors: int = 40
    num_secondary_distractors: int = 5

    main_obj_path: Path = Path("path/to/model.obj")
    num_main_objs: int = 20
    scale: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.7
    hue: float = 0.5
    saturation: float = 1.0
    value: float = 1.0

    @classmethod
    def get_description(cls):
        description = super().get_description()
        description.update(
            {
                "drop_height": "Height from which objects are dropped",
                "drop_spread": "Spread of objects when dropped (distance from origin in XY)",
                "time_step": "Physics time step",
                "num_time_steps": "Number of physics time steps",
                "num_camera_locations": "Number of camera locations per timestep",
                "friction": "Friction of objects",
                "use_stereo": "Use stereo camera",
                "cam_hfov": "Camera horizontal field of view",
                "cam_baseline": "Camera baseline",
                "cam_dist_range": "Camera distance range to origin",
                "img_w": "Image width",
                "img_h": "Image height",
                "floor_textures_dir": "Path to directory with textures for the floor",
                "num_primary_distractors": "Number of distractors that are dropped first",
                "num_secondary_distractors": "Number of distractors that are dropped together with the main objects",
                "main_obj_path": "Path to main object .obj file",
                "num_main_objs": "Number of main objects",
                "scale": "Main object scale",
                "metallic": "Default Main object metallic value",
                "roughness": "Default Main object roughness value",
                "hue": "Default Main object hue value",
                "saturation": "Default Main object saturation value",
                "value": "Default Main object value value",
            }
        )
        return description


def indent(text: str) -> str:
    pad = "  "
    return "\n".join(pad + line for line in text.split("\n"))


def entry(name: str, type: str, params: str):
    spec = indent(f"type: {type}\nparams:\n{indent(params)}")
    return f"{name}:\n{spec}"


class DroppedObjects(Generator):
    params: DroppedObjectsConfig
    appearance_randomizer: sp.random.AppearanceRandomizer

    @staticmethod
    def generate_template_config() -> str:
        gen_params = DroppedObjectsConfig.dump_with_comments(n_workers=1, n_parallel_on_gpu=1)
        writer_params = sp.writers.WriterConfig.dump_with_comments()

        app_params = sp.random.AppearanceRandomizerConfig.dump_with_comments(
            trigger=sp.Event.BEFORE_RENDER
        )

        light_params = sp.random.LightRandomizerConfig.dump_with_comments(
            trigger=sp.Event.BEFORE_RENDER
        )

        bg_params = sp.random.BackgroundRandomizerConfig.dump_with_comments(
            trigger=sp.Event.BEFORE_RENDER
        )
        cam_loc_params = sp.random.CameraPlacementRandomizerConfig.dump_with_comments(
            trigger=sp.Event.BEFORE_RENDER
        )

        ycb_params = sp.random.ModelLoaderConfig.dump_with_comments(
            root=Path("path/to/ycb/models"),
            trigger=sp.Event.NONE,
            source=sp.random.ModelSource.YCB,
        )

        ugreal_params = sp.random.ModelLoaderConfig.dump_with_comments(
            root=Path("path/to/ugreal/models"),
            trigger=sp.Event.NONE,
            source=sp.random.ModelSource.SYNTHDET,
        )
        ycb_entry = entry("ycb_loader", "ModelLoader", ycb_params)
        ugreal_entry = entry("ugreal_loader", "ModelLoader", ugreal_params)

        randomizers_entries = "\n".join(
            [
                entry("appearance", "AppearanceRandomizer", app_params),
                entry("light", "LightRandomizer", light_params),
                entry("background", "BackgroundRandomizer", bg_params),
                entry("camera_placement", "CameraPlacementRandomizer", cam_loc_params),
                entry("distractors", "Join", "\n".join((ycb_entry, ugreal_entry))),
            ]
        )

        output = "\n".join(
            [
                entry("Generator", "DroppedObjects", gen_params),
                entry("Writer", "SimposeWriter", writer_params),
                "Randomizers:",
                indent(randomizers_entries),
            ]
        )

        return output

    @staticmethod
    def generate_data(
        config: DroppedObjectsConfig,
        writer: sp.writers.Writer,
        randomizers: dict[str, sp.random.Randomizer],
        indices: list[int],
    ):
        p = config
        assert p.num_main_objs > 0, "num_main_objs must be > 0"

        proc_name = mp.current_process().name
        is_primary_worker = (
            proc_name == "Process-1"
            or proc_name == "MainProcess"
            or proc_name == "ForkPoolWorker-2"
        )

        debug = is_primary_worker and sp.logger.level < logging.DEBUG

        # -- SCENE --
        scene = scene = sp.Scene.create(img_h=p.img_h, img_w=p.img_w, debug=debug)
        plane = scene.create_plane()

        # -- CAMERA --
        if p.use_stereo:
            cam = scene.create_stereo_camera("Camera", baseline=p.cam_baseline)
        else:
            cam = scene.create_camera("Camera")
        cam.set_from_hfov(p.cam_hfov, scene.resolution_x, scene.resolution_y, degrees=True)

        # -- RANDOMIZERS --
        for randomizer in randomizers.values():
            randomizer.reset_listening()
            randomizer.listen_to(scene)

        cfg = RandomImagerPickerConfig(
            img_dir=p.floor_textures_dir, trigger=sp.Event.BEFORE_RENDER
        )
        randimages = RandomImagePicker(cfg)
        randimages.listen_to(scene)
        randimages.randomize_plane(plane)

        # -- OBJECTS --
        main_obj = scene.create_object(
            p.main_obj_path,
            mass=0.2,
            friction=p.friction,
            add_semantics=True,
            scale=p.scale,
        )
        main_obj.set_metallic(p.metallic)
        main_obj.set_roughness(p.roughness)
        main_obj.set_hue(p.hue)
        main_obj.set_saturation(p.saturation)
        main_obj.set_value(p.value)

        if is_primary_worker:
            scene.export_meshes(writer.output_dir / "meshes")

        main_objs = [main_obj]

        for i in range(p.num_main_objs - 1):
            main_objs.append(scene.create_copy(main_obj))

        for obj in main_objs:
            obj.hide()

        # --- Generation params ---
        i = 0
        while True:
            DroppedObjects.setup_new_scene(p, scene, randomizers, main_objs)
            for _ in range(p.num_time_steps):
                scene.step_physics(p.time_step)

                for _ in range(p.num_camera_locations):
                    writer.write_data(scene, indices[i])

                    if i == 0 and debug:
                        scene.export_blend()

                    i += 1
                    if i == len(indices):
                        return scene


    @staticmethod
    def setup_new_scene(
        config: DroppedObjectsConfig,
        scene: sp.Scene,
        randomizers: dict[str, sp.random.Randomizer],
        main_objs: list[sp.Object],
    ):
        model_loader: sp.random.ModelLoader = randomizers["distractors"]  # type: ignore
        model_loader.reset()
        # add 20 objects and let fall
        for _ in range(config.num_primary_distractors):
            # for obj in model_loader.get_objects(
            #     self.scene, p.num_primary_distractors, mass=1, friction=p.friction, hide=True
            # ):
            # obj.show()
            obj = model_loader.get_object(scene, mass=0.2, friction=config.friction)
            obj.set_location(
                (
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.05, 0.05),
                    config.drop_height,
                )
            )
            obj.set_rotation(R.random())
            scene.step_physics(0.4)  # initial fall

        distractors = []
        for _ in range(config.num_secondary_distractors):
            obj = model_loader.get_object(scene, mass=0.2, friction=config.friction, hide=True)
            distractors.append(obj)

        drop_objects = main_objs + distractors
        random.shuffle(drop_objects)

        for obj in drop_objects:
            obj.show()
            obj.set_location(
                (
                    np.random.uniform(-config.drop_spread, config.drop_spread),
                    np.random.uniform(-config.drop_spread, config.drop_spread),
                    config.drop_height,
                )
            )
            obj.set_rotation(R.random())
            scene.step_physics(0.4)
