import simpose as sp
from .generator import Generator, GeneratorParams

import multiprocessing as mp
from pathlib import Path
import logging
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import random


class FlyingObjectsConfig(GeneratorParams):
    time_step: float = 0.25
    num_time_steps: int = 10

    friction: float = 0.8
    use_stereo: bool = True
    cam_hfov: float = 70
    cam_baseline: float = 0.063
    img_w: int = 1920
    img_h: int = 1080

    num_distractors: int = 20

    main_obj_path: Path = Path("path/to/model.obj")
    num_main_objs: int = 10
    scale: float = 1.0
    metallic: float = 0.0
    roughness: float = 0.7
    hue: float = 0.5
    saturation: float = 1.0
    value: float = 1.0

    @classmethod
    def get_description(cls):
        desc = super().get_description()
        desc.update(
            {
                "drop_height": "Height from which objects are dropped",
                "time_step": "Physics time step",
                "num_time_steps": "Number of physics time steps",
                "friction": "Friction of objects",
                "use_stereo": "Use stereo camera",
                "cam_hfov": "Camera horizontal field of view",
                "cam_baseline": "Camera baseline",
                "img_w": "Image width",
                "img_h": "Image height",
                "num_distractors": "Number of distractor objects",
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
        return desc


def indent(text: str) -> str:
    pad = "  "
    return "\n".join(pad + line for line in text.split("\n"))


def entry(name: str, type: str, params: str):
    spec = indent(f"type: {type}\nparams:\n{indent(params)}")
    return f"{name}:\n{spec}"


class FlyingObjects(Generator):
    params: FlyingObjectsConfig
    appearance_randomizer: sp.random.AppearanceRandomizer

    @staticmethod
    def generate_template_config() -> str:
        gen_params = FlyingObjectsConfig.dump_with_comments(n_workers=1)
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

    def generate_data(self, indices: list[int]):
        p = self.params
        assert p.num_main_objs > 0, "num_main_objs must be > 0"

        proc_name = mp.current_process().name
        is_primary_worker = proc_name == "Process-1" or proc_name == "MainProcess"

        debug = is_primary_worker and sp.logger.level < logging.DEBUG

        # -- SCENE --
        self.scene = scene = sp.Scene.create(img_h=p.img_h, img_w=p.img_w, debug=debug)

        # -- CAMERA --
        if p.use_stereo:
            cam = scene.create_stereo_camera("Camera", baseline=p.cam_baseline)
        else:
            cam = scene.create_camera("Camera")
        cam.set_from_hfov(p.cam_hfov, scene.resolution_x, scene.resolution_y, degrees=True)

        # -- RANDOMIZERS --
        for _, randomizer in self.randomizers.items():
            randomizer.listen_to(scene)

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
            scene.export_meshes(self.writer.output_dir / "meshes")

        main_objs = [main_obj]

        for i in range(p.num_main_objs - 1):
            main_objs.append(scene.create_copy(main_obj))

        for obj in main_objs:
            obj.hide()

        # TODO BUILD FRUSTUM BOX WITH OBJECTS
        # TODO set gravity to 0
        # TODO object velocity
        cam.set_location((0, 0, 0))
        cam.set_rotation(R.from_euler("x", -90, degrees=True))

        # --- Generation params ---
        i = 0
        bar = tqdm(
            total=len(indices), desc="Process-1", smoothing=0.0, disable=not is_primary_worker
        )
        while True:
            self.setup_new_scene(main_objs)
            for _ in range(p.num_time_steps):
                scene.step_physics(p.time_step)
                self.writer.write_data(scene, indices[i])
                i += 1
                if i == len(indices):
                    bar.close()
                    if debug:
                        scene.export_blend()
                    return scene
                bar.update(1)

    def setup_new_scene(self, main_objs: list[sp.Object]):
        model_loader: sp.random.ModelLoader = self.randomizers["distractors"]  # type: ignore
        p = self.params

        model_loader.reset()
        distractors = model_loader.get_objects(
            self.scene, p.num_distractors, mass=0.2, friction=p.friction
        )

        drop_objects = main_objs + distractors
        random.shuffle(drop_objects)

        # TODO camerafrustum placer?
        # TODO give nudge to make them float
