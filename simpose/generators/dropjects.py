import simpose as sp
from .generator import Generator, GeneratorParams

import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


class RandomImagePickerConfig(sp.random.RandomizerConfig):
    img_dir: Path

    @classmethod
    def get_description(cls) -> dict[str, str]:
        description = super().get_description()
        return description


class RandomImagePicker(sp.random.Randomizer):
    def __init__(self, params: RandomImagePickerConfig) -> None:
        super().__init__(params)
        self._img_paths = [
            x for x in params.img_dir.expanduser().iterdir() if "norm" not in x.name
        ]

    def randomize_plane(self, plane: sp.entities.Plane) -> None:
        self._plane = plane

    def call(self, _: sp.observers.Observable) -> None:
        i = np.random.randint(0, len(self._img_paths))
        img_path = self._img_paths[i]
        sp.logger.debug(f"Set plane texture to {img_path.name}")
        self._plane.set_image(img_path)


class DropjectsConfig(GeneratorParams):
    n_objects_per_scene: int = 50
    n_images_per_scene: int = 50
    drop_height: float = 1.0
    drop_spread: float = 0.4
    friction: float = 0.9
    restitution: float = 0.1
    use_stereo: bool = True
    cam_hfov: float = 70
    cam_baseline: float = 0.063
    img_w: int = 1920
    img_h: int = 1080
    floor_textures_dir: Path = Path("path/to/floor/textures")

    @classmethod
    def get_description(cls) -> dict[str, str]:
        description = super().get_description()
        description.update(
            {
                "drop_height": "Height from which objects are dropped",
                "drop_spread": "Spread of objects when dropped (distance from origin in XY)",
                "friction": "Friction of objects",
                "use_stereo": "Use stereo camera",
                "cam_hfov": "Camera horizontal field of view",
                "cam_baseline": "Camera baseline",
                "cam_dist_range": "Camera distance range to origin",
                "img_w": "Image width",
                "img_h": "Image height",
                "floor_textures_dir": "Path to directory with textures for the floor",
            }
        )
        return description


def indent(text: str) -> str:
    pad = "  "
    return "\n".join(pad + line for line in text.split("\n"))


def entry(name: str, type: str, params: str) -> str:
    spec = indent(f"type: {type}\nparams:\n{indent(params)}")
    return f"{name}:\n{spec}"


class Dropjects(Generator):
    params: DropjectsConfig
    appearance_randomizer: sp.random.AppearanceRandomizer

    @staticmethod
    def generate_template_config() -> str:
        gen_params = DropjectsConfig.dump_with_comments(n_workers=1, n_parallel_on_gpu=1)
        writer_params = sp.writers.WriterConfig.dump_with_comments()

        app_params = sp.random.AppearanceRandomizerConfig.dump_with_comments(
            trigger=sp.observers.Event.BEFORE_RENDER
        )

        light_params = sp.random.LightRandomizerConfig.dump_with_comments(
            trigger=sp.observers.Event.BEFORE_RENDER
        )

        bg_params = sp.random.BackgroundRandomizerConfig.dump_with_comments(
            trigger=sp.observers.Event.BEFORE_RENDER
        )
        cam_loc_params = sp.random.CameraPlacementRandomizerConfig.dump_with_comments(
            trigger=sp.observers.Event.BEFORE_RENDER
        )

        ycb_params = sp.random.ModelLoaderConfig.dump_with_comments(
            root=Path("path/to/ycb/models"),
            trigger=sp.observers.Event.NONE,
            source=sp.random.ModelSource.YCB,
        )

        ugreal_params = sp.random.ModelLoaderConfig.dump_with_comments(
            root=Path("path/to/ugreal/models"),
            trigger=sp.observers.Event.NONE,
            source=sp.random.ModelSource.SYNTHDET,
        )
        omni3d_params = sp.random.ModelLoaderConfig.dump_with_comments(
            root=Path("path/to/omni3d/models"),
            trigger=sp.observers.Event.NONE,
            source=sp.random.ModelSource.OMNI3D,
        )

        ycb_entry = entry("ycb_loader", "ModelLoader", ycb_params)
        ugreal_entry = entry("ugreal_loader", "ModelLoader", ugreal_params)
        omni3d_entry = entry("omni_loader", "ModelLoader", omni3d_params)

        randomizers_entries = "\n".join(
            [
                entry("appearance", "AppearanceRandomizer", app_params),
                entry("light", "LightRandomizer", light_params),
                entry("background", "BackgroundRandomizer", bg_params),
                entry("camera_placement", "CameraPlacementRandomizer", cam_loc_params),
                entry("distractors", "Join", "\n".join((ycb_entry, ugreal_entry, omni3d_entry))),
            ]
        )

        output = "\n".join(
            [
                entry("Generator", "Dropjects", gen_params),
                entry("Writer", "TFRecordWriter", writer_params),
                "Randomizers:",
                indent(randomizers_entries),
            ]
        )

        return output

    @staticmethod
    def generate_data(
        config: DropjectsConfig,
        writer: sp.writers.Writer,
        randomizers: dict[str, sp.random.Randomizer],
        indices: np.ndarray,
    ) -> None:
        p = config

        # --- Generation params ---
        i = 0
        while True:
            scene = Dropjects.setup_new_scene(p, randomizers)
            scene.export_meshes(writer.output_dir / "meshes")

            for _ in range(p.n_images_per_scene):
                writer.write_data(indices[i], scene=scene)

                # scene.export_blend()
                # return

                i += 1
                if i == len(indices):
                    sp.logger.info(f"Finished generating data {mp.current_process().name}")
                    return

    @staticmethod
    def setup_new_scene(
        p: DropjectsConfig,
        randomizers: dict[str, sp.random.Randomizer],
    ) -> sp.Scene:

        # -- SCENE --
        scene = sp.Scene(img_h=p.img_h, img_w=p.img_w)
        plane = scene.create_plane(size=2.0)

        # -- CAMERA --
        if p.use_stereo:
            cam = scene.create_stereo_camera("Camera", baseline=p.cam_baseline)
        else:
            cam = scene.create_camera("Camera")
        cam.set_from_hfov(p.cam_hfov, scene.resolution_x, scene.resolution_y, degrees=True)

        # -- RANDOMIZERS --
        for randomizer in randomizers.values():
            randomizer.listen_to(scene)

        cfg = RandomImagePickerConfig(
            img_dir=p.floor_textures_dir, trigger=sp.observers.Event.BEFORE_RENDER
        )
        randimages = RandomImagePicker(cfg)
        randimages.listen_to(scene)
        randimages.randomize_plane(plane)

        model_loader: sp.random.ModelLoader = randomizers["distractors"]  # type: ignore
        model_loader.reset()

        n_classes = np.random.randint(1, p.n_objects_per_scene + 1)

        class_indices = np.random.randint(0, n_classes, size=p.n_objects_per_scene - n_classes)

        kwargs = dict(
            scene=scene, mass=0.2, friction=p.friction, hide=True, restitution=p.restitution
        )

        first_pick = model_loader.get_object(**kwargs)
        obj_classes = [first_pick]
        base_diameter = first_pick.get_diameter()

        for _ in range(n_classes - 1):
            obj = model_loader.get_object(**kwargs)
            ratio = np.abs(obj.get_diameter() - base_diameter) / base_diameter

            while ratio > 0.7:
                obj.remove()
                obj = model_loader.get_object(**kwargs)
                ratio = np.abs(obj.get_diameter() - base_diameter) / base_diameter

            obj_classes.append(obj)

        objs = [*obj_classes]
        for class_index in class_indices:
            obj = scene.create_copy(obj_classes[class_index])
            objs.append(obj)
            obj.hide()

        sp.logger.debug(
            f"Setting up scene with {n_classes} object classes (in total {len(objs)} objects)"
        )

        max_obj_diameter = max(np.max(o._bl_object.dimensions) for o in obj_classes)  # [8, 3]

        for obj in objs:
            heighest_point = max([o.location[2] for o in objs])

            obj.show()
            obj.set_location(
                (
                    np.random.uniform(-p.drop_spread, p.drop_spread),
                    np.random.uniform(-p.drop_spread, p.drop_spread),
                    heighest_point + 2 * max_obj_diameter,
                )
            )
            obj.set_rotation(R.random())
            # this takes forever! run with p.GUI and check simulation and scaling?
            scene.step_physics(0.4)

        return scene
