import simpose as sp
from simpose.random import Randomizer, RandomizerConfig
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


class RandomImagePicker(Randomizer):
    def __init__(self, params: RandomImagerPickerConfig) -> None:
        super().__init__(params)
        self._img_paths = [x for x in params.img_dir.iterdir() if "norm" not in x.name]

    def randomize_plane(self, plane: sp.Plane):
        self._plane = plane

    def call(self, _: sp.Scene):
        i = np.random.randint(0, len(self._img_paths))
        img_path = self._img_paths[i]
        sp.logger.debug(f"Set plane texture to {img_path.name}")
        self._plane.set_image(img_path)


class DroppedObjectsConfig(GeneratorParams):
    drop_height: float
    time_step: float
    num_time_steps: int
    num_camera_locations: int
    friction: float
    use_stereo: bool
    cam_hfov: float
    cam_baseline: float
    cam_dist_range: tuple[float, float]
    img_w: int
    img_h: int

    num_primary_distractors: int
    num_secondary_distractors: int

    main_obj_path: Path
    num_main_objs: int
    scale: float
    metallic: float
    roughness: float
    hue: float
    saturation: float
    value: float


class DroppedObjects(Generator):
    params: DroppedObjectsConfig
    appearance_randomizer: sp.random.AppearanceRandomizer

    def generate_data(self, indices: list[int]):
        p = self.params
        assert p.num_main_objs > 0, "num_main_objs must be > 0"

        proc_name = mp.current_process().name
        is_primary_worker = proc_name == "Process-1" or proc_name == "MainProcess"

        debug = is_primary_worker and sp.logger.level < logging.DEBUG

        # -- SCENE --
        self.scene = scene = sp.Scene(img_h=p.img_h, img_w=p.img_w, debug=debug)
        plane = scene.create_plane()

        # -- CAMERA --
        if p.use_stereo:
            cam = scene.create_stereo_camera("Camera", baseline=p.cam_baseline)
        else:
            cam = scene.create_camera("Camera")
        cam.set_from_hfov(p.cam_hfov, scene.resolution_x, scene.resolution_y, degrees=True)

        # -- RANDOMIZERS --
        for _, randomizer in self.randomizers.items():
            randomizer.listen_to(scene)
        cfg = RandomImagerPickerConfig(
            img_dir=Path("~/Pictures/textures").expanduser(), trigger=sp.Event.BEFORE_RENDER
        )
        randimages = RandomImagePicker(cfg)
        randimages.listen_to(scene)
        randimages.randomize_plane(plane)

        appearance_randomizer: sp.random.AppearanceRandomizer = self.randomizers["appearance"]  # type: ignore
        self.appearance_randomizer = appearance_randomizer
        self.appearance_randomizer.add(plane)

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
            appearance_randomizer.add(obj)
            obj.hide()

        # --- Generation params ---
        i = 0
        bar = tqdm(
            total=len(indices), desc="Process-1", smoothing=0.0, disable=not is_primary_worker
        )
        while True:
            self.setup_new_scene(main_objs)
            for _ in range(p.num_time_steps):
                scene.step_physics(p.time_step)

                cam_locations = self.get_camera_locations(p.num_camera_locations)

                for cam_location in cam_locations:
                    cam.set_location(cam_location)
                    cam.point_at(np.array([0.0, 0.0, 0.0]))  # with z up

                    cam.apply_local_rotation_offset(
                        R.from_euler("z", np.random.uniform(-20, 20), degrees=True)
                    )

                    self.writer.write_data(scene, indices[i])

                    i += 1
                    if i == len(indices):
                        bar.close()
                        if debug:
                            scene.export_blend()
                        return scene

                    bar.update(1)

    def get_camera_locations(self, num_cam_locs: int):
        p = self.params
        rots = R.random(num=num_cam_locs)
        cam_view = np.array([0.0, 0.0, 1.0])
        radius = np.random.uniform(*p.cam_dist_range, size=(num_cam_locs,))

        cam_locations = rots.apply(cam_view) * radius[:, None]
        cam_locations[:, 2] *= np.sign(cam_locations[:, 2])  # flip lower hemisphere up
        cam_locations[:, 2] += 0.2  # lift up a bit
        return cam_locations

    def setup_new_scene(self, main_objs: list[sp.Object]):
        model_loader: sp.random.ModelLoader = self.randomizers["distractors"]  # type: ignore
        p = self.params

        model_loader.reset()
        # add 20 objects and let fall
        for _ in range(p.num_primary_distractors):
            # for obj in model_loader.get_objects(
            #     self.scene, p.num_primary_distractors, mass=1, friction=p.friction, hide=True
            # ):
            # obj.show()
            obj = model_loader.get_object(self.scene, mass=0.2, friction=p.friction)
            self.appearance_randomizer.add(obj)
            obj.set_location(
                (
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.05, 0.05),
                    p.drop_height,
                )
            )
            obj.set_rotation(R.random())
            self.scene.step_physics(0.4)  # initial fall

        # mix main_objects and a few distractors
        # distractors = model_loader.get_objects(
        #    self.scene, p.num_secondary_distractors, mass=0.2, friction=p.friction, hide=True
        # )
        # for obj in distractors:
        #     self.appearance_randomizer.add(obj)

        distractors = []
        for _ in range(p.num_secondary_distractors):
            obj = model_loader.get_object(self.scene, mass=0.2, friction=p.friction, hide=True)
            self.appearance_randomizer.add(obj)
            distractors.append(obj)

        drop_objects = main_objs + distractors
        random.shuffle(drop_objects)

        for obj in drop_objects:
            obj.show()
            obj.set_location(
                (
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.05, 0.05),
                    p.drop_height,
                )
            )
            obj.set_rotation(R.random())
            self.scene.step_physics(0.4)
