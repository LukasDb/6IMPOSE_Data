import simpose as sp
from .generator import Generator, GeneratorParams

import multiprocessing as mp
from pathlib import Path
import numpy as np
import trimesh
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


class SimpleGenConfig(GeneratorParams):
    obj_path: Path = Path("path/to/obj")
    friction: float = 0.9
    restitution: float = 0.1
    cam_hfov: float = 70
    img_w: int = 640
    img_h: int = 480
    floor_textures_dir: Path = Path("path/to/textures")

    @classmethod
    def get_description(cls) -> dict[str, str]:
        description = super().get_description()
        description.update(
            {
                "obj_path": "Path to the object to be dropped",
                "friction": "Friction of objects",
                "restitution": "Restitution of objects",
                "cam_hfov": "Camera horizontal field of view",
                "img_w": "Image width",
                "img_h": "Image height",
                "floor_textures_dir": "Directory containing floor textures",
            }
        )
        return description


def indent(text: str) -> str:
    pad = "  "
    return "\n".join(pad + line for line in text.split("\n"))


def entry(name: str, type: str, params: str) -> str:
    spec = indent(f"type: {type}\nparams:\n{indent(params)}")
    return f"{name}:\n{spec}"


class SimpleGen(Generator):
    params: SimpleGenConfig

    @staticmethod
    def generate_template_config() -> str:
        gen_params = SimpleGenConfig.dump_with_comments(n_workers=1, n_parallel_on_gpu=1)
        writer_params = sp.writers.WriterConfig.dump_with_comments()

        light_params = sp.random.LightRandomizerConfig.dump_with_comments(
            trigger=sp.observers.Event.BEFORE_RENDER
        )

        randomizers_entries = "\n".join(
            [
                entry("light", "LightRandomizer", light_params),
            ]
        )

        output = "\n".join(
            [
                entry("Generator", "SimpleGen", gen_params),
                entry("Writer", "TFRecordWriter", writer_params),
                "Randomizers:",
                indent(randomizers_entries),
            ]
        )

        return output

    @staticmethod
    def generate_data(
        config: SimpleGenConfig,
        writer: sp.writers.Writer,
        randomizers: dict[str, sp.random.Randomizer],
        indices: np.ndarray,
    ) -> None:
        p = config

        # --- Generation params ---
        i = 0

        # -- SCENE --
        sp.logger.debug("Setting up Scene...")
        scene = sp.Scene(img_h=p.img_h, img_w=p.img_w)
        plane = scene.create_plane(size=1.0, with_physics=False)
        # plane.set_saturation(1.0)
        # plane.set_value(0.8)

        # -- CAMERA --
        sp.logger.debug("Setting up camera....")
        cam = scene.create_camera("Camera")
        cam.set_from_hfov(p.cam_hfov, scene.resolution_x, scene.resolution_y, degrees=True)
        cam.set_location((0, -0.5, 0.5))
        cam.set_rotation(R.from_euler("xyz", [-140, 0, 0], degrees=True))  # slightly down
        vfov = p.cam_hfov * p.img_h / p.img_w
        half_x_extent = np.tan(np.deg2rad(p.cam_hfov / 2))
        half_y_extent = np.tan(np.deg2rad(vfov / 2))

        # -- RANDOMIZERS --
        for randomizer in randomizers.values():
            randomizer.listen_to(scene)

        cfg = RandomImagePickerConfig(
            img_dir=p.floor_textures_dir, trigger=sp.observers.Event.BEFORE_RENDER
        )
        randimages = RandomImagePicker(cfg)
        randimages.listen_to(scene)
        randimages.randomize_plane(plane)

        sp.logger.debug("Loading object....")
        obj = scene.create_object(
            obj_path=p.obj_path,
            obj_name=p.obj_path.stem,
            add_semantics=True,
        )

        mesh = trimesh.load_mesh(str(p.obj_path.expanduser().resolve()))
        poses, probs = trimesh.poses.compute_stable_poses(
            mesh, center_mass=None, sigma=0.0, n_samples=1, threshold=0.0
        )
        sp.logger.debug(f"Found {len(poses)} stable poses")

        while True:
            dir_x = np.random.uniform(-half_x_extent, half_x_extent)
            dir_y = np.random.uniform(-half_y_extent, half_y_extent)
            direction = np.array([dir_x, dir_y, 1])
            direction /= np.linalg.norm(direction)
            direction = cam.rotation.apply(direction)

            # get a random position on the table
            obj_pos = get_xy_intersect(cam.location, direction)

            sp.logger.debug(
                f"Setting up object ({obj_pos[0]:.2f},{obj_pos[1]:.2f},{obj_pos[2]:.2f})...."
            )
            # sample an orientation using the probabilities
            selected = np.random.choice(np.arange(len(poses)), p=probs)
            obj_pose = poses[selected].copy()

            # apply random yaw
            yaw = np.random.uniform(-np.pi, np.pi)
            obj_pose[:3, :3] = R.from_euler("z", yaw).as_matrix() @ obj_pose[:3, :3]

            obj_pose[:2, 3] = obj_pos[:2]
            obj.set_location(obj_pose[:3, 3])
            obj.set_rotation(R.from_matrix(obj_pose[:3, :3]))

            sp.logger.debug("Rendering...")
            writer.write_data(indices[i], scene=scene)

            i += 1
            if i == len(indices):
                sp.logger.info(f"Finished generating data {mp.current_process().name}")
                return


def get_xy_intersect(start, direction):
    # z = 0 = start[2] + t * dir[2]
    t = -start[2] / direction[2]
    return start + t * direction
