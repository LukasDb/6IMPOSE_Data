import simpose as sp
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from .randomizer import Randomizer, RandomizerConfig


class CameraSceneRandomizerConfig(RandomizerConfig):
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    pitch_range: Tuple[float, float] = (20, 90)
    fill_range: tuple[float, float] = (0.1, 0.4)
    roll_jitter: float = 20.0
    pitch_jitter: float = 5.0
    yaw_jitter: float = 5.0
    min_cam2obj_distance: float = 0.1

    @staticmethod
    def get_description() -> dict[str, str]:
        return {
            "origin": "point the camera towards this point",
            "pitch_range": "Elevation of the camera to the origin (Z+ is up == 90°)",
            "roll_jitter": "Maximum jitter applied to camera roll angle when looking at the origin",
            "fill_range": "How much percent of the image should the largest object fill",
            "pitch_jitter": "Maximum jitter applied to camera pitch angle when looking at the origin",
            "yaw_jitter": "Maximum jitter applied to camera yaw angle when looking at the origin",
        }


class CameraSceneRandomizer(Randomizer):
    def __init__(self, params: CameraSceneRandomizerConfig):
        super().__init__(params)
        self.params = params

    def call(self, caller: sp.observers.Observable) -> None:
        scene = caller
        assert isinstance(scene, sp.Scene)
        p = self.params
        cam_view = np.array([0.0, 0.0, 1.0])
        cam = scene.get_cameras()[0]

        import bpy

        bpy.context.view_layer.update()

        obj_locations = list([obj.location for obj in scene.get_active_objects()])
        obj_diameters = list([obj.get_diameter() for obj in scene.get_active_objects()])

        max_object_diameter = max(obj_diameters)

        # the largest object should subtend fill * hfov of the image:
        # object_fov = fill * cam_fov
        # tan(object_fov / 2) = (max_object_diameter / 2) / dist2cam
        # dist2cam = (max_object_diameter / 2) / tan(fill * cam_fov / 2)
        fov = cam.hfov
        distance_range = (
            max_object_diameter / 2 / np.tan(p.fill_range[0] * fov / 2),
            max_object_diameter / 2 / np.tan(p.fill_range[1] * fov / 2),
        )

        assert (p.pitch_range[1] - p.pitch_range[0]) >= 5, "Pitch range is too small"

        # rejection sample until found valid angle and far away enough from other objects
        loc = None
        for _ in range(10000):
            radius = np.random.uniform(
                distance_range[1], distance_range[0]
            )  # small fill range -> large distance
            rot = R.random()
            candidate = rot.apply(cam_view) * radius + np.array(p.origin)
            d = np.linalg.norm(candidate[:2])
            pitch = np.arctan2(candidate[2], d) * 180 / np.pi
            if not (p.pitch_range[0] < pitch < p.pitch_range[1]):
                continue

            dists2objects = [
                np.linalg.norm(np.array(candidate) - np.array(obj_loc)) - obj_dia / 2
                for obj_loc, obj_dia in zip(obj_locations, obj_diameters)
            ]
            if all(np.array(dists2objects) > p.min_cam2obj_distance):
                loc = candidate
                break

        if loc is None:
            raise RuntimeError("Could not find a valid camera location.")

        cam.set_location(loc)
        cam.point_at(np.array(p.origin))  # with z up

        cam.apply_local_rotation_offset(
            R.from_euler(
                "xyz",
                (
                    np.random.uniform(-p.pitch_jitter, p.pitch_jitter),
                    np.random.uniform(-p.yaw_jitter, p.yaw_jitter),
                    np.random.uniform(-p.roll_jitter, p.roll_jitter),
                ),
                degrees=True,
            )
        )
        sp.logger.debug(f"Placed camera at {loc}.")
