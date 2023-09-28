from scipy.spatial.transform import Rotation as R
from typing import Tuple
import numpy as np


class Placeable:
    """Object that can be placed in the 3D scene."""

    def __init__(self, bl_object) -> None:
        import bpy

        self._bl_object: bpy.types.Object = bl_object  # reference to internal blender object

    @property
    def location(self) -> Tuple:
        import mathutils

        loc: mathutils.Vector = self._bl_object.location  # type: ignore
        return loc.to_tuple()

    @property
    def rotation(self) -> R:
        self._bl_object.rotation_mode = "QUATERNION"
        q = self._bl_object.rotation_quaternion
        orn = R.from_quat([q.x, q.y, q.z, q.w])  # type: ignore
        return orn

    def set_location(self, location: Tuple | np.ndarray):
        import mathutils

        if isinstance(location, np.ndarray):
            location = tuple(location)
        self._bl_object.location = mathutils.Vector(location)

    def set_rotation(self, rotation: R):
        r = rotation.as_quat(canonical=False)
        self._bl_object.rotation_mode = "QUATERNION"
        # blender: scalar first, scipy: scalar last
        blender_quat = [r[3], r[0], r[1], r[2]]
        self._bl_object.rotation_quaternion = blender_quat

    def point_at(self, location: np.ndarray):
        """point z+ towards location with world z-axis pointing up"""
        to_point = self.location - location
        yaw = np.arctan2(to_point[1], to_point[0]) + np.pi / 2
        pitch = -np.pi / 2 - np.arctan2(to_point[2], np.linalg.norm(to_point[:2]))
        towards_origin = R.from_euler("ZYX", [yaw, 0.0, pitch])
        self.set_rotation(towards_origin)

    def apply_global_offset(self, offset: Tuple | np.ndarray):
        """Apply offset to location."""
        import mathutils

        if isinstance(offset, np.ndarray):
            offset = tuple(offset)
        self._bl_object.location += mathutils.Vector(offset)

    def apply_local_offset(self, offset: Tuple | np.ndarray):
        """Apply offset to location."""
        import mathutils

        if isinstance(offset, np.ndarray):
            offset = tuple(offset)
        self._bl_object.rotation_mode = "QUATERNION"
        self._bl_object.location += self._bl_object.rotation_quaternion @ mathutils.Vector(offset)  # type: ignore

    def apply_global_rotation_offset(self, rotation: R):
        """Apply offset to rotation."""
        self.set_rotation(rotation * self.rotation)

    def apply_local_rotation_offset(self, rotation: R):
        """Apply offset to rotation."""
        self.set_rotation(self.rotation * rotation)
