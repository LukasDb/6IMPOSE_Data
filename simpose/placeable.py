import bpy
import mathutils
import math
from scipy.spatial.transform import Rotation as R
from typing import Tuple
import numpy as np


class Placeable:
    """Object that can be placed in the 3D scene."""

    def __init__(self, bl_object) -> None:
        self._bl_object = bl_object  # reference to internal blender object

    @property
    def location(self) -> Tuple:
        return self._bl_object.location.to_tuple()

    @property
    def rotation(self) -> R:
        """returns orientation of camera in OpenCV format"""
        self._bl_object.rotation_mode = "QUATERNION"
        q = self._bl_object.rotation_quaternion
        orn = R.from_quat([q.x, q.y, q.z, q.w])
        to_cv2 = R.from_euler("x", 180, degrees=True)
        return orn * to_cv2

    def set_location(self, location: Tuple | np.ndarray):
        if isinstance(location, np.ndarray):
            location = tuple(location)
        self._bl_object.location = mathutils.Vector(location)

    def set_rotation(self, rotation: R):
        to_blender = R.from_euler("x", -180, degrees=True)
        r = (rotation * to_blender).as_quat()
        self._bl_object.rotation_mode = "QUATERNION"
        # blender: scalar first, scipy: scalar last
        blender_quat = [r[3], r[0], r[1], r[2]]
        self._bl_object.rotation_quaternion = blender_quat
