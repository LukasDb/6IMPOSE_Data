import bpy
import mathutils
import math
from scipy.spatial.transform import Rotation as R
from typing import Tuple

class Camera:
    def __init__(self, name: str):
        self.name = name
        bpy.ops.object.camera_add()
        self._bl_object = bpy.data.objects['Camera']
        self._bl_object.name = name
        bpy.context.scene.camera = self._bl_object

    @property
    def location(self) -> Tuple:
        return self._bl_object.location.to_tuple()

    @property
    def rotation(self) -> R:
        """ returns orientation of camera in OpenCV format """
        self._bl_object.rotation_mode = "QUATERNION"
        q = self._bl_object.rotation_quaternion
        orn = R.from_quat([q.x, q.y, q.z, q.w])
        to_cv2 = R.from_euler("x", 180, degrees=True)
        return orn * to_cv2
    