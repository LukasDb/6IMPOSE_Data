import bpy
import simpose as sp
from .placeable import Placeable


class Camera(Placeable):
    def __init__(self, name: str):
        bpy.ops.object.camera_add()
        bl_cam = bpy.context.selected_objects[0]
        bl_cam.name = name
        super().__init__(bl_object=bl_cam)
        bpy.context.scene.camera = bl_cam

    def __str__(self) -> str:
        return f"Camera(name={self._bl_object.name})"
