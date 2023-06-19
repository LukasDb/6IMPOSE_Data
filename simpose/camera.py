import bpy
import simpose as sp
from .placeable import Placeable


class Camera(Placeable):
    def __init__(self, name: str):
        bpy.ops.object.camera_add()
        bl_cam = bpy.context.selected_objects[0]
        bl_cam.name = name
        super().__init__(bl_object=bl_cam)
        self.data = bl_cam.data  # Set the 'data' attribute to the camera's data
        bpy.context.scene.camera = bl_cam

    def add_background(self):
        bg_image_path = "backgrounds/geo_image.jpg"  # Replace with the path to your image file
        bg_image = bpy.data.images.load(bg_image_path)
        bg_image_obj = bpy.data.objects.new("BackgroundImage", None)
        bg_image_obj.data = bg_image
        bpy.context.collection.objects.link(bg_image_obj)
        bg_image_slot = self.data.background_images.new()
        bg_image_slot.image = bg_image
        bg_image_obj.location = (-2, 0, 0)  # Adjust the position (move it closer or farther from the camera
        bg_image_obj.scale = (1, 1, 1)  # Adjust the scale (make it larger or smaller)

    def __str__(self) -> str:
        return f"Camera(name={self._bl_object.name})"

