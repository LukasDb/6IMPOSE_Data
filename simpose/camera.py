import bpy
import simpose as sp
from .placeable import Placeable
import mathutils 

class Camera(Placeable):
    def __init__(self, name: str):
        bpy.ops.object.camera_add()
        bl_cam = bpy.context.selected_objects[0]
        bl_cam.name = name
        super().__init__(bl_object=bl_cam)
        self.data = bl_cam.data  # Set the 'data' attribute to the camera's data
        bpy.context.scene.camera = bl_cam

    def __str__(self) -> str:
        return f"Camera(name={self._bl_object.name})"
    
    def get_calibration_matrix_K_from_blender(self):
        # always assume square pixels

        f_in_mm = self.data.lens
        scene = bpy.context.scene
        resolution_x_in_px = scene.render.resolution_x
        resolution_y_in_px = scene.render.resolution_y
        scale = scene.render.resolution_percentage / 100
        sensor_width_in_mm = self.data.sensor_width
        sensor_height_in_mm = self.data.sensor_height
        
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        #s_v = resolution_y_in_px * scale / sensor_height_in_mm # wrong for some reason
        pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
        s_v = s_u * pixel_aspect_ratio

        # Parameters of intrinsic calibration matrix K
        alpha_u = f_in_mm * s_u
        alpha_v = f_in_mm * s_v
        u_0 = resolution_x_in_px * (0.5 - self.data.shift_x)
        v_0 = resolution_y_in_px * (0.5 + self.data.shift_y) # because flipped blender camera
        skew = 0 # only use rectangular pixels

        K = mathutils.Matrix(
            ((alpha_u, skew,    u_0),
            (    0  , alpha_v, v_0),
            (    0  , 0,        1 )))
        return K
