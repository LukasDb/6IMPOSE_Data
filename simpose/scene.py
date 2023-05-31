from typing import Dict
from simpose.camera import Camera
from simpose.light import Light
from simpose.object import Object
import bpy

class Scene:
    def __init__(self) -> None:
        self._bl_scene = bpy.data.scenes.new("6impose Scene")
        bpy.context.window.scene = self._bl_scene

        # setup settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPTIX'
        bpy.context.scene.cycles.samples = 64
        bpy.context.scene.cycles.caustics_reflective = False
        bpy.context.scene.cycles.caustics_refractive = False
        bpy.context.scene.cycles.use_auto_tile = False
        bpy.context.scene.render.resolution_x = 640
        bpy.context.scene.render.resolution_y = 480
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.use_persistent_data = True
        bpy.context.scene.view_layers[0].cycles.use_denoising = True

        # setup output settings
        


    def export_blend(self, filepath):
        bpy.ops.wm.save_as_mainfile(filepath=filepath)

    def __enter__(self):
        bpy.context.window.scene = self._bl_scene
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    
    def render(self):
        bpy.ops.render.render(write_still=False)