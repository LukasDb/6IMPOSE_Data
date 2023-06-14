import bpy
from .materials.common import PrincipledBSDFMaterial
import mathutils
import math
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from .placeable import Placeable
import logging
from .redirect_stdout import redirect_stdout


class Object(Placeable):
    """Renderable Object with semantics"""

    def __init__(self, bl_object, object_id: int | None = None) -> None:
        super().__init__(bl_object)
        self.object_id = object_id

    @staticmethod
    def from_obj(filepath, object_id: int):
        with redirect_stdout():
            bpy.ops.wm.obj_import(filepath=filepath)
        return Object(bpy.context.selected_objects[0], object_id=object_id)

    def add_material(self):
        # Create the material instance
        material = PrincipledBSDFMaterial()
        # Create the Blender material and shader node
        blender_material, shader_node = material.create_material(name="MyMaterial")
        self._bl_object.data.materials.append(blender_material)
    def set_metallic_value(self,value):
        self._bl_object.data.materials[1].metallic = value
        
    def set_roughness_value(self,value):
        self._bl_object.data.materials[1].roughness = value
        
    
    def __str__(self) -> str:
        return f"Object(id={self.object_id}, name={self._bl_object.name})"
