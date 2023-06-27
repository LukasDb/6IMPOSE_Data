import bpy
from bpy.types import ShaderNodeBsdfPrincipled, Material
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

        self.material: Material = self._bl_object.data.materials[0]
        self.shader_node: ShaderNodeBsdfPrincipled = self.material.node_tree.nodes["Principled BSDF"]
        
        # set object id to be rendered as object index
        self._bl_object.pass_index = self.object_id


    @staticmethod
    def from_obj(filepath, object_id: int):
        with redirect_stdout():
            bpy.ops.wm.obj_import(filepath=filepath)
        return Object(bpy.context.selected_objects[0], object_id=object_id)


    def set_metallic_value(self, value):
        self.shader_node.inputs["Metallic"].default_value = value

    def set_roughness_value(self, value):
        self.shader_node.inputs["Roughness"].default_value = value
        
    def get_name(self) -> str:
        return self._bl_object.name
    
    def __str__(self) -> str:
        return f"Object(id={self.object_id}, name={self._bl_object.name})"
