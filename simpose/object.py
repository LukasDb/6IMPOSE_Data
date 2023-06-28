import bpy
from bpy.types import ShaderNodeBsdfPrincipled, Material
import mathutils
import math
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from .placeable import Placeable
import logging
from .redirect_stdout import redirect_stdout
import re


class Object(Placeable):
    """Renderable Object with semantics"""

    def __init__(self, bl_object) -> None:
        super().__init__(bl_object)


        self.object_id = bpy.context.window.scene["id_counter"]
        bpy.context.window.scene["id_counter"] += 1


        self.material: Material = self._bl_object.data.materials[0]
        self.shader_node: ShaderNodeBsdfPrincipled = self.material.node_tree.nodes["Principled BSDF"]
        
        # set object id to be rendered as object index
        self._bl_object.pass_index = self.object_id

    @staticmethod
    def from_obj(filepath):
        with redirect_stdout():
            bpy.ops.wm.obj_import(filepath=filepath)
        return Object(bpy.context.selected_objects[0])


    def set_metallic_value(self, value):
        self.shader_node.inputs["Metallic"].default_value = value

    def set_roughness_value(self, value):
        self.shader_node.inputs["Roughness"].default_value = value
        
    def get_name(self) -> str:
        return self._bl_object.name
    
    def get_class(self) -> str:
        return re.match("([\w]+)(.[0-9])*", self.get_name()).group(1)
    
    def __str__(self) -> str:
        return f"Object(id={self.object_id}, name={self._bl_object.name})"
    
    def copy(self, linked: bool)->"Object":
        # clear blender selection
        bpy.ops.object.select_all(action='DESELECT')
        # select object
        self._bl_object.select_set(True)
        # returns a new object with a linked data block
        with redirect_stdout():
            bpy.ops.object.duplicate(linked=linked)
        return Object(bpy.context.selected_objects[0])