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
from pathlib import Path


class Object(Placeable):
    """This is just a functional wrapper around the blender object.
    It is not meant to be instantiated directly. Use the factory methods of
        simpose.Scene instead
    It has no internal state, everything is delegated to the blender object.
    """

    def __init__(self, bl_object) -> None:
        super().__init__(bl_object)

    @property
    def material(self) -> Material:
        return self._bl_object.data.materials[0]

    @property
    def shader_node(self) -> ShaderNodeBsdfPrincipled:
        return self.material.node_tree.nodes["Principled BSDF"]

    @staticmethod
    def from_obj(
        filepath: Path,
        object_id,
        add_physics: bool = False,
        mass: float = 1,
        friction: float = 0.5,
        restitution: float = 0.5,
        mesh_collision: bool = False,
    ):
        # clear selection
        bpy.ops.object.select_all(action="DESELECT")
        with redirect_stdout():
            bpy.ops.wm.obj_import(filepath=str(filepath.resolve()))
        bl_object = bpy.context.selected_objects[0]

        bl_object.pass_index = object_id

        if add_physics:
            # add active rigid body
            bpy.ops.rigidbody.object_add(type="ACTIVE")
            # change to mesh collision
            if mesh_collision:
                bpy.context.object.rigid_body.collision_shape = "MESH"
            # set mass
            bpy.context.object.rigid_body.mass = mass
            # set friction
            bpy.context.object.rigid_body.friction = friction
            # set restitution
            bpy.context.object.rigid_body.restitution = restitution

        return Object(bl_object)

    @property
    def object_id(self):
        return self._bl_object.pass_index

    def set_metallic_value(self, value):
        self.shader_node.inputs["Metallic"].default_value = value

    def set_roughness_value(self, value):
        self.shader_node.inputs["Roughness"].default_value = value

    def get_name(self) -> str:
        return self._bl_object.name

    def get_class(self) -> str:
        return re.match("([\w]+)(.[0-9])*", self.get_name()).group(1)

    def __str__(self) -> str:
        return f"Object( name={self.get_name()}, class={self.get_class()}"
