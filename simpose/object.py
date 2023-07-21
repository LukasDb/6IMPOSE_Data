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
import numpy as np
import pybullet as p


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

    def copy(self, linked: bool) -> "Object":
        # clear blender selection
        bpy.ops.object.select_all(action="DESELECT")
        # select object
        self._bl_object.select_set(True)
        # returns a new object with a linked data block
        with redirect_stdout():
            bpy.ops.object.duplicate(linked=linked)
        bl_object = bpy.context.selected_objects[0]

        try:
            coll_id = self._bl_object["coll_id"]
            mass = self._bl_object["mass"]

            pb_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=coll_id,
                basePosition=[0.0, 0.0, 0.0],
            )
            p.changeDynamics(pb_id, -1, lateralFriction=self._bl_object["friction"])
            bl_object["pb_id"] = pb_id
        except KeyError:
            pass

        return Object(bl_object)

    @staticmethod
    def from_obj(
        filepath: Path,
        add_physics: bool = False,
        mass: float = 1,
        friction: float = 0.5,
        restitution: float = 0.5,
    ):
        # clear selection
        bpy.ops.object.select_all(action="DESELECT")
        with redirect_stdout():
            bpy.ops.wm.obj_import(filepath=str(filepath.resolve()))
        try:
            bl_object = bpy.context.selected_objects[0]
        except IndexError:
            raise RuntimeError(f"Could not import {filepath}")

        # set material output to cycles only
        bl_object.data.materials[0].use_nodes = True
        tree = bl_object.data.materials[0].node_tree
        tree.nodes["Material Output"].target = "CYCLES"

        attr_node = tree.nodes.new("ShaderNodeAttribute")
        attr_node.attribute_type = "VIEW_LAYER"
        attr_node.attribute_name = "object_index"
        obj_info_node = tree.nodes.new("ShaderNodeObjectInfo")
        compare_node = tree.nodes.new("ShaderNodeMath")
        compare_node.operation = "COMPARE"
        compare_node2 = tree.nodes.new("ShaderNodeMath")
        compare_node2.operation = "COMPARE"
        mat_output = tree.nodes.new("ShaderNodeOutputMaterial")
        mat_output.target = "EEVEE"
        transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
        mix_shader = tree.nodes.new("ShaderNodeMixShader")
        mix_shader2 = tree.nodes.new("ShaderNodeMixShader")

        # connect the attribute to the compare node
        tree.links.new(attr_node.outputs[0], compare_node.inputs[0])
        # conect object_info->pass_index to compare_node
        tree.links.new(obj_info_node.outputs["Object Index"], compare_node.inputs[1])
        # conenct output of compare_node mix shader
        tree.links.new(compare_node.outputs[0], mix_shader.inputs[0])
        tree.links.new(compare_node.outputs[0], mix_shader.inputs[2])
        # connect transparent shader to mix2 shader
        tree.links.new(transparent.outputs[0], mix_shader2.inputs[1])
        # add camera data to mix2 (z depth)
        tree.links.new(obj_info_node.outputs["Object Index"], mix_shader2.inputs[2])
        # connect attribute to compare2
        tree.links.new(attr_node.outputs[0], compare_node2.inputs[0])
        # set value2 of compare2 to 0
        compare_node2.inputs[1].default_value = 0
        # connect compare2 to mix2
        tree.links.new(compare_node2.outputs[0], mix_shader2.inputs[0])
        # connect mix2 to mix
        tree.links.new(mix_shader2.outputs[0], mix_shader.inputs[1])
        # output of mix shader to material output
        tree.links.new(mix_shader.outputs[0], mat_output.inputs[0])

        # set blend_method of material to alpha blend
        bl_object.data.materials[0].blend_method = "BLEND"

        # if add_physics:
        #     bpy.ops.rigidbody.object_add(type="ACTIVE")
        #     if mesh_collision:
        #         bpy.context.object.rigid_body.collision_shape = "MESH"
        #     bpy.context.object.rigid_body.mass = mass
        #     bpy.context.object.rigid_body.friction = friction
        #     bpy.context.object.rigid_body.restitution = restitution
        if add_physics:
            # add custom 'pb id' attribute to object
            coll_id = p.createCollisionShape(p.GEOM_MESH, fileName=str(filepath))
            pb_id = p.createMultiBody(
                baseMass=mass,
                baseCollisionShapeIndex=coll_id,
                basePosition=[0.0, 0.0, 0.0],
                baseOrientation=[0.0, 0.0, 0.0, 1.0],
            )
            p.changeDynamics(pb_id, -1, lateralFriction=friction)

            bl_object["pb_id"] = pb_id
            bl_object["mass"] = mass
            bl_object["coll_id"] = coll_id
            bl_object["friction"] = friction

            # set friction

        obj = Object(bl_object)
        obj.set_location((0.0, 0.0, 0.0))
        obj.set_rotation(R.from_euler("x", 0, degrees=True))
        return obj

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
        return f"Object(name={self.get_name()}, class={self.get_class()}"

    def set_location(self, location: Tuple | np.ndarray):
        try:
            # update physics representation
            pb_id = self._bl_object["pb_id"]
            q = self._bl_object.rotation_quaternion
            p.resetBasePositionAndOrientation(pb_id, location, [q.x, q.y, q.z, q.w])
        except KeyError:
            pass
        return super().set_location(location)

    def set_rotation(self, rotation: R):
        try:
            # update physics representation
            pb_id = self._bl_object["pb_id"]
            p.resetBasePositionAndOrientation(
                pb_id, self._bl_object.location, rotation.as_quat()
            )
        except KeyError:
            pass
        return super().set_rotation(rotation)
