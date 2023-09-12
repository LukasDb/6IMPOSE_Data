import bpy
from bpy.types import ShaderNodeBsdfPrincipled, Material
import mathutils
import math
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List
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
    def materials(self) -> List[Material]:
        return list(self._bl_object.data.materials)  # type: ignore

    @property
    def shader_nodes(self) -> List[ShaderNodeBsdfPrincipled]:
        return [m.node_tree.nodes["Principled BSDF"] for m in self.materials]  # type: ignore

    @property
    def is_hidden(self) -> bool:
        return self._bl_object.hide_render

    @property
    def has_semantics(self) -> bool:
        return self._bl_object["semantics"]

    @property
    def object_id(self):
        return self._bl_object.pass_index

    def copy(self) -> "Object":
        # clear blender selection
        bpy.ops.object.select_all(action="DESELECT")
        # select object
        self._bl_object.select_set(True)
        # returns a new object with a linked data block
        with redirect_stdout():
            bpy.ops.object.duplicate(linked=False)
        bl_object = bpy.context.selected_objects[0]
        bl_object.active_material = self._bl_object.active_material.copy()  # type: ignore

        new_obj = Object(bl_object)

        if self._bl_object.get("pb_id") is not None:
            new_obj._add_pybullet_object()

        return new_obj

    @staticmethod
    def from_obj(
        filepath: Path,
        add_semantics: bool = False,
        mass: float | None = None,
        friction: float = 0.5,
        scale: float = 1.0,
    ):
        # clear selection
        bpy.ops.object.select_all(action="DESELECT")
        with redirect_stdout():
            bpy.ops.wm.obj_import(filepath=str(filepath.resolve()))
        try:
            bl_object = bpy.context.selected_objects[0]
        except IndexError:
            raise RuntimeError(f"Could not import {filepath}")

        obj = Object._initialize_bl_object(
            bl_object=bl_object,
            scale=scale,
            mass=mass,
            obj_path=filepath,
            add_semantics=add_semantics,
            friction=friction,
        )
        return obj

    @staticmethod
    def from_ply(
        filepath: Path,
        add_semantics: bool = False,
        mass: float | None = None,
        friction: float = 0.5,
        scale: float = 1.0,
    ):
        # clear selection
        bpy.ops.object.select_all(action="DESELECT")
        with redirect_stdout():
            bpy.ops.import_mesh.ply(filepath=str(filepath.resolve()))  # type: ignore
        try:
            bl_object: bpy.types.Object = bpy.context.selected_objects[0]
        except IndexError:
            raise RuntimeError(f"Could not import {filepath}")

        # convert ply to .obj for pybullet
        collision_obj_path = (
            filepath.with_name("collision_" + filepath.name).with_suffix(".obj").resolve()
        )
        with redirect_stdout():
            if not collision_obj_path.exists():
                bpy.ops.wm.obj_export(
                    filepath=str(collision_obj_path),
                    export_materials=False,
                    export_colors=False,
                    export_normals=True,
                )
        # create new material for object
        material: bpy.types.Material = bpy.data.materials.new(name="Material")
        material.use_nodes = True
        # create color attribute node and connect to base color of principled bsdf
        color_node: bpy.types.ShaderNodeAttribute = material.node_tree.nodes.new(
            "ShaderNodeAttribute"
        )  # type: ignore
        color_node.attribute_name = "Col"
        material.node_tree.links.new(
            color_node.outputs["Color"],
            material.node_tree.nodes["Principled BSDF"].inputs["Base Color"],
        )
        # add to object
        bl_object.data.materials.append(material)  # type: ignore

        # shade smooth
        bpy.ops.object.shade_smooth()

        obj = Object._initialize_bl_object(
            bl_object=bl_object,
            scale=scale,
            mass=mass,
            obj_path=collision_obj_path,
            add_semantics=add_semantics,
            friction=friction,
        )
        return obj

    def export_mesh(self, output_dir: Path):
        """export mesh as ply file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        bpy.ops.object.select_all(action="DESELECT")
        self._bl_object.select_set(True)
        with redirect_stdout():
            old_loc = self.location
            old_rot = self.rotation
            try:
                self.set_location((0, 0, 0))
                self.set_rotation(R.from_euler("x", 0, degrees=True))
                bpy.ops.export_mesh.ply(  # type: ignore
                    filepath=str(output_dir / f"{self.get_class()}.ply"),
                    use_selection=True,
                    use_normals=True,
                    use_uv_coords=False,
                    use_colors=False,
                    use_mesh_modifiers=False,
                    use_ascii=False,
                )
            finally:
                self.set_location(old_loc)
                self.set_rotation(old_rot)

        logging.info("Exported mesh to " + str(output_dir / f"{self.get_class()}.ply"))

    def hide(self):
        if self.is_hidden:
            return

        try:
            self._remove_pybullet_object()
        except KeyError:
            pass

        self._bl_object.hide_render = True

    def show(self):
        if not self.is_hidden:
            return
        try:
            self._add_pybullet_object()
        except KeyError:
            pass
        self._bl_object.hide_render = False

    def set_metallic_value(self, value):
        for shader_node in self.shader_nodes:
            shader_node.inputs["Metallic"].default_value = value  # type: ignore

    def set_roughness_value(self, value):
        for shader_node in self.shader_nodes:
            shader_node.inputs["Roughness"].default_value = value  # type: ignore

    def get_name(self) -> str:
        return self._bl_object.name

    def get_class(self) -> str:
        result = re.match(r"([\w]+)(.[0-9])*", self.get_name())
        assert result is not None, "Could not extract class name from object name"
        return result.group(1)

    def __str__(self) -> str:
        return f"Object(name={self.get_name()}, class={self.get_class()}"

    def set_location(self, location: Tuple | np.ndarray):
        try:
            # update physics representation
            pb_id = self._bl_object["pb_id"]
            q = self._bl_object.rotation_quaternion
            p.resetBasePositionAndOrientation(pb_id, location, [q.x, q.y, q.z, q.w])  # type: ignore
        except KeyError:
            pass
        return super().set_location(location)

    def set_rotation(self, rotation: R):
        try:
            # update physics representation
            pb_id = self._bl_object["pb_id"]
            p.resetBasePositionAndOrientation(pb_id, self._bl_object.location, rotation.as_quat())  # type: ignore
        except KeyError:
            pass
        return super().set_rotation(rotation)

    def remove(self):
        try:
            self._remove_pybullet_object()
        except KeyError:
            pass

        materials = self.materials
        # keep reference to mesh before removing object
        mesh: bpy.types.Mesh = self._bl_object.data  # type: ignore
        bpy.data.objects.remove(self._bl_object)
        if mesh.users == 0:
            # first, remove mesh data
            bpy.data.meshes.remove(mesh, do_unlink=True)
            try:
                coll_id = self._bl_object["coll_id"]
                p.removeCollisionShape(coll_id)
            except Exception:
                pass
        for material in materials:
            if material.users == 0:
                bpy.data.materials.remove(material, do_unlink=True)

    def _add_pybullet_object(self) -> int:
        coll_id = self._bl_object["coll_id"]
        mass = self._bl_object["mass"]
        friction = self._bl_object["friction"]

        pb_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=coll_id,
            basePosition=[0.0, 0.0, 0.0],
        )
        p.changeDynamics(pb_id, -1, lateralFriction=friction)
        p.resetBasePositionAndOrientation(
            pb_id, self._bl_object.location, self.rotation.as_quat(canonical=True)
        )
        self._bl_object["pb_id"] = pb_id
        return pb_id

    def _remove_pybullet_object(self):
        p.removeBody(self._bl_object["pb_id"])
        del self._bl_object["pb_id"]

    def set_semantic_id(self, id: int):
        self._bl_object.pass_index = id

    @staticmethod
    def _initialize_bl_object(
        bl_object: bpy.types.Object,
        obj_path: Path,
        add_semantics: bool,
        scale: float,
        mass: float | None,
        friction: float,
    ) -> "Object":
        # scale object
        bl_object.scale = (scale, scale, scale)
        materials: list[bpy.types.Material] = bl_object.data.materials  # type: ignore
        for material in materials:
            tree: bpy.types.NodeTree = material.node_tree
            material.blend_method = "BLEND"

            # set current material output to cycles only
            mat_output: bpy.types.ShaderNodeOutputMaterial = tree.nodes["Material Output"]  # type: ignore
            mat_output.target = "CYCLES"

            #          | vl > 0 | vl == 0
            # vl == oi |    1   |   0 or oi
            # vl != oi |  trans | oi

            # --> vl == 0 -> oi
            # else:
            #  if vl == oi: 1
            #  if vl != oi: trans

            # add eevee output with above truth table (vl == view_layer["Object Index"]; oi == object_index)
            vl: bpy.types.ShaderNodeAttribute = tree.nodes.new("ShaderNodeAttribute")  # type: ignore
            vl.attribute_type = "VIEW_LAYER"
            vl.attribute_name = "object_index"
            vl.location = (300, 100)

            oi = tree.nodes.new("ShaderNodeObjectInfo")
            oi.location = (300, -200)

            vl_is_0: bpy.types.ShaderNodeMath = tree.nodes.new("ShaderNodeMath")  # type: ignore
            vl_is_0.operation = "COMPARE"
            vl_is_0.inputs[2].default_value = 0.1  # type: ignore
            vl_is_0.inputs[1].default_value = 0  # type: ignore
            vl_is_0.location = (600, 100)

            vl_is_oi: bpy.types.ShaderNodeMath = tree.nodes.new("ShaderNodeMath")  # type: ignore
            vl_is_oi.operation = "COMPARE"
            vl_is_oi.inputs[2].default_value = 0.1  # type: ignore
            vl_is_oi.location = (600, -200)

            id_output: bpy.types.ShaderNodeOutputMaterial = tree.nodes.new(
                "ShaderNodeOutputMaterial"
            )  # type: ignore
            id_output.target = "EEVEE"
            id_output.location = (1100, 0)

            transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
            transparent.location = (600, -100)
            mix_shader = tree.nodes.new("ShaderNodeMixShader")
            mix_shader.location = (1000, 0)
            mix_shader2 = tree.nodes.new("ShaderNodeMixShader")
            mix_shader2.location = (800, -200)

            # connect vl_is_0
            tree.links.new(vl.outputs[0], vl_is_0.inputs[0])
            # connect result to mix shader1
            tree.links.new(vl_is_0.outputs[0], mix_shader.inputs[0])
            # if vl_is_0 -> return oi -> output of mix shader is output
            tree.links.new(oi.outputs["Object Index"], mix_shader.inputs[2])
            tree.links.new(mix_shader.outputs[0], id_output.inputs[0])

            # if vl is not 0 then the other mix input is used, which is either transparent or 1 (from mix2)
            tree.links.new(mix_shader2.outputs[0], mix_shader.inputs[1])

            # connect vl_is_oi
            tree.links.new(vl.outputs[0], vl_is_oi.inputs[0])
            tree.links.new(oi.outputs["Object Index"], vl_is_oi.inputs[1])
            # connect result to mix shader2
            tree.links.new(vl_is_oi.outputs[0], mix_shader2.inputs[0])
            # connect transparent shader to mix2 shader (if vl != oi)
            tree.links.new(transparent.outputs[0], mix_shader2.inputs[1])
            # else we return 1 which is also the result of vl_is_oi
            tree.links.new(vl_is_oi.outputs[0], mix_shader2.inputs[2])

        obj = Object(bl_object)
        if mass is not None:
            # use vhacd to create collision shape
            out_path = obj_path.resolve().with_name(obj_path.stem + "_vhacd.obj")
            if not out_path.exists():
                # hierarchical decomposition for dynamic collision of concave objects
                # logging.info(f"running vhacd for {obj_path}...")
                with redirect_stdout():
                    p.vhacd(
                        str(obj_path.resolve()),
                        str(out_path),
                        str(obj_path.parent.joinpath("log.txt").resolve()),
                    )
            else:
                pass
                # logging.info(f"Reusing vhacd from {out_path}")

            try:
                with redirect_stdout():
                    coll_id = p.createCollisionShape(
                        p.GEOM_MESH, fileName=str(out_path), meshScale=[scale] * 3
                    )
            except Exception as e:
                import traceback

                logging.error(
                    f"Collision shape from {out_path} failed, using convex hull from {obj_path} instead!\n{e}\n{traceback.format_exc()}"
                )
                with redirect_stdout():
                    coll_id = p.createCollisionShape(
                        p.GEOM_MESH, fileName=str(obj_path.resolve()), meshScale=[scale] * 3
                    )

            obj._bl_object["coll_id"] = coll_id
            obj._bl_object["mass"] = mass
            obj._bl_object["friction"] = friction
            obj._add_pybullet_object()

        obj._bl_object["semantics"] = add_semantics
        obj._bl_object.pass_index = 0

        obj.set_location((0.0, 0.0, 0.0))
        obj.set_rotation(R.from_euler("x", 0, degrees=True))

        return obj
