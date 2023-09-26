import bpy
from bpy.types import ShaderNodeBsdfPrincipled, Material, ShaderNodeHueSaturation
from scipy.spatial.transform import Rotation as R
from typing import Tuple, List
from .placeable import Placeable
import logging
from .redirect_stdout import redirect_stdout
import re
from pathlib import Path
import numpy as np
import pybullet as p
from enum import Enum

logger = logging.getLogger("simpose")


class _ObjectAppearance(Enum):
    METALLIC = "Metallic"
    ROUGHNESS = "Roughness"
    HUE = "Hue"
    SATURATION = "Saturation"
    VALUE = "Value"


class Object(Placeable):
    """This is just a functional wrapper around the blender object.
    It is not meant to be instantiated directly. Use the factory methods of
        simpose.Scene instead
    It has no internal state, everything is delegated to the blender object.
    """

    ObjectAppearance = _ObjectAppearance

    def __init__(self, bl_object) -> None:
        super().__init__(bl_object)

    @property
    def materials(self) -> List[Material]:
        return list(self._bl_object.data.materials)  # type: ignore

    @property
    def shader_nodes(self) -> List[ShaderNodeBsdfPrincipled]:
        return [m.node_tree.nodes["Principled BSDF"] for m in self.materials]  # type: ignore

    @property
    def hsv_nodes(self) -> List[ShaderNodeHueSaturation]:
        return [m.node_tree.nodes["sp_hsv"] for m in self.materials]  # type: ignore

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
            bpy.ops.import_scene.obj(filepath=str(filepath.resolve()), use_split_objects=False)
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
    def from_gltf(
        filepath: Path,
        add_semantics: bool = False,
        mass: float | None = None,
        friction: float = 0.5,
        scale: float = 1.0,
    ):
        # clear selection
        bpy.ops.object.select_all(action="DESELECT")
        with redirect_stdout():
            bpy.ops.import_scene.gltf(filepath=str(filepath.resolve()))

        try:
            bl_object = bpy.context.selected_objects[0]
        except IndexError:
            raise RuntimeError(f"Could not import {filepath}")

        collision_obj_path = filepath.with_suffix(".obj").resolve()
        with redirect_stdout():
            if not collision_obj_path.exists():
                bpy.ops.wm.obj_export(
                    filepath=str(collision_obj_path),
                    export_materials=False,
                    export_colors=False,
                    export_normals=True,
                    export_selected_objects=True,
                    apply_modifiers=False,
                )

        obj = Object._initialize_bl_object(
            bl_object=bl_object,
            scale=scale,
            mass=mass,
            obj_path=collision_obj_path,
            add_semantics=add_semantics,
            friction=friction,
        )
        return obj

    @staticmethod
    def from_fbx(
        filepath: Path,
        add_semantics: bool = False,
        mass: float | None = None,
        friction: float = 0.5,
        scale: float = 1.0,
    ):
        # clear selection
        bpy.ops.object.select_all(action="DESELECT")
        with redirect_stdout():
            bpy.ops.import_scene.fbx(filepath=str(filepath.resolve()), use_anim=False)

        # how many objects are selected?
        if len(bpy.context.selected_objects) > 1:
            # choose active object to be joined to
            # select the one not containing "transparent" -> more likely to have a proper name
            bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
            for obj in bpy.context.selected_objects:
                if "transparent" not in obj.name:
                    bpy.context.view_layer.objects.active = obj
                    break
            bpy.ops.object.join()

        try:
            bl_object = bpy.context.selected_objects[0]
        except IndexError:
            raise RuntimeError(f"Could not import {filepath}")

        collision_obj_path = filepath.with_suffix(".obj").resolve()
        with redirect_stdout():
            if not collision_obj_path.exists():
                bpy.ops.wm.obj_export(
                    filepath=str(collision_obj_path),
                    export_materials=False,
                    export_colors=False,
                    export_normals=True,
                    export_selected_objects=True,
                    apply_modifiers=False,
                )

        obj = Object._initialize_bl_object(
            bl_object=bl_object,
            scale=scale,
            mass=mass,
            obj_path=collision_obj_path,
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
        collision_obj_path = filepath.with_suffix(".obj").resolve()
        with redirect_stdout():
            if not collision_obj_path.exists():
                bpy.ops.wm.obj_export(
                    filepath=str(collision_obj_path),
                    export_materials=False,
                    export_colors=False,
                    export_normals=True,
                    export_selected_objects=True,
                    apply_modifiers=False,
                )
        # create new material for object
        material: bpy.types.Material = bpy.data.materials.new(name="sp_Material")
        material.use_nodes = True
        # create color attribute node and connect to base color of principled bsdf
        color_node: bpy.types.ShaderNodeAttribute = material.node_tree.nodes.new(
            "ShaderNodeAttribute"
        )  # type: ignore
        color_node.name = "sp_" + color_node.name
        color_node.attribute_name = "Col"
        material.node_tree.links.new(
            color_node.outputs["Color"],
            material.node_tree.nodes["Principled BSDF"].inputs["Base Color"],
        )
        # add to object
        bl_object.data.materials.append(material)  # type: ignore

        obj = Object._initialize_bl_object(
            bl_object=bl_object,
            scale=scale,
            mass=mass,
            obj_path=collision_obj_path,
            add_semantics=add_semantics,
            friction=friction,
        )
        return obj

    def export_as_ply(self, output_dir: Path):
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

        logger.info("Exported mesh to " + str(output_dir / f"{self.get_class()}.ply"))

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

    def get_appearance(self, appearance: ObjectAppearance) -> float:
        if appearance.value in [
            _ObjectAppearance.METALLIC.value,
            _ObjectAppearance.ROUGHNESS.value,
        ]:
            return self.shader_nodes[0].inputs[appearance.value].default_value  # type: ignore

        elif appearance.value in [
            _ObjectAppearance.HUE.value,
            _ObjectAppearance.SATURATION.value,
            _ObjectAppearance.VALUE.value,
        ]:
            return self.hsv_nodes[0].inputs[appearance.value].default_value  # type: ignore
        else:
            raise ValueError(f"Unknown appearance: {appearance}")

    def get_default_appearance(self, appearance: ObjectAppearance) -> float:
        return self._bl_object[f"default_{appearance.value}"]

    def set_appearance(self, appearance: ObjectAppearance, value, set_default=True):
        if set_default:
            self._bl_object[f"default_{appearance.value}"] = value

        if appearance.value in [
            _ObjectAppearance.METALLIC.value,
            _ObjectAppearance.ROUGHNESS.value,
        ]:
            for shader_node in self.shader_nodes:
                shader_node.inputs[appearance.value].default_value = value  # type: ignore

        elif appearance.value in [
            _ObjectAppearance.HUE.value,
            _ObjectAppearance.SATURATION.value,
            _ObjectAppearance.VALUE.value,
        ]:
            for hsv_node in self.hsv_nodes:
                hsv_node.inputs[appearance.value].default_value = value  # type: ignore

    def set_metallic(self, value, set_default=True):
        self.set_appearance(_ObjectAppearance.METALLIC, value, set_default=set_default)

    def set_roughness(self, value, set_default=True):
        self.set_appearance(_ObjectAppearance.ROUGHNESS, value, set_default=set_default)

    def set_hue(self, value, set_default=True):
        self.set_appearance(_ObjectAppearance.HUE, value, set_default=set_default)

    def set_saturation(self, value, set_default=True):
        self.set_appearance(_ObjectAppearance.SATURATION, value, set_default=set_default)

    def set_value(self, value, set_default=True):
        self.set_appearance(_ObjectAppearance.VALUE, value, set_default=set_default)

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
            self._set_pybullet_pose(location, self.rotation)
        except KeyError:
            pass
        return super().set_location(location)

    def set_rotation(self, rotation: R):
        try:
            self._set_pybullet_pose(self.location, rotation)
        except KeyError:
            pass
        return super().set_rotation(rotation)

    def _set_pybullet_pose(self, location: Tuple | np.ndarray, rotation: R):
        try:
            # update physics representation
            pb_id = self._bl_object["pb_id"]
            com = np.array(self._bl_object["COM"])

            loc = np.array(location) - rotation.apply(np.array(com))
            p.resetBasePositionAndOrientation(pb_id, loc, rotation.as_quat(canonical=True))
        except KeyError:
            pass

    def apply_pybullet_pose(self):
        pb_id = self._bl_object["pb_id"]
        com = np.array(self._bl_object["COM"])
        pos, orn = p.getBasePositionAndOrientation(pb_id)

        location = pos + self.rotation.apply(np.array(com))

        self.set_location(location)
        self.set_rotation(R.from_quat(orn))

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
            baseInertialFramePosition=[0.0, 0.0, 0.0],
            useMaximalCoordinates=True,
        )
        p.changeDynamics(
            pb_id,
            -1,
            lateralFriction=friction,
            spinningFriction=friction,
        )
        self._bl_object["pb_id"] = pb_id
        self._set_pybullet_pose(self.location, self.rotation)
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
            material.blend_method = "CLIP"

            # set current material output to cycles only
            mat_output: bpy.types.ShaderNodeOutputMaterial = tree.nodes["Material Output"]  # type: ignore
            mat_output.target = "CYCLES"

            Object.add_ID_to_material(material)

            # insert hsv node in between current color and bsdf node
            hsv_node: bpy.types.ShaderNodeHueSaturation = tree.nodes.new("ShaderNodeHueSaturation")  # type: ignore
            hsv_node.name = "sp_hsv"
            hsv_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # type: ignore
            hsv_node.inputs["Saturation"].default_value = 0.0  # type: ignore
            hsv_node.inputs["Value"].default_value = 1.0  # type: ignore
            hsv_node.location = (-300, 0)

            # find node that is connected to base color of bsdf node
            prev_color_node = None
            rgb_shader = tree.nodes["Principled BSDF"]
            assert rgb_shader.inputs["Base Color"].links is not None
            for link in rgb_shader.inputs["Base Color"].links:
                prev_color_node = link.from_node
                output_socket_name = link.from_socket.name
                prev_color_node.location = (-600, 0)

            if prev_color_node is not None:
                tree.links.new(prev_color_node.outputs[output_socket_name], hsv_node.inputs["Color"])  # type: ignore
                # connect hsv node to bsdf node
                tree.links.new(hsv_node.outputs["Color"], rgb_shader.inputs["Base Color"])  # type: ignore

        # initialize object
        obj = Object(bl_object)
        obj.set_location((0.0, 0.0, 0.0))
        obj.set_rotation(R.from_euler("x", 0, degrees=True))
        obj._bl_object["semantics"] = add_semantics
        obj._bl_object.pass_index = 0

        # set defaults for appearance
        obj.set_metallic(0.0, set_default=True)
        obj.set_roughness(0.5, set_default=True)
        obj.set_hue(0.5, set_default=True)
        obj.set_saturation(1.0, set_default=True)
        obj.set_value(1.0, set_default=True)

        # load physics object if given
        if mass is not None:
            out_path = obj_path.resolve().with_name(obj_path.stem + "_vhacd.obj")
            if not out_path.exists():
                # hierarchical decomposition for dynamic collision of concave objects
                with redirect_stdout():
                    p.vhacd(
                        str(obj_path.resolve()),
                        str(out_path),
                        str(obj_path.parent.joinpath("log.txt").resolve()),
                    )

            # find the center of gravity
            bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_VOLUME")
            offset = -np.array(bl_object.location)
            bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
            bpy.ops.object.origin_set(type="ORIGIN_CURSOR")

            try:
                with redirect_stdout():
                    coll_id = p.createCollisionShape(
                        p.GEOM_MESH,
                        fileName=str(out_path),
                        meshScale=[scale] * 3,
                        collisionFramePosition=offset,
                    )
            except Exception as e:
                import traceback

                logger.error(
                    f"Collision shape from {out_path} failed, using convex hull from {obj_path} instead!\n{e}\n{traceback.format_exc()}"
                )
                # find center of mass of the object

                with redirect_stdout():
                    coll_id = p.createCollisionShape(
                        p.GEOM_MESH,
                        fileName=str(obj_path.resolve()),
                        meshScale=[scale] * 3,
                        collisionFramePosition=offset,
                    )

            obj._bl_object["coll_id"] = coll_id
            obj._bl_object["mass"] = mass
            obj._bl_object["friction"] = friction
            obj._bl_object["COM"] = offset.tolist()
            obj._add_pybullet_object()
        return obj

    @staticmethod
    def add_ID_to_material(mat: bpy.types.Material):
        tree = mat.node_tree

        # create material output
        mat_output: bpy.types.ShaderNodeOutputMaterial = tree.nodes.new("ShaderNodeOutputMaterial")  # type: ignore
        mat_output.location = (1200, 0)
        mat_output.target = "EEVEE"

        # view layer object index: -1: rgb, 0: visb, 1: obj1, 2: obj2, ...
        vl: bpy.types.ShaderNodeAttribute = tree.nodes.new("ShaderNodeAttribute")  # type: ignore
        vl.attribute_type = "VIEW_LAYER"
        vl.attribute_name = "object_index"
        vl.location = (300, 100)

        oi = tree.nodes.new("ShaderNodeObjectInfo")
        oi.location = (300, -200)

        # check if view layer is 0 -> object index mode
        vl_is_object_index: bpy.types.ShaderNodeMath = tree.nodes.new("ShaderNodeMath")  # type: ignore
        vl_is_object_index.operation = "COMPARE"
        vl_is_object_index.inputs[2].default_value = 0.1  # type: ignore
        vl_is_object_index.inputs[1].default_value = 0  # type: ignore
        vl_is_object_index.location = (600, 100)
        tree.links.new(vl.outputs[0], vl_is_object_index.inputs[0])  # connect to view layer

        # check if view layer== object_index -> 1 if object_index==self.index else 0
        vl_is_current_object: bpy.types.ShaderNodeMath = tree.nodes.new("ShaderNodeMath")  # type: ignore
        vl_is_current_object.operation = "COMPARE"
        vl_is_current_object.inputs[2].default_value = 0.1  # type: ignore
        vl_is_current_object.location = (600, -200)
        # connect to view layer
        tree.links.new(vl.outputs[0], vl_is_current_object.inputs[0])
        # connect to own object index
        tree.links.new(oi.outputs["Object Index"], vl_is_current_object.inputs[1])

        transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
        transparent.location = (600, -100)

        # if not in RGB and not in object_index mode, 1 or transparent
        choose_1_or_transparent = tree.nodes.new("ShaderNodeMixShader")
        choose_1_or_transparent.location = (800, -200)
        # if current object is chosen
        tree.links.new(vl_is_current_object.outputs[0], choose_1_or_transparent.inputs[0])
        # if true: emit 1 (from the if statement)
        tree.links.new(vl_is_current_object.outputs[0], choose_1_or_transparent.inputs[2])
        # else: emit transparent
        tree.links.new(transparent.outputs[0], choose_1_or_transparent.inputs[1])

        # if not in RGB mode, choose transparent, 1, or object index
        choose_index_or_else = tree.nodes.new("ShaderNodeMixShader")
        choose_index_or_else.location = (1000, 0)
        # if object index mode
        tree.links.new(vl_is_object_index.outputs[0], choose_index_or_else.inputs[0])
        # if true: emit object index
        tree.links.new(oi.outputs["Object Index"], choose_index_or_else.inputs[2])
        # else: emit 1 or transparent
        tree.links.new(choose_1_or_transparent.outputs[0], choose_index_or_else.inputs[1])

        # connect to material output
        tree.links.new(choose_index_or_else.outputs[0], mat_output.inputs["Surface"])  # type: ignore
