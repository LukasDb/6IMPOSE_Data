import simpose as sp
from .object import Object
from pathlib import Path

with sp._redirect_stdout():
    import pybullet as p
    import pybullet_data


class Plane(Object):
    def __init__(self, bl_object) -> None:
        super().__init__(bl_object)

    @staticmethod
    def create(size: float = 2, with_physics: bool = True):
        import bpy

        if with_physics:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            pb_id = p.loadURDF("plane.urdf")  # XY ground plane
            p.changeDynamics(pb_id, -1, lateralFriction=0.5, restitution=0.9)

        # deselect blender objects
        bpy.ops.object.select_all(action="DESELECT")
        # generate a plane in blender
        bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, location=(0, 0, 0))

        bl_object = bpy.context.active_object

        bl_object["semantics"] = False
        bl_object.pass_index = 0

        # create material
        material = bpy.data.materials.new(name="plane_material")
        material.use_nodes = True
        material.blend_method = "CLIP"
        # add material
        bl_object.data.materials.append(material)  # type: ignore

        tree = material.node_tree
        rgb_output: bpy.types.ShaderNodeOutputMaterial = tree.nodes["Material Output"]  # type: ignore
        rgb_output.target = "CYCLES"
        bsdf = tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # type: ignore
        bsdf.inputs["Roughness"].default_value = 0.5  # type: ignore
        bsdf.inputs["Metallic"].default_value = 0.0  # type: ignore
        tree.links.new(bsdf.outputs["BSDF"], rgb_output.inputs["Surface"])

        hsv_node: bpy.types.ShaderNodeHueSaturation = tree.nodes.new("ShaderNodeHueSaturation")  # type: ignore
        hsv_node.name = "sp_hsv"
        hsv_node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # type: ignore
        hsv_node.inputs["Saturation"].default_value = 0.0  # type: ignore
        hsv_node.inputs["Value"].default_value = 1.0  # type: ignore
        hsv_node.location = (-300, 0)
        tree.links.new(hsv_node.outputs["Color"], bsdf.inputs["Base Color"])

        # image texture
        image_node: bpy.types.ShaderNodeTexImage = tree.nodes.new("ShaderNodeTexImage")  # type: ignore
        image_node.name = "sp_image"
        image_node.location = (-600, 0)
        tree.links.new(image_node.outputs["Color"], hsv_node.inputs["Color"])

        Plane.add_ID_to_material(material)

        # initialize object
        obj = Plane(bl_object)
        obj.set_hue(0.5, set_default=True)
        obj.set_saturation(1.0, set_default=True)
        obj.set_value(1.0, set_default=True)
        obj.set_metallic(0.0, set_default=True)
        obj.set_roughness(0.5, set_default=True)

        return obj

    def set_image(self, filepath: Path):
        import bpy

        image_node: bpy.types.ShaderNodeTexImage = self.materials[0].node_tree.nodes["sp_image"]  # type: ignore
        if image_node.image is not None:
            bpy.data.images.remove(image_node.image)
        self.img = bpy.data.images.load(str(filepath.expanduser().resolve()))
        image_node.image = self.img
