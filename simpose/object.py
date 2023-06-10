import bpy
from .materials.common import PrincipledBSDFMaterial
class Object:
    def __init__(self, bl_object, object_id: int) -> None:
        self._bl_object = bl_object # reference to internal blender object
        self.object_id = object_id

    @staticmethod
    def from_obj(filepath, object_id:int):
        bpy.ops.wm.obj_import(filepath=filepath)
        return Object(bpy.context.selected_objects[0], object_id=object_id)
    @staticmethod
    def add_material(self):
         # Create the material instance
        material = PrincipledBSDFMaterial(metallic=0.5, roughness=0.2)
        # Create the Blender material and shader node
        blender_material, shader_node = material.create_material(name="MyMaterial")
        self._bl_object.data.materials.clear()  # Clear existing materials if needed
        self._bl_object.data.materials.append(blender_material)
