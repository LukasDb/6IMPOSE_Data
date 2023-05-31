import bpy

class Object:
    def __init__(self, bl_object, object_id: int) -> None:
        self._bl_object = bl_object # reference to internal blender object
        self.object_id = object_id

    @staticmethod
    def from_obj(filepath, object_id:int):
        bpy.ops.wm.obj_import(filepath=filepath)
        return Object(bpy.context.selected_objects[0], object_id=object_id)