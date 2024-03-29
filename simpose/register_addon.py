# this file will be run in Blender to install the GUI elements of 6IMPOSE
import sys
import bpy


class SimposePanel(bpy.types.Panel):
    bl_idname = "SimposePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "6IMPOSE"
    bl_label = "6IMPOSE"

    def draw(self, _: bpy.types.Context) -> None:
        col = self.layout.column(align=True)
        for op in sp.BL_OPS:
            col.operator(op.bl_idname, text=op.bl_label)


def register() -> None:
    print("Registering 6IMPOSE GUI...")
    for op in sp.BL_OPS:
        bpy.utils.register_class(op)
    bpy.utils.register_class(SimposePanel)


def unregister() -> None:
    for op in sp.BL_OPS:
        bpy.utils.unregister_class(op)
    bpy.utils.unregister_class(SimposePanel)


if __name__ == "__main__":
    # TODO fix this!
    additional_paths = [
        "/Users/ldirnberger/micromamba/envs/blender/lib/python3.10/site-packages",
        "/Users/ldirnberger/dev/6IMPOSE_Data",
    ]
    sys.path.extend(additional_paths)

    # re-import simpose to reflect changes for developping
    mods = list(sys.modules.keys())
    for mod in mods:
        if mod.startswith("simpose"):
            del sys.modules[mod]
    import simpose as sp

    register()
