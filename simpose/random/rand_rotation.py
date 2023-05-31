from scipy.spatial.transform import Rotation as R
import bpy

def randomize_rotation(subject: bpy.types.Object):
    subject.rotation_mode = "XYZ"
    subject.rotation_euler = R.random().as_euler("xyz")