import bpy
from mathutils import Vector
import numpy as np
import simpose

def randomize_in_camera_frustum(subject: bpy.types.Object, cam: simpose.Camera, r_range, yp_limit=(0.9, 0.9)):
    render = bpy.context.scene.render

    aspect_ratio = render.resolution_x / render.resolution_y

    hfov = cam._bl_object.data.angle_x / 2.0
    vfov = cam._bl_object.data.angle_x / 2.0 / aspect_ratio

    # min_fov = min(hfov, vfov)

    r = np.random.uniform(r_range[0], r_range[1])
    yaw = yp_limit[0] * np.random.uniform(-hfov, hfov)
    pitch = yp_limit[1] * np.random.uniform(-vfov, vfov)

    x = np.cos(pitch) * np.sin(yaw) * r
    y = np.sin(pitch) * r
    z = np.cos(pitch) * np.cos(yaw) * r

    cam_origin = cam.location
    cam_rot = cam.rotation
    pos = cam_rot.apply([x, y, z]) + cam_origin
    subject.location = pos
    return np.array([x, y, z])