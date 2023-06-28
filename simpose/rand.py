import bpy
import numpy as np
import simpose
import numpy as np
from pathlib import Path
from mathutils import Vector
import logging
from scipy.spatial.transform import Rotation as R
from .placeable import Placeable


class Random:
    def randomize_lighting(no_of_lights,cam,d_lights,energy_range):
        for key in bpy.data.lights:
            bpy.data.lights.remove(key, do_unlink=True)
                
        n_lights  = np.random.randint(no_of_lights[0],no_of_lights[1]+1)
        for i in range(n_lights):
            Energy = np.random.uniform(energy_range[0],energy_range[1])
            light = simpose.Light(f"Light_{i}", type="POINT", energy=Energy)
            light.set_location(light.randomize_position_rel_to_camera(cam,d_lights))
            light.set_color((np.random.uniform(0.0,1.0),np.random.uniform(0.0,1.0),np.random.uniform(0.0,1.0)))
            
    def random_background():
        bg = "//" + str(np.random.choice(list(Path("backgrounds").glob("*.jpg"))))
        return bg
    
    def randomize_in_camera_frustum(
        subject: simpose.Object,
        cam: simpose.Camera,
        r_range,
        yp_limit=(0.9, 0.9),
    ):
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

        pos = np.array([x, y, z]) @ cam_rot.as_matrix() + np.array(cam_origin)

        subject.set_location(pos)
        logging.info(
            f"randomize_in_camera_frustum: {subject} randomzied to {pos}"
        )
        return np.array([x, y, z])
    
    def randomize_rotation(subject: Placeable):
        subject.set_rotation(R.random())
