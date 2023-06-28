import bpy
from .placeable import Placeable
import logging
import numpy as np
import math
from mathutils import Vector
class Light(Placeable):
    def __init__(self, name, type="POINT", energy=1.0):
        self.name = name
        self.type = type
        self.energy = energy

        self.light_data = bpy.data.lights.new(name=self.name, type=self.type)
        self.light_data.energy = self.energy

        self.light_object = bpy.data.objects.new(
            name=self.name, object_data=self.light_data
        )
        self.light_object.name = name
        bpy.context.collection.objects.link(self.light_object)

        super().__init__(bl_object=self.light_object)

    def set_energy(self, energy):
        logging.info(f"Setting energy of light {self.name} to {energy}")
        self.light_data.energy = energy
        
    def randomize_position_rel_to_camera(self, cam,d_lights):

        r = np.random.uniform(d_lights[0], d_lights[1])
        
        elev = np.random.uniform(-np.pi*0.75, np.pi*0.75)
        yaw = np.random.uniform(np.pi/4, 7./4*np.pi)
        
        x = math.sin(yaw) * math.cos(elev) * r
        z = math.cos(yaw) * math.cos(elev) * r
        y = math.sin(elev) * r
        
        cam_pos = cam.location
        cam_rot = cam.rotation
        pos = (cam_rot.as_matrix() @ Vector((x, y, z))) + cam_pos
        return pos
    
    def set_color(self, color):
        self.light_data.color = color
