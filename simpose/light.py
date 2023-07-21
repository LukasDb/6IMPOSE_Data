import bpy
from .placeable import Placeable
import logging
import numpy as np
import math
from mathutils import Vector


class Light(Placeable):
    """ This is just a functional wrapper around the blender object.
    It is not meant to be instantiated directly. Use the factory methods of 
        simpose.Scene instead
    It has no internal state, everything is delegated to the blender object.
    """
    def __init__(self, bl_light):
        super().__init__(bl_object=bl_light)

    @staticmethod
    def create(name: str, energy, type="POINT"):
        light_data = bpy.data.lights.new(name=name, type=type)
        light_data.energy = energy

        bl_light = bpy.data.objects.new(
            name=name, object_data=light_data
        )
        bl_light.name = name
        return Light(bl_light)
    

    @property
    def light_data(self):
        return self._bl_object.data
    

    @property
    def energy(self):
        return self.light_data.energy

    @energy.setter
    def energy(self, energy):
        self.light_data.energy = energy
    
    @property
    def color(self):
        return self.light_data.color
    
    @color.setter
    def color(self, color):
        self.light_data.color = color


    