import bpy
from .placeable import Placeable


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
        self.light_data.energy = energy