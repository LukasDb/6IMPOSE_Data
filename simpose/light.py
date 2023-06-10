


import bpy

class Light:
    def __init__(self, name, type='POINT', energy=1.0, location=(0, 0, 0)):
        self.name = name
        self.type = type
        self.energy = energy
        self.location = location
        self.light_data = None
        self.light_object = None
        
    def create(self):
        # Create a new light data block
        self.light_data = bpy.data.lights.new(name=self.name, type=self.type)

        # Create a new light object and link it to the scene
        self.light_object = bpy.data.objects.new(name=self.name, object_data=self.light_data)
        bpy.context.collection.objects.link(self.light_object)

        # Set the light object's location
        self.light_object.location = self.location

        # Set light properties
        self.light_data.energy = self.energy
