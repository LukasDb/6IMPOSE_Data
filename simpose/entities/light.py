from scipy.spatial.transform import Rotation as R
from .placeable import Placeable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bpy


class Light(Placeable):
    """This is just a functional wrapper around the blender object.
    It is not meant to be instantiated directly. Use the factory methods of
        simpose.Scene instead
    It has no internal state, everything is delegated to the blender object.
    """

    TYPE_POINT = "POINT"
    TYPE_SUN = "SUN"
    TYPE_SPOT = "SPOT"
    TYPE_AREA = "AREA"

    def __init__(self, bl_light):
        super().__init__(bl_object=bl_light)

    @staticmethod
    def create(name: str, energy, type="POINT"):
        import bpy

        light_data = bpy.data.lights.new(name=name, type=type)
        light_data.energy = energy  # type: ignore

        bl_light = bpy.data.objects.new(name=name, object_data=light_data)
        bl_light.name = name
        return Light(bl_light)

    def remove(self):
        import bpy

        try:
            light_data = self.light_data
        except ReferenceError:
            return
        finally:
            bpy.data.objects.remove(self._bl_object)  # remove 'container'

        if light_data.users == 0:
            # first, remove mesh data
            bpy.data.lights.remove(light_data, do_unlink=True)

    def set_rotation(self, rotation: R):
        """set rotation, with z+ pointing"""
        rot = rotation * R.from_euler("x", 180, degrees=True)
        return super().set_rotation(rot)

    @property
    def light_data(
        self,
    ) -> "bpy.types.PointLight | bpy.types.SpotLight | bpy.types.SunLight | bpy.types.AreaLight":
        return self._bl_object.data  # type: ignore

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

    @property
    def size(self):
        import bpy

        assert isinstance(self.light_data, bpy.types.AreaLight), "Not an area light"
        return self.light_data.size

    @size.setter
    def size(self, size):
        import bpy

        assert isinstance(self.light_data, bpy.types.AreaLight), "Not an area light"
        self.light_data.size = size