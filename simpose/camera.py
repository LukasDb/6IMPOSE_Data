import bpy
from scipy.spatial.transform import Rotation as R
from .placeable import Placeable
import mathutils
import numpy as np


class Camera(Placeable):
    """This is just a functional wrapper around the blender object.
    It is not meant to be instantiated directly. Use the factory methods of
        simpose.Scene instead
    It has no internal state, everything is delegated to the blender object.
    """

    def __init__(self, bl_cam):
        super().__init__(bl_object=bl_cam)
        bpy.context.scene.camera = bl_cam

    @staticmethod
    def create(name: str):
        bpy.ops.object.camera_add()
        bl_cam = bpy.context.selected_objects[0]
        bl_cam.name = name
        return Camera(bl_cam)

    @property
    def name(self) -> str:
        return self._bl_object.name

    @property
    def data(self) -> bpy.types.Camera:
        return self._bl_object.data  # type: ignore

    @property
    def rotation(self) -> R:
        """returns orientation of camera in OpenCV format"""
        orn = super().rotation
        to_cv2 = R.from_euler("x", 180, degrees=True)
        return orn * to_cv2

    def set_rotation(self, rotation: R):
        """sets orientation of camera in OpenCV format"""
        to_blender = R.from_euler("x", -180, degrees=True)
        super().set_rotation(rotation * to_blender)

    def __str__(self) -> str:
        return f"Camera(name={self._bl_object.name})"

    def point_at(self, location: np.ndarray):
        """point camera towards location with z-axis pointing up"""
        to_point = self.location - location
        yaw = np.arctan2(to_point[1], to_point[0]) + np.pi / 2
        pitch = -np.pi / 2 - np.arctan2(to_point[2], np.linalg.norm(to_point[:2]))
        towards_origin = R.from_euler("ZYX", [yaw, 0.0, pitch])
        self.set_rotation(towards_origin)

    def get_calibration_matrix_K_from_blender(self):
        # always assume square pixels

        f_in_mm = self.data.lens
        scene = bpy.context.scene
        resolution_x_in_px = scene.render.resolution_x
        resolution_y_in_px = scene.render.resolution_y
        scale = scene.render.resolution_percentage / 100
        sensor_width_in_mm = self.data.sensor_width
        sensor_height_in_mm = self.data.sensor_height

        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        # s_v = resolution_y_in_px * scale / sensor_height_in_mm # wrong for some reason
        pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
        s_v = s_u * pixel_aspect_ratio

        # Parameters of intrinsic calibration matrix K
        alpha_u = f_in_mm * s_u
        alpha_v = f_in_mm * s_v
        u_0 = resolution_x_in_px * (0.5 - self.data.shift_x)
        v_0 = resolution_y_in_px * (0.5 + self.data.shift_y)  # because flipped blender camera
        skew = 0  # only use rectangular pixels

        K = mathutils.Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
        return K

    # def create_camera(self,intrinsic_matrix,resolution_x,resolution_y):
    #     # Create a new camera object
    #     camera_data = bpy.data.cameras.new("CustomCamera")
    #     camera_object = bpy.data.objects.new("CustomCamera", camera_data)
    #     bpy.context.collection.objects.link(camera_object)

    #     # Set the camera resolution
    #     bpy.context.scene.render.resolution_x = resolution_x
    #     bpy.context.scene.render.resolution_y = resolution_y

    #     # Set the camera matrix transform
    #     camera_object.matrix_world = intrinsic_matrix.to_4x4()  # Set the intrinsic matrix as the camera transform

    #     camera_object.data.lens_unit = 'FOV'  # Set the lens unit to Field of View
    #     camera_object.data.angle = 2 * atan(intrinsic_matrix[][0] / (2 * intrinsic_matrix[0][0]))  # Set the field of view angle
