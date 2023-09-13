import bpy
from scipy.spatial.transform import Rotation as R
from .placeable import Placeable
import mathutils
import numpy as np
import logging


class Camera(Placeable):
    """This is just a functional wrapper around the blender object.
    It is not meant to be instantiated directly. Use the factory methods of
        simpose.Scene instead
    It has no internal state, everything is delegated to the blender object.

    In this case, a camera is an 'Empty' Blender Object with one (or two) cameras attached to it.
    """

    def __init__(self, bl_cam):
        super().__init__(bl_object=bl_cam)

    @staticmethod
    def create(
        name: str,
        baseline: float | None = None,
    ):
        # create empty blender
        bpy.ops.object.empty_add(type="PLAIN_AXES")
        frame = bpy.context.selected_objects[0]
        frame.name = name

        # create camera
        bpy.ops.object.camera_add()
        bl_cam = bpy.context.selected_objects[0]
        bl_cam.name = name + "_L"
        # rotate by 180 degrees around x to follow OpenCV convention
        # blender: scalar first, scipy: scalar last
        r = R.from_euler("x", -180, degrees=True).as_quat(canonical=False)
        blender_quat = [r[3], r[0], r[1], r[2]]
        bl_cam.rotation_mode = "QUATERNION"
        bl_cam.rotation_quaternion = blender_quat
        bl_cam.location = mathutils.Vector([0, 0, 0])
        bl_cam.parent = frame

        # set scene's active camera to the left one
        bpy.context.scene.camera = bl_cam

        if baseline is not None:
            bpy.ops.object.camera_add()
            bl_cam_right = bpy.context.selected_objects[0]
            bl_cam_right.name = name + "_R"

            bl_cam_right.location = mathutils.Vector([baseline, 0, 0])
            bl_cam_right.rotation_mode = "QUATERNION"
            bl_cam_right.rotation_quaternion = blender_quat
            bl_cam_right.parent = frame

            frame["sp_baseline"] = baseline
        return Camera(frame)

    @property
    def name(self) -> str:
        return self._bl_object.name

    @property
    def baseline(self) -> float:
        return self._bl_object["sp_baseline"]

    @property
    def left_camera(self) -> bpy.types.Object:
        child = None
        for child in self._bl_object.children:
            if child.name == self.name + "_L":
                break
        assert child is not None
        return child  # type: ignore

    @property
    def right_camera(self) -> bpy.types.Object:
        child = None
        for child in self._bl_object.children:
            if child.name == self.name + "_R":
                break
        assert child is not None, "Right camera not found. Did you create a stereo camera?"
        return child

    @property
    def data(self) -> bpy.types.Camera:
        return self.left_camera.data  # type: ignore

    def is_stereo_camera(self) -> bool:
        return self.name + "_R" in [x.name for x in self._bl_object.children]

    def __str__(self) -> str:
        return f"Camera(name={self._bl_object.name})"

    def point_at(self, location: np.ndarray):
        """point camera towards location with z-axis pointing up"""
        to_point = self.location - location
        yaw = np.arctan2(to_point[1], to_point[0]) + np.pi / 2
        pitch = -np.pi / 2 - np.arctan2(to_point[2], np.linalg.norm(to_point[:2]))
        towards_origin = R.from_euler("ZYX", [yaw, 0.0, pitch])
        self.set_rotation(towards_origin)

    def calculate_intrinsics(self, img_w: int, img_h: int) -> np.ndarray:
        return self._calculate_intrinsics(self.data, img_w, img_h)

    @staticmethod
    def _calculate_intrinsics(cam_data: bpy.types.Camera, img_w, img_h):
        cam_data.lens_unit = "MILLIMETERS"  # switch to focal length
        cam_data.sensor_fit = (
            "HORIZONTAL"  # sensor width is fixed, height is variable, depending on aspect ratio
        )

        f_in_mm = cam_data.lens
        sensor_width_in_mm = cam_data.sensor_width
        sensor_height_in_mm = sensor_width_in_mm / img_w * img_h

        scale_x = img_w / sensor_width_in_mm
        scale_y = img_h / sensor_height_in_mm

        # Parameters of intrinsic calibration matrix K
        alpha_u = f_in_mm * scale_x
        alpha_v = f_in_mm * scale_y
        u_0 = img_w * (0.5 - cam_data.shift_x)
        v_0 = img_h * (0.5 + cam_data.shift_y)  # because flipped blender camera

        return np.array([[alpha_u, 0.0, u_0], [0, alpha_v, v_0], [0, 0, 1]])

    def set_from_hfov(self, hfov, img_w, img_h, degrees: bool = False):
        """Set camera intrinsics from horizontal field of view"""
        if degrees:
            hfov = np.deg2rad(hfov)
        self._set_from_hfov(self.data, hfov, img_w, img_h)
        if self.is_stereo_camera():
            self._set_from_hfov(self.right_camera.data, hfov, img_w, img_h)  # type: ignore

    @staticmethod
    def _set_from_hfov(cam_data: bpy.types.Camera, hfov, img_w, img_h):
        """Set camera intrinsics from horizontal field of view"""
        cam_data.sensor_fit = (
            "HORIZONTAL"  # sensor width is fixed, height is variable, depending on aspect ratio
        )
        cam_data.lens_unit = "FOV"
        cam_data.angle = hfov
        cam_data.shift_x = 0.0
        cam_data.shift_y = 0.0

    def set_from_intrinsics(self, intrinsic_matrix: np.ndarray, img_w: int, img_h: int):
        if np.abs(intrinsic_matrix[1][1] - intrinsic_matrix[0][0]) > 0.001:
            logging.warning(
                "Intrinsic matrix is not symmetric. At the moment only square pixels are supported."
            )

        logging.info(f"Setting {self} intrinsics to {intrinsic_matrix}, (h,w): ({img_h}, {img_w})")
        self._set_intrinsics(self.data, intrinsic_matrix, img_w, img_h)
        if self.is_stereo_camera():
            self._set_intrinsics(self.right_camera.data, intrinsic_matrix, img_w, img_h)  # type: ignore

    @staticmethod
    def _set_intrinsics(
        cam_data: bpy.types.Camera, intrinsic_matrix: np.ndarray, img_w: int, img_h: int
    ):
        cam_data.lens_unit = "MILLIMETERS"  # switch to focal length
        cam_data.sensor_fit = (
            "HORIZONTAL"  # sensor width is fixed, height is variable, depending on aspect ratio
        )

        sensor_width_in_mm = cam_data.sensor_width

        f_x = intrinsic_matrix[0][0]
        c_x, c_y = intrinsic_matrix[0][2], intrinsic_matrix[1][2]
        scale_x = img_w / sensor_width_in_mm

        shift_x = c_x / img_w - 0.5
        shift_y = 0.5 - c_y / img_h  # inverse because in blender y is flipped

        cam_data.shift_x = shift_x
        cam_data.shift_y = shift_y
        cam_data.lens = f_x / scale_x  # f_in_mm
        # np.testing.assert_almost_equal(
        #     intrinsic_matrix, Camera._calculate_intrinsics(cam_data, img_w, img_h), decimal=2
        # )
