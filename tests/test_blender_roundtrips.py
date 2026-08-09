"""Round trips through Blender and PyBullet: camera intrinsics, object poses
(including the COM offset algebra), quaternion conventions, and the camera
fill-ratio geometry. Requires the bpy module; skipped otherwise."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

bpy = pytest.importorskip("bpy")

import simpose as sp

from helpers import write_cube_obj

# level < DEBUG makes sp.redirect_stdout a no-op; its fd-level redirection would
# otherwise fight pytest's output capture
sp.logger.setLevel(1)

IMG_W, IMG_H = 1920, 1080


@pytest.fixture(scope="module")
def scene():
    return sp.Scene(img_h=IMG_H, img_w=IMG_W)


@pytest.fixture(scope="module")
def camera(scene):
    cam = scene.create_camera("TestCam")
    cam.set_from_hfov(70.0, IMG_W, IMG_H, degrees=True)
    return cam


@pytest.fixture(scope="module")
def cube(scene, tmp_path_factory):
    # vertices span 0..1, so the center of volume is NOT the mesh origin and the
    # COM offset handling between Blender and PyBullet is actually exercised
    obj_path = write_cube_obj(tmp_path_factory.mktemp("models"), offset=0.0)
    return scene.create_object(obj_path, "cube", mass=1.0, friction=0.5)


def test_hfov_intrinsics(camera):
    K = camera.calculate_intrinsics(IMG_W, IMG_H)
    f_expected = (IMG_W / 2) / np.tan(np.deg2rad(70.0) / 2)
    np.testing.assert_allclose(K[0, 0], f_expected, rtol=1e-5)
    np.testing.assert_allclose(K[1, 1], f_expected, rtol=1e-5)
    np.testing.assert_allclose(K[0, 2], IMG_W / 2, atol=1e-3)
    np.testing.assert_allclose(K[1, 2], IMG_H / 2, atol=1e-3)


def test_intrinsics_roundtrip_off_center(camera):
    K = np.array(
        [
            [1050.0, 0.0, 980.0],  # principal point off-center in both axes
            [0.0, 1050.0, 502.0],
            [0.0, 0.0, 1.0],
        ]
    )
    camera.set_from_intrinsics(K, IMG_W, IMG_H)
    K_back = camera.calculate_intrinsics(IMG_W, IMG_H)
    np.testing.assert_allclose(K_back, K, atol=0.01)
    # restore the module-scoped camera for other tests
    camera.set_from_hfov(70.0, IMG_W, IMG_H, degrees=True)


def test_pose_roundtrip_blender_pybullet(scene, cube):
    location = (0.15, -0.2, 0.5)
    rotation = R.from_euler("xyz", (25, -40, 130), degrees=True)

    cube.set_location(location)
    cube.set_rotation(rotation)

    # read the pose back out of PyBullet; without stepping, the COM offset
    # algebra must be a perfect inverse pair
    cube.apply_pybullet_pose()

    np.testing.assert_allclose(np.array(cube.location), location, atol=1e-5)
    delta = (cube.rotation.inv() * rotation).magnitude()
    assert delta < 1e-5, f"rotation changed by {np.rad2deg(delta)} deg in the round trip"


def test_quaternion_convention_roundtrip(scene, cube):
    quat_xyzw = R.from_euler("zyx", (10, 20, 30), degrees=True).as_quat(canonical=False)
    cube.set_rotation(R.from_quat(quat_xyzw))
    out = cube.rotation.as_quat(canonical=False)
    # q and -q encode the same rotation
    assert abs(np.dot(out, quat_xyzw)) > 1 - 1e-6


def test_vhacd_cache_written_atomically(cube, tmp_path_factory):
    mesh_dirs = [p for p in tmp_path_factory.getbasetemp().glob("models*")]
    vhacd_files = [f for d in mesh_dirs for f in d.glob("*_vhacd.obj")]
    assert any(f.name == "cube_vhacd.obj" for f in vhacd_files), "V-HACD cache missing"
    leftovers = [f for f in vhacd_files if ".tmp" in f.name]
    assert leftovers == [], f"temporary V-HACD files left behind: {leftovers}"


def test_camera_fill_ratio(scene, camera, cube):
    fill = 0.3
    config = sp.random.CameraSceneRandomizerConfig(
        trigger=sp.observers.Event.NONE,
        origin=(0.0, 0.0, 0.0),
        pitch_range=(20.0, 88.0),
        fill_range=(fill, fill),
        roll_jitter=0.0,
        pitch_jitter=0.0,
        yaw_jitter=0.0,
        min_cam2obj_distance=0.01,
    )
    randomizer = sp.random.CameraSceneRandomizer(config)
    randomizer.call(scene)

    max_diameter = max(o.get_diameter() for o in scene.get_active_objects())
    dist = np.linalg.norm(np.array(camera.location))
    recovered_fill = 2 * np.arctan(max_diameter / 2 / dist) / camera.hfov
    np.testing.assert_allclose(recovered_fill, fill, atol=0.01)
