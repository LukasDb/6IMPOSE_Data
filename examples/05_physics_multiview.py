import simpose as sp
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.INFO)

scene = sp.Scene()

writer = sp.Writer(scene, Path("output_05"))

cam = scene.create_camera("Camera")
rand_lights = sp.random.LightRandomizer(
    scene,
    cam,
    sp.CallbackType.BEFORE_RENDER,
    no_of_lights_range=(3, 6),
    energy_range=(300, 1000),
    color_range=(0.8, 1.0),
    distance_range=(3.0, 10.0),
)

rand_scene = sp.random.BackgroundRandomizer(
    scene, sp.CallbackType.BEFORE_RENDER, backgrounds_dir=Path("backgrounds")
)

obj_path = Path("meshes/cpsduck/cpsduck.obj")
duck = scene.create_from_obj(obj_path, add_physics=True, mass=0.2, friction=1.0)
duck.set_metallic_value(0.0)
duck.set_roughness_value(0.5)

ducks = [duck]

for i in range(20):
    ducks.append(scene.create_copy(duck, linked=True))


# data generation params
dt = 1 / 5.0  # 5 FPS
total_len = 20  # for 20 seconds
num_drops = 50
num_dt_step = int(total_len / dt)
num_cam_locs = 10
print(f"Generating {num_dt_step * num_cam_locs * num_drops} images")

i = 0
bar = tqdm(total=num_drops * num_dt_step * num_cam_locs)
for run in range(num_drops):  # 50 different drops
    for j, duck in enumerate(ducks):
        duck.set_location(
            (
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.05, 0.05),
                j * 0.1 + 0.1,
            )
        )
        duck.set_rotation(R.random())

    # let initiall fall for 1 second
    scene.step_physics(1.0)

    for _ in range(num_dt_step):
        scene.step_physics(dt)

        # sample 20 camera locations in upper hemisphere (uniformly)
        rots = R.random(num=num_cam_locs)
        cam_view = np.array([0.0, 0.0, 1.0])
        rads = np.random.uniform(0.5, 1.0, size=(num_cam_locs,))

        cam_locations = rots.apply(cam_view) * rads[:, None]
        cam_locations[:, 2] *= np.sign(cam_locations[:, 2])
        cam_locations[:, 2] += 0.2

        for cam_location in cam_locations:
            cam.set_location(cam_location)
            cam.point_at(np.array([0.0, 0.0, 0.0]))

            cam.apply_local_rotation_offset(
                R.from_euler("z", np.random.uniform(-5, 5), degrees=True)
            )

            writer.generate_data(i)
            i += 1
            bar.update(1)

bar.close()


# export Scene as .blend file, so we can open it in Blender and check results
scene.export_blend()
