import logging, coloredlogs

coloredlogs.install(logging.DEBUG, fmt="%(asctime)s %(levelname)s %(message)s")

import simpose as sp
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
from itertools import cycle


scene = sp.Scene(512, 512)
scene.set_gravity((0.0, 0.0, 0.0))

writer = sp.Writer(scene, Path("duck_animation"))

rand_lights = sp.LightRandomizer(
    scene,
    no_of_lights_range=(2, 5),
    energy_range=(100, 500),
    color_range=(0.9, 1.0),
    distance_range=(3.0, 10.0),
)

cam = scene.create_camera("Camera")
cam.set_location((0.0, -1.5, 0.0))
cam.set_rotation(R.from_euler("x", -90, degrees=True))

l1 = scene.create_light("light1", 500.0, type="POINT")
l1.set_location((2.0, -2.0, 1.0))

l2 = scene.create_light("light2", 300.0, type="POINT")
l2.set_location((-2.0, -2.0, 0.0))

l3 = scene.create_light("light3", 1000.0, type="POINT")
l3.set_location((-2.0, 2.0, 0.0))


obj_path = Path("meshes/cpsduck/cpsduck.obj")
duck = scene.create_from_obj(obj_path)

duck.set_metallic_value(0.0)
duck.set_roughness_value(0.5)

n_ducks = 200
ducks = [duck]
ducks.extend([scene.create_copy(duck, linked=True) for _ in range(n_ducks - 1)])

z_step = 0.1
n_bins = 8
z_vals = np.random.uniform(-0.8, -0.5, size=(n_bins,))
x_vals = np.linspace(-0.5, 0.5, n_bins, endpoint=True)
y_vals = np.random.uniform(-0.1, 0.2, size=(n_bins,))

for idx, duck in enumerate(iter(ducks)):
    bin_i = idx % n_bins

    x = x_vals[bin_i]
    z = z_vals[bin_i]
    y = y_vals[bin_i]
    duck.set_location((x, y, z))
    z_vals[bin_i] += z_step

    duck.set_rotation(R.from_euler("xz", [-90, np.random.uniform(-180, 180)], degrees=True))

    if np.random.uniform() < 0.2:
        z_vals[bin_i] += z_step * 2

rots = [R.from_euler("z", np.random.choice([-45, 45]), degrees=True) for _ in range(n_ducks)]

scene.export_blend(str(Path("scene.blend").resolve()))
for i in trange(20):
    for duck, rot in zip(ducks, rots):
        down = np.array(duck.location) + np.array((0.0, 0.0, -0.05))
        duck.set_location(down)
        duck.set_rotation(rot * duck.rotation)

    writer.generate_data(i)  # Save the rendered image to the specified file path
