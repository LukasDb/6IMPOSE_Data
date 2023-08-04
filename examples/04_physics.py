import simpose as sp
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm, trange
from scipy.spatial.transform import Rotation as R

logging.basicConfig(level=logging.WARN)

scene = sp.Scene()

writer = sp.Writer(scene, Path("output_04"))

cam = scene.create_camera("Camera")
cam.set_location((0.0, -0.7, 0.2))
cam.set_rotation(R.from_euler("x", -100, degrees=True))

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
duck = scene.create_from_obj(obj_path, add_physics=True, mass=0.2)
duck.set_metallic_value(0.0)
duck.set_roughness_value(0.5)


ducks = [duck]

for i in range(20):
    ducks.append(scene.create_copy(duck, linked=True))

for i, duck in enumerate(ducks):
    duck.set_location(
        (np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), i * 0.1 + 0.1)
    )
    duck.set_rotation(R.random())

# scene.export_blend(str(Path("scene.blend").resolve()))

dt = 1 / 24.0
total_len = 4
for i in trange(int(4 / dt)):
    scene.step_physics(dt)
    writer.generate_data(i)  # Save the rendered image to the specified file path

# export Scene as .blend file, so we can open it in Blender and check results
scene.export_blend(str(Path("scene.blend").resolve()))
