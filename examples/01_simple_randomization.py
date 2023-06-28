import simpose as sp
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm, trange

logging.basicConfig(level=logging.WARN)

scene = sp.Scene()

writer = sp.Writer(scene, Path("output_01"))

rand_lights = sp.LightRandomizer(
    scene,
    no_of_lights_range=(2, 5),
    energy_range=(100, 500),
    color_range=(0.9, 1.0),
    distance_range=(3.0, 10.0)
)

rand_scene = sp.SceneRandomizer(
    scene=scene,
    backgrounds_dir=Path("backgrounds")
)

rand_obj = sp.ObjectRandomizer(
    r_range = (0.3, 1.0)
)

for obj_path in Path("meshes").glob("*/*.obj"):
    new = scene.create_from_obj(obj_path)
    new.set_metallic_value(1.0)
    new.set_roughness_value(0.5)
    rand_obj.add(new)
cam = scene.create_camera("Camera")


for i in trange(10):
    rand_scene.randomize_background()
    rand_lights.randomize_lighting_around_cam(cam)
    rand_obj.randomize_in_camera_frustum(cam)
    rand_obj.randomize_orientation()
    writer.generate_data(i)  # Save the rendered image to the specified file path


print(cam.get_calibration_matrix_K_from_blender())
    
# export Scene as .blend file, so we can open it in Blender and check results
scene.export_blend(str(Path("scene.blend").resolve()))
