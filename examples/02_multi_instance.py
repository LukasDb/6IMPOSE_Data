import simpose as sp
from pathlib import Path
import logging
import numpy as np
import tqdm

logging.basicConfig(level=logging.WARN)

scene = sp.Scene()

writer = sp.Writer(scene, Path("output_02"))

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

rand_obj = sp.random.CameraFrustumRandomizer(
    scene, cam, sp.CallbackType.BEFORE_RENDER, r_range=(0.3, 1.0)
)


obj_path = Path("meshes/cpsduck/cpsduck.obj")
duck = scene.create_object(obj_path, add_semantics=True)
duck.set_metallic_value(0.0)
duck.set_roughness_value(0.5)
rand_obj.add(duck)

obj_path = Path("meshes/wrench_13/wrench_13.obj")
wrench = scene.create_object(obj_path, add_semantics=True)
wrench.set_metallic_value(1.0)
wrench.set_roughness_value(0.1)
rand_obj.add(wrench)

for i in range(50):
    duck_copy = scene.create_copy(duck)
    rand_obj.add(duck_copy)

scene.export_blend(str(Path("scene.blend").resolve()))

for i in tqdm.trange(10):
    writer.generate_data(i)  # Save the rendered image to the specified file path

print(cam.get_calibration_matrix_K_from_blender())

# export Scene as .blend file, so we can open it in Blender and check results
scene.export_blend(str(Path("scene.blend").resolve()))
