import simpose as sp
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm, trange

logging.basicConfig(level=logging.WARN)

scene = sp.Scene()

# use context manager to explicitly specify the blender scene, where the objects are created
obj = []
with scene:
    obj_path = Path("meshes/cpsduck/cpsduck.obj")
    duck = sp.Object.from_obj(str(obj_path.resolve()))
    obj.append(duck)
    duck.set_metallic_value(0.0)
    duck.set_roughness_value(0.5)

    obj_path = Path("meshes/wrench_13/wrench_13.obj")
    wrench = sp.Object.from_obj(str(obj_path.resolve()))
    obj.append(wrench)
    wrench.set_metallic_value(1.0)
    wrench.set_roughness_value(0.1)

    for i in range(50):
        obj.append(duck.copy(linked=True))

    cam = sp.Camera("Camera")
    light = sp.Light("Light", type="POINT", energy=100.0)
    light2 = sp.Light("Light2", type="POINT", energy=100.0)

    random = sp.Random
    bg = random.random_background()
    scene.set_background(bg)
    

light.set_location((1.0, 1.0, -0.2))
light2.set_location((-1.0, -1.0, -0.2))


for i in trange(10):
    bg = random.random_background()
    scene.set_background(bg)
    for j in obj:
        random.randomize_rotation(j)
        random.randomize_in_camera_frustum(j, cam, (0.3, 1.0), (0.9, 0.9))

    random.randomize_lighting((3,6),cam,(1.0,3.0),(40,100))

    scene.generate_data("render/gt/",obj,cam,i)  # Save the rendered image to the specified file path

print(cam.get_calibration_matrix_K_from_blender())
    
# export Scene as .blend file, so we can open it in Blender and check results
scene.export_blend(str(Path("scene.blend").resolve()))
