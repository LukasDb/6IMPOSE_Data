import simpose as sp
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm, trange

logging.basicConfig(level=logging.INFO)

scene = sp.Scene()

# use context manager to explicitly specify the blender scene, where the objects are created
obj = []
with scene:
    for i, obj_path in enumerate(Path("meshes").glob("*/*.obj")):
        new = sp.Object.from_obj(
            str(obj_path.resolve()),
            object_id=i,
        )
        obj.append(new)
        new.set_metallic_value(1.0)
        new.set_roughness_value(0.5)
    cam = sp.Camera("Camera")
    random = sp.Random
    bg = random.random_background()
    scene.set_background(bg)
    
for i in trange(10):
    bg = random.random_background()
    scene.set_background(bg)
    for j in obj:
        random.randomize_rotation(j)
        random.randomize_in_camera_frustum(j, cam, (0.3, 1.0), (0.9, 0.9))
    random.randomize_lighting((3,6),cam,(1.0,3.0),(40,100))
    # Render the scene
    scene.generate_data("render/gt/",obj,cam,i)  # Save the rendered image to the specified file path
print(cam.get_calibration_matrix_K_from_blender())
    
# export Scene as .blend file, so we can open it in Blender and check results
scene.export_blend(str(Path("scene.blend").resolve()))
