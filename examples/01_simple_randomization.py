import simpose as sp
from pathlib import Path
import logging
from tqdm import tqdm
import multiprocessing
import random
import time
import numpy as np

logging.basicConfig(level=logging.WARN)

def launch_multiple_instances(output_path,num_instances, frames_per_instance):
    processes = []
    for i in range(num_instances):
        start_frame = i * frames_per_instance
        end_frame = start_frame + frames_per_instance - 1

        process = multiprocessing.Process(target=render_frames, args=(output_path,start_frame, end_frame, i))
        processes.append(process)
        process.start()

    # Wait for all instances to complete
    for process in processes:
        process.join()

def render_frames(output_path,start_frame, end_frame, index):
    seed = int(time.time()) + index
    random.seed(seed)
    np.random.seed(seed)  # Set seed for numpy's random number generator
    
    scene = sp.Scene()
    writer = sp.Writer(scene, Path(output_path))

    rand_lights = sp.LightRandomizer(
        scene,
        no_of_lights_range=(3, 6),
        energy_range=(300, 1000),
        color_range=(0.8, 1.0),
        distance_range=(3.0, 10.0),
    )

    rand_scene = sp.SceneRandomizer(scene, backgrounds_dir=Path("backgrounds"))
    rand_obj = sp.ObjectRandomizer(scene, r_range=(0.3, 1.0))

    for obj_path in Path("meshes").glob("*/*.obj"):
        new = scene.create_from_obj(obj_path)
        rand_obj.add(new)

    cam = scene.create_camera("Camera")
    
    for frame in range(start_frame, end_frame + 1):
        rand_scene.randomize_background()
        rand_lights.randomize_lighting_around_cam(cam)
        rand_obj.randomize_appearance([0.1, 1.0], [0.1, 0.5])
        rand_obj.randomize_geometry((0.8, 1.2), (-0.8, 0.8))
        rand_obj.randomize_pose_in_camera_view(cam)

        writer.generate_data(frame)  # Save the rendered image to the specified file path

output_path = "output_folder"
num_instances = 2  # Number of Blender instances to launch
frames_per_instance = 3  # Number of frames each instance will render

launch_multiple_instances(output_path,num_instances, frames_per_instance)
