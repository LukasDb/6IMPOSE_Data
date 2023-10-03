import logging, coloredlogs

coloredlogs.install(logging.DEBUG, fmt="%(asctime)s %(levelname)s %(message)s")

import simpose as sp
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import random
import time
import numpy as np

from simpose.callback import CallbackType


def launch_multiple_instances(output_path, num_instances, frames_per_instance):
    processes = []
    for i in range(num_instances):
        start_frame = i * frames_per_instance
        end_frame = start_frame + frames_per_instance - 1

        process = multiprocessing.Process(
            target=render_frames, args=(output_path, start_frame, end_frame, i)
        )
        processes.append(process)
        process.start()

    # Wait for all instances to complete
    for process in processes:
        process.join()


def render_frames(output_path, start_frame, end_frame, index):
    seed = int(time.time()) + index
    random.seed(seed)

    scene = sp.Scene()
    writer = sp.Writer(scene, Path(output_path))

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

    for obj_path in Path("meshes").glob("*/*.obj"):
        new = scene.create_object(obj_path, add_semantics=True)
        rand_obj.add(new)

    for frame in range(start_frame, end_frame + 1):
        writer.generate_data(frame)  # Save the rendered image to the specified file path


output_path = "output_folder"
num_instances = 2  # Number of Blender instances to launch
frames_per_instance = 3  # Number of frames each instance will render

launch_multiple_instances(output_path, num_instances, frames_per_instance)
