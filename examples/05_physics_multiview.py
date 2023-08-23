import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
import logging
import click
from typing import List

logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(0)


@click.command()
@click.argument("output_path", type=str)
@click.argument("main_obj_path", type=str)
@click.option("--scale", default=1.0, help="Scale of the object.", type=float)
@click.option("--start", default=0, help="Start index for rendering.", type=int)
@click.option("--end", default=19999, help="End index for rendering.", type=int)
@click.option("--n_workers", default=8, type=int, help="Number of workers.")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing files.")
def main(
    start: int,
    end: int,
    n_workers: int,
    output_path: str,
    main_obj_path: str,
    scale: float,
    overwrite: bool,
):
    if not overwrite:
        existing_files = Path(output_path).joinpath("gt").glob("*.json")
        existing_ids = [int(x.stem.split("_")[-1]) for x in existing_files]
        all_inds = np.setdiff1d(np.arange(start, end + 1), existing_ids)
    else:
        all_inds = np.arange(start, end + 1)

    if len(all_inds) == 0:
        logging.info("No images to render.")
        return

    n_workers = min(n_workers, len(all_inds))
    logging.info("Rendering %d images with %d workers.", len(all_inds), n_workers)
    inds = np.array_split(all_inds, n_workers)

    queue = mp.Queue()
    for i in range(n_workers):
        queue.put((inds[i], Path(output_path), Path(main_obj_path), scale))

    for _ in range(n_workers):
        queue.put(None)

    processes = [mp.Process(target=process, args=(queue,)) for _ in range(n_workers)]
    for p in processes:
        p.start()
        time.sleep(10)  # 'load balancing'
    for p in processes:
        p.join()


def process(queue: mp.Queue):
    while True:
        job = queue.get()
        if job is None:
            break
        generate_data(*job)


def generate_data(inds: List[int], output_path: Path, obj_path: Path, scale: float):
    import simpose as sp
    from tqdm import tqdm
    from scipy.spatial.transform import Rotation as R
    import random

    scene = sp.Scene()

    writer = sp.Writer(scene, output_path)

    shapenet_root = Path("/media/lukas/G-RAID/datasets/shapenet/ShapeNetCore")
    shapenet = sp.random.ShapenetLoader(
        scene, sp.CallbackType.NONE, shapenet_root=shapenet_root, num_objects=20
    )

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

    rand_bg = sp.random.BackgroundRandomizer(
        scene,
        sp.CallbackType.BEFORE_RENDER,
        backgrounds_dir=Path("/media/lukas/G-RAID/datasets/backgrounds"),
    )

    friction = 0.8

    main_obj = scene.create_object(
        obj_path, mass=0.2, friction=friction, add_semantics=True, scale=scale
    )
    main_obj.set_metallic_value(0.0)
    main_obj.set_roughness_value(0.5)

    main_objs = [main_obj]

    for i in range(19):
        main_objs.append(scene.create_copy(main_obj))

    # data generation params
    dt = 1 / 4.0
    drop_duration = 8
    num_dt_step = int(drop_duration / dt)
    num_cam_locs = 1  # 20

    i = 0
    if mp.current_process().name == "Process-1":
        bar = tqdm(total=len(inds), desc="Process-1", smoothing=0.0)
    else:
        bar = None
    while True:
        drop_objects = main_objs + shapenet.get_objects(mass=0.1, friction=friction)
        random.shuffle(drop_objects)

        for j, obj in enumerate(drop_objects):
            obj.show()
            obj.set_location(
                (
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.05, 0.05),
                    j * 0.05 + 0.1,
                )
            )
            obj.set_rotation(R.random())

        scene.step_physics(1.0)  # initial fall

        for _ in range(num_dt_step):
            scene.step_physics(dt)

            # sample 20 camera locations in upper hemisphere
            rots = R.random(num=num_cam_locs)
            cam_view = np.array([0.0, 0.0, 1.0])
            radius = np.random.uniform(0.5, 1.5, size=(num_cam_locs,))

            cam_locations = rots.apply(cam_view) * radius[:, None]
            cam_locations[:, 2] *= np.sign(cam_locations[:, 2])
            cam_locations[:, 2] += 0.2

            for cam_location in cam_locations:
                cam.set_location(cam_location)
                cam.point_at(np.array([0.0, 0.0, 0.0]))  # with z up

                cam.apply_local_rotation_offset(
                    R.from_euler("z", np.random.uniform(-5, 5), degrees=True)
                )  # minor rotation noise

                for obj in drop_objects:
                    obj.set_metallic_value(np.random.uniform(0, 1.0))
                    obj.set_roughness_value(np.random.uniform(0, 1.0))

                writer.generate_data(inds[i])

                i += 1
                if i == len(inds):
                    if bar is not None:
                        bar.close()
                    return
                if bar is not None:
                    bar.update(1)
    bar.close()


if __name__ == "__main__":
    main()
