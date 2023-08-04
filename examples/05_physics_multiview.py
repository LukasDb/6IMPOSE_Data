import numpy as np
import multiprocessing as mp
from pathlib import Path


def main():
    n_frames = 20000
    start_from = 0
    n_workers = 4
    inds = np.arange(start_from, start_from + n_frames)
    inds = np.array_split(inds, n_workers)

    output_path = Path("6IMPOSE/multi_pliers")

    queue = mp.Queue()
    for i in range(n_workers):
        queue.put((inds[i][0], inds[i][-1], output_path))

    for _ in range(n_workers):
        queue.put(None)

    processes = [mp.Process(target=process, args=(queue,), daemon=True) for _ in range(n_workers)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def process(queue: mp.Queue):
    while True:
        job = queue.get()
        if job is None:
            break
        generate_data(*job)


def generate_data(start: int, end: int, output_path: Path):
    import simpose as sp
    import logging
    from tqdm import tqdm
    from scipy.spatial.transform import Rotation as R
    import random

    logging.basicConfig(level=logging.INFO)

    scene = sp.Scene()

    writer = sp.Writer(scene, output_path, render_object_masks=False)

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

    rand_scene = sp.random.BackgroundRandomizer(
        scene, sp.CallbackType.BEFORE_RENDER, backgrounds_dir=Path("backgrounds")
    )

    # obj_path = Path("meshes/cpsduck/cpsduck.obj")
    obj_path = Path("meshes/pliers/pliers.obj")
    main_obj = scene.create_from_obj(obj_path, mass=0.2, friction=0.8, add_semantics=True)
    main_obj.set_metallic_value(0.0)
    main_obj.set_roughness_value(0.5)

    main_objs = [main_obj]

    for i in range(19):
        main_objs.append(scene.create_copy(main_obj, linked=True))

    # data generation params
    dt = 1 / 5.0  # 5 FPS
    drop_duration = 5
    num_drops = 100
    num_dt_step = int(drop_duration / dt)
    num_cam_locs = 20
    # print(f"Generating {num_dt_step * num_cam_locs * num_drops} images")

    i = start
    # bar = tqdm(total=num_drops * num_dt_step * num_cam_locs)
    bar = tqdm(total=end - start + 1)
    # for run in range(num_drops):  # 50 different drops
    while True:
        drop_objects = main_objs + shapenet.get_objects(mass=0.1, friction=0.8)

        random.shuffle(drop_objects)

        for j, obj in enumerate(drop_objects):
            obj.show()
            obj.set_location(
                (
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.05, 0.05),
                    j * 0.1 + 0.1,
                )
            )
            obj.set_rotation(R.random())

        scene.step_physics(1.0)  # initial fall

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
                cam.point_at(np.array([0.0, 0.0, 0.0]))  # with z up

                cam.apply_local_rotation_offset(
                    R.from_euler("z", np.random.uniform(-5, 5), degrees=True)
                )  # minor rotation noise

                writer.generate_data(i)
                i += 1
                bar.update(1)
                if i > end:
                    bar.close()
                    return


if __name__ == "__main__":
    main()
