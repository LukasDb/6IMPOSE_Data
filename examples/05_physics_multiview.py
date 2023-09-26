import logging, coloredlogs

coloredlogs.install(logging.DEBUG, fmt="%(asctime)s %(levelname)s %(message)s")

import numpy as np
import multiprocessing as mp
from pathlib import Path
import click


@click.command()
@click.argument("output_path", type=click.Path(path_type=Path))
@click.argument("main_obj_path", type=click.Path(path_type=Path))
@click.option("--scale", default=1.0, help="Scale of the object.", type=float)
@click.option("--start", default=0, help="Start index for rendering.", type=int)
@click.option("--end", default=19999, help="End index for rendering.", type=int)
@click.option("--n_workers", default=8, type=int, help="Number of workers.")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing files.")
@click.option("--debug", is_flag=True, default=False, help="Debug mode.")
def main(
    start: int,
    end: int,
    n_workers: int,
    output_path: Path,
    main_obj_path: Path,
    scale: float,
    overwrite: bool,
    debug: bool = False,
):
    if debug:
        logging.getLogger().setLevel(0)
        scene = generate_data([0], Path("/tmp/debug_6IMPOSE"), main_obj_path, scale=scale)
        scene.export_blend()
        scene.run_simulation()
        exit()

    if not overwrite:
        existing_files = output_path.joinpath("gt").glob("*.json")
        existing_ids = [int(x.stem.split("_")[-1]) for x in existing_files]
        all_indices = np.setdiff1d(np.arange(start, end + 1), existing_ids)
    else:
        all_indices = np.arange(start, end + 1)

    if len(all_indices) == 0:
        logging.info("No images to render.")
        return

    n_workers = min(n_workers, len(all_indices))

    if n_workers == 1:
        logging.info("Using single process.")
        generate_data(all_indices.tolist(), output_path, main_obj_path, scale)
        return

    logging.info("Rendering %d images with %d workers.", len(all_indices), n_workers)
    indices = np.array_split(all_indices, n_workers)

    queue = mp.Queue()
    processes = [mp.Process(target=process, args=(queue,)) for _ in range(n_workers)]
    for p in processes:
        p.start()

    # load jobs
    for i in range(n_workers):
        queue.put((indices[i], output_path, main_obj_path, scale))
    # send stop signal
    for _ in range(n_workers):
        queue.put(None)

    for p in processes:
        p.join()


def process(queue: mp.Queue):
    while True:
        job = queue.get()
        if job is None:
            break
        generate_data(*job)


def generate_data(indices: list[int], output_path: Path, obj_path: Path, scale: float):
    import numpy as np

    np.random.seed(mp.current_process().pid)

    import simpose as sp
    import logging

    from tqdm import tqdm
    from scipy.spatial.transform import Rotation as R
    import random

    fh = logging.FileHandler(f"{mp.current_process().name}.log")
    fh.setLevel(0)
    logging.getLogger().addHandler(fh)

    proc_name = mp.current_process().name
    is_primary_worker = proc_name == "Process-1" or proc_name == "MainProcess"

    scene = sp.Scene(img_h=1080, img_w=1920)

    writer = sp.Writer(scene, output_path)

    # shapenet_root = Path.home().joinpath("data/shapenet/ShapeNetCore")
    ycb_root = Path.home().joinpath("data/ycb_models")
    ugreal_root = Path.home().joinpath("data/SynthDet")

    model_loader = sp.random.ModelLoader(
        scene,
        sp.CallbackType.NONE,
        root=ycb_root,
        model_source=sp.random.ModelSource.YCB,
    ) + sp.random.ModelLoader(
        scene,
        sp.CallbackType.NONE,
        root=ugreal_root,
        model_source=sp.random.ModelSource.SYNTHDET,
        scale_range=(0.25, 1),
    )
    # + model_loader = sp.random.ModelLoader(
    #     scene, sp.CallbackType.NONE, root=shapenet_root, num_objects=20
    # )

    appearance_randomizer = sp.random.AppearanceRandomizer(
        scene,
        sp.CallbackType.BEFORE_RENDER,
    )

    # cam = scene.create_camera("Camera")
    cam = scene.create_stereo_camera("Camera", baseline=0.063)
    cam.set_from_hfov(70, scene.resolution_x, scene.resolution_y, degrees=True)

    sp.random.LightRandomizer(
        scene,
        cam,
        sp.CallbackType.BEFORE_RENDER,
        no_of_lights_range=(3, 6),
        energy_range=(300, 1000),
        color_range=(0.8, 1.0),
        distance_range=(3.0, 10.0),
    )

    sp.random.BackgroundRandomizer(
        scene,
        sp.CallbackType.BEFORE_RENDER,
        backgrounds_dir=Path.home().joinpath("data/backgrounds"),
    )

    friction = 0.5

    main_obj = scene.create_object(
        obj_path, mass=0.2, friction=friction, add_semantics=True, scale=scale
    )
    main_obj.set_metallic(0.0)
    main_obj.set_roughness(0.5)

    if is_primary_worker:
        scene.export_meshes(output_path / "meshes")

    main_objs = [main_obj]

    for i in range(19):
        main_objs.append(scene.create_copy(main_obj))

    for obj in main_objs:
        appearance_randomizer.add(obj)
        obj.hide()

    # data generation params
    dt = 1 / 4.0
    drop_duration = 8
    num_dt_step = int(drop_duration / dt)
    num_cam_locs = 1  # 20

    i = 0
    bar = tqdm(total=len(indices), desc="Process-1", smoothing=0.0, disable=not is_primary_worker)

    drop_height = 0.8

    while True:
        model_loader.reset()
        # new_objs = model_loader.get_objects(mass=0.1, friction=friction)
        # for new_obj in new_objs:
        #     appearance_randomizer.add(new_obj)

        # random.shuffle(new_objs)
        # drop_objects = new_objs + main_objs

        # for j, obj in enumerate(drop_objects):
        #     obj.show()
        #     obj.set_location(
        #         (
        #             np.random.uniform(-0.05, 0.05),
        #             np.random.uniform(-0.05, 0.05),
        #             j * 0.1 + 0.1,
        #         )
        #     )
        #     obj.set_rotation(R.random())
        # scene.step_physics(1.0)  # initial fall

        # add 20 objects and let fall
        for obj in model_loader.get_objects(40, mass=1, friction=friction, hide=True):
            appearance_randomizer.add(obj)
            obj.show()
            obj.set_location(
                (
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.05, 0.05),
                    drop_height,
                )
            )
            obj.set_rotation(R.random())
            scene.step_physics(0.5)  # initial fall

        # mix main_objects and a few distractors
        distractors = model_loader.get_objects(5, mass=0.2, friction=friction, hide=True)
        for obj in distractors:
            appearance_randomizer.add(obj)

        drop_objects = main_objs + distractors
        random.shuffle(drop_objects)

        for obj in drop_objects:
            obj.show()
            appearance_randomizer.add(obj)

            obj.set_location(
                (
                    np.random.uniform(-0.05, 0.05),
                    np.random.uniform(-0.05, 0.05),
                    drop_height,
                )
            )
            obj.set_rotation(R.random())
            scene.step_physics(0.5)

        for _ in range(num_dt_step):
            scene.step_physics(dt)

            # sample 20 camera locations in upper hemisphere
            rots = R.random(num=num_cam_locs)
            cam_view = np.array([0.0, 0.0, 1.0])
            radius = np.random.uniform(0.3, 1.5, size=(num_cam_locs,))

            cam_locations = rots.apply(cam_view) * radius[:, None]
            cam_locations[:, 2] *= np.sign(cam_locations[:, 2])
            cam_locations[:, 2] += 0.2

            for cam_location in cam_locations:
                cam.set_location(cam_location)
                cam.point_at(np.array([0.0, 0.0, 0.0]))  # with z up

                cam.apply_local_rotation_offset(
                    R.from_euler("z", np.random.uniform(-5, 5), degrees=True)
                )  # minor rotation noise

                writer.generate_data(indices[i])

                i += 1
                if i == len(indices):
                    bar.close()
                    return scene

                bar.update(1)


if __name__ == "__main__":
    main()
