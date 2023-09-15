from ast import Assert
import logging, coloredlogs

coloredlogs.install(logging.DEBUG, fmt="%(asctime)s %(levelname)s %(message)s")

from pathlib import Path
import multiprocessing as mp
from turtle import st
from tqdm import tqdm
from simpose.redirect_stdout import redirect_stdout
import logging
import pybullet as p
import subprocess
import shutil


def remove(folder: Path):
    shutil.rmtree(folder)


def process(folder: Path):
    # validates, converts to gltf and creates vhacd collision shape
    obj_path = folder.joinpath("models/model_normalized.obj")
    if not obj_path.exists():
        print("model_normalized.obj not found")
        remove(folder)
        return

    vhacd_path = obj_path.resolve().with_name(obj_path.stem + "_vhacd.obj")
    if not vhacd_path.exists():
        try:
            subprocess.run(
                [
                    "python",
                    "-c",
                    f"import pybullet as p;p.vhacd(\"{str(obj_path.resolve())}\",\"{str(vhacd_path)}\",\"{str(obj_path.parent.joinpath('log.txt').resolve())}\",depth=5,)",
                ],
                timeout=30.0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            print("Error while creating vhacd: ", obj_path)
            remove(folder)
            return

    gltf_path = obj_path.resolve().with_suffix(".gltf")

    return
    if not gltf_path.exists():
        try:
            result = subprocess.run(
                ["obj2gltf", "-i", str(obj_path), "-o", str(gltf_path)],
                stdout=subprocess.DEVNULL,
                timeout=30.0,
            )
        except subprocess.TimeoutExpired:
            print("Error while converting to gltf: ", obj_path)
            remove(folder)
            return

        if result.returncode != 0:
            print("ERROR while converting to gltf: ", obj_path)
            remove(folder)


if __name__ == "__main__":
    _shapenet_root = Path("/media/lukas/G-RAID/datasets/shapenet/ShapeNetCore")
    # get a list of all folders in shapenet root
    shapenet_contents = _shapenet_root.iterdir()
    _shapenet_types = list([x for x in shapenet_contents if x.is_dir()])

    obj_paths = []
    print("looking for objs...")

    for obj_type in tqdm(_shapenet_types):
        sublist = list(obj_type.iterdir())
        obj_paths += [x for x in sublist if x.is_dir()]

    print(f"Found {len(obj_paths)} objs.")

    # single process for debugging
    # for obj_type in tqdm(obj_paths):
    #     process(obj_type)
    # exit()

    with mp.Pool() as pool:
        pool.map_async(process, tqdm(obj_paths, smoothing=0.0), chunksize=100).get()
        pool.join()
