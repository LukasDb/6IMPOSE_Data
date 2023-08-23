from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from simpose.redirect_stdout import redirect_stdout
import logging
import pybullet as p

logging.getLogger().setLevel(logging.INFO)


def process_type(obj_type: Path):
    from simpose.redirect_stdout import redirect_stdout

    objs = list(obj_type.iterdir())
    for obj_path in objs:
        obj_path = obj_path.joinpath("models/model_normalized.obj")
        out_path = obj_path.resolve().with_name(obj_path.stem + "_vhacd.obj")
        if not out_path.exists():
            print(f"VHACD for {obj_path}")
            with redirect_stdout():
                p.vhacd(
                    str(obj_path.resolve()),
                    str(out_path),
                    str(obj_path.parent.joinpath("log.txt").resolve()),
                )


def process_single(obj_path: Path):
    obj_path = obj_path.joinpath("models/model_normalized.obj")
    out_path = obj_path.resolve().with_name(obj_path.stem + "_vhacd.obj")
    if not out_path.exists():
        # print(f"VHACD for {obj_path}")
        with redirect_stdout():
            p.vhacd(
                str(obj_path.resolve()),
                str(out_path),
                str(obj_path.parent.joinpath("log.txt").resolve()),
                depth=5,  # default = 10
            )


if __name__ == "__main__":
    _shapenet_root = Path("/media/lukas/G-RAID/datasets/shapenet/ShapeNetCore")
    # get a list of all folders in shapenet root
    shapenet_contents = _shapenet_root.iterdir()
    _shapenet_types = list([x for x in shapenet_contents if x.is_dir()])

    obj_paths = []
    print("looking for objs...")
    for type in tqdm(_shapenet_types):
        obj_paths += list(type.iterdir())
    print(f"Found {len(obj_paths)} objs.")
    # for obj_type in _shapenet_types:
    #    process_type(obj_type)
    with mp.Pool() as pool:
        # pool.map_async(process_type, _shapenet_types).get()

        pool.map_async(process_single, tqdm(obj_paths, smoothing=0.0), chunksize=100).get()
        # , total=len(obj_paths), smoothing=0.0))

        pool.join()
