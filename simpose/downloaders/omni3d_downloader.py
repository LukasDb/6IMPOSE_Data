import simpose as sp
from tqdm import tqdm
import os
import multiprocessing as mp
from pathlib import Path
from openxlab.dataset import download

from .downloader import Downloader


def untar_and_remove(tar_file: Path) -> None:
    # print(f"Extracting {tar_file.name}")
    output_dir = tar_file.parent.parent.parent.parent.joinpath("models").resolve()
    os.system(f"tar -xvf {tar_file} -C {output_dir}")
    os.remove(tar_file)


def simplify_meshes(input_file: Path) -> None:
    import pymeshlab

    # print(f"Simplifying {input_file}")
    out_path = input_file.resolve().with_stem("simplified")
    if not out_path.exists():
        # get size of input_file in mb
        size_mb = input_file.stat().st_size / 1e6
        # reduce size to around 2MB per mesh
        targetperc = 2 / size_mb

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_file.resolve()))
        try:
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                # targetfacenum=6,  # Target number of faces:
                targetperc=targetperc,  # Percentage reduction (0..1): If non zero, this parameter specifies the desired final size of the mesh as a percentage of the initial mesh.
                qualitythr=0.5,  # Quality threshold: Quality threshold for penalizing bad shaped faces. The value is in the range [0..1] 0 accept any kind of face (no penalties), 0.5 penalize faces with quality < 0.5, proportionally to their shape
                extratcoordw=1,  # Texture Weight: Additional weight for each extra Texture Coordinates for every (selected) vertex
                preserveboundary=False,  #  Preserve Boundary of the mesh: The simplification process tries not to destroy mesh boundaries
                boundaryweight=1,  # Boundary Preserving Weight: The importance of the boundary during simplification. Default (1.0) means that the boundary has the same importance of the rest. Values greater than 1.0 raise boundary importance and has the effect of removing less vertices on the border. Admitted range of values (0,+inf).
                optimalplacement=True,  # Optimal position of simplified vertices: Each collapsed vertex is placed in the position minimizing the quadric error. It can fail (creating bad spikes) in case of very flat areas.If disabled edges are collapsed onto one of the two original vertices and the final mesh is composed by a subset of the original vertices.
                preservenormal=True,  # Preserve Normal: Try to avoid face flipping effects and try to preserve the original orientation of the surface (default False)
                planarquadric=False,  # Planar Simplification: Add additional simplification constraints that improves the quality of the simplification of the planar portion of the mesh.
                selected=False,
            )
        except Exception:
            print(f"Got error for {input_file}")
            pass
        ms.save_current_mesh(str(out_path.resolve()), save_textures=False)

        # overwrite dummpy.png with old Scan.jbg in mtl
        mtl_path = input_file.parent.joinpath("simplified.obj.mtl")
        if not mtl_path.exists():
            return
        with open(mtl_path, "r") as f:
            mtl = f.read()
        mtl = mtl.replace("dummy.png", "Scan.jpg")
        with open(mtl_path, "w") as f:
            f.write(mtl)


def run_vhacd(input_file: Path) -> None:
    import pybullet as p

    # print(f"Running VHACD on {input_file}")
    out_path = input_file.resolve().with_name(input_file.stem + "_vhacd.obj")
    if not out_path.exists():
        # hierarchical decomposition for dynamic collision of concave objects
        with sp.redirect_stdout():
            p.vhacd(
                str(input_file.resolve()),
                str(out_path),
                str(input_file.parent.joinpath("log.txt").resolve()),
            )


class Omni3DDownloader(Downloader):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)

    def run(self) -> None:
        if (
            input(
                "The extracted Omni3D objects dataset will take around 980 GB. Do you want to continue? (y/n)"
            )
            != "y"
        ):
            return

        if self.output_dir.joinpath("models").exists():
            print("Omni3D dataset already exists.")
            return

        if os.system("openxlab login"):
            print("Please login to OpenXLab.")
            return

        self.output_dir.joinpath("models").mkdir(parents=True, exist_ok=True)

        download(
            dataset_repo="OpenXDLab/OmniObject3D-New",
            source_path="/raw/raw_scans",
            target_path=str(self.output_dir.expanduser().resolve()),
        )

        # untar & remove tars
        tar_dir = self.output_dir.joinpath("OpenXDLab___OmniObject3D-New/raw/raw_scans")
        with mp.Pool() as pool:
            pool.map_async(untar_and_remove, tar_dir.glob("*.tar.gz")).get()

        with mp.Pool() as pool:
            scans = list(self.output_dir.joinpath("models").glob("*/Scan/Scan.obj"))
            tqdm(
                list(
                    pool.imap_unordered(
                        simplify_meshes,
                        scans,
                        chunksize=1,
                    )
                ),
                total=len(scans),
            )

        simplified = list(self.output_dir.joinpath("models").glob("*/Scan/simplified.obj"))
        with mp.Pool() as pool:
            tqdm(
                list(
                    pool.imap_unordered(
                        run_vhacd,
                        simplified,
                    )
                ),
                total=len(simplified),
            )

        # TODO collect metadata and statistics for all objects -> to be used in modelloader
