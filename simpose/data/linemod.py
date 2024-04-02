import simpose as sp
from scipy.spatial.transform import Rotation as R
import json
import numpy as np
import requests
from simpose.data.tfrecord_dataset import TFRecordDataset
from pathlib import Path
import tensorflow as tf
import shutil
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp


class LineMod(TFRecordDataset):
    @staticmethod
    def get(
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "*.tfrecord",
        num_parallel_files: int = 16,
    ) -> dict[str, tf.data.Dataset]:  # download and extract if tfrecord not found
        if (
            not root_dir.joinpath("data").exists()
            or len(list(root_dir.joinpath("data").glob("*"))) == 0
        ):
            # download and extract
            sp.logger.warning(
                "Could not find pre-processed LineMOD. Downloading and extracting may take a while..."
            )

            lm_url = "https://bop.felk.cvut.cz/media/data/bop_datasets/"

            for file in ["lm_base.zip", "lm_models.zip", "lm_test_all.zip"]:
                with requests.get(lm_url + file, allow_redirects=True, stream=True) as r:
                    total_length = r.headers.get("content-length")
                    bar = tqdm(total=int(total_length), unit="B", unit_scale=True)

                    r.raise_for_status()
                    with root_dir.joinpath("temp.zip").open("wb") as F:
                        # F.write(r.content)
                        for chunk in r.iter_content(chunk_size=8192):
                            F.write(chunk)
                            bar.update(len(chunk))
                    bar.close()
                # unzip
                shutil.unpack_archive(root_dir.joinpath("temp.zip"), extract_dir=root_dir)

            # delete temp.zip
            shutil.rmtree(root_dir.joinpath("temp.zip"))

            root_dir.joinpath("data").mkdir(exist_ok=True)

            subsets = list(root_dir.joinpath("test").glob("*"))
            with mp.Manager() as manager:
                qs = [manager.Queue() for _ in subsets]
                bars: dict[str, tqdm] = {}
                with mp.Pool() as pool:
                    res = pool.starmap_async(
                        LineMod._process_subset,
                        [(root_dir, subset, q) for subset, q in zip(subsets, qs)],
                    )

                    while not res.ready():
                        for q in qs:
                            try:
                                num_imgs = q.get(block=False)
                                if q not in bars:
                                    bars[q] = tqdm(total=num_imgs, unit="images")

                                bars[q].update(1)
                            except:
                                pass
                    while not all(q.empty() for q in qs):
                        for q in qs:
                            try:
                                num_imgs = q.get(block=False)
                                bars[q].update(num_imgs)
                            except:
                                pass
                for bar in bars.values():
                    bar.close()

            shutil.rmtree(root_dir.joinpath("test"))

        subsets = root_dir.joinpath("data").glob("*")
        return {
            s.name: TFRecordDataset.get(
                s, get_keys=get_keys, pattern=pattern, num_parallel_files=num_parallel_files
            )
            for s in subsets
        }

    @staticmethod
    def _get_bbox(mask):
        y, x = np.where(mask > 0)
        if len(y) == 0:
            box = np.array([0.0, 0.0, 0.0, 0.0])
        else:
            x1 = np.min(x).tolist()
            x2 = np.max(x).tolist()
            y1 = np.min(y).tolist()
            y2 = np.max(y).tolist()
            box = np.array((x1, y1, x2, y2))
        return box.astype(np.int32)

    @staticmethod
    def _process_subset(root_dir: Path, subset: Path, q):
        with open(root_dir.joinpath("lm/camera.json")) as F:
            cam_info = json.load(F)
        fx, fy = cam_info["fx"], cam_info["fy"]
        cx, cy = cam_info["cx"], cam_info["cy"]
        depth_scale = cam_info["depth_scale"]
        intrinsics = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        ).astype(np.float32)

        num_imgs = len(list(subset.joinpath("rgb").glob("*.png")))
        output_dir = root_dir.joinpath(f"data/{subset.name}")
        output_dir.mkdir(exist_ok=True)

        start_index = 0
        end_index = num_imgs - 1
        # convert to tfrecord
        writer_config = sp.writers.WriterConfig(
            output_dir=output_dir,
            overwrite=True,
            start_index=start_index,
            end_index=end_index,
        )

        with subset.joinpath("scene_gt.json").open("r") as F:
            all_annotations = json.load(F)

        with sp.writers.TFRecordWriter(writer_config, None) as writer:
            for i in range(num_imgs):
                rgb = np.asarray(Image.open(subset.joinpath(f"rgb/{i:06d}.png"))).astype(np.uint8)
                depth = (
                    np.asarray(Image.open(subset.joinpath(f"depth/{i:06d}.png")))
                    * depth_scale
                    / 1000.0
                ).astype(
                    np.float32
                )  # to m

                annotations = all_annotations[str(i)]

                object_annotations: list[sp.ObjectAnnotation] = []
                mask = np.zeros_like(depth, dtype=np.int32)
                for index, annotation in enumerate(annotations):
                    obj_mask = np.asarray(
                        Image.open(subset.joinpath(f"mask/{i:06d}_{index:06d}.png"))
                    )
                    obj_mask_visib = np.asarray(
                        Image.open(subset.joinpath(f"mask_visib/{i:06d}_{index:06d}.png"))
                    )
                    mask[obj_mask_visib > 0] = annotation["obj_id"]

                    px_count_visib = np.count_nonzero(obj_mask_visib > 0)
                    px_count_all = np.count_nonzero(obj_mask > 0)
                    px_count_valid = np.count_nonzero((obj_mask_visib > 0) & (depth > 0))
                    if px_count_all != 0:
                        visib_fract = px_count_visib / px_count_all

                    rot = np.array(annotation["cam_R_m2c"]).reshape(3, 3)
                    t = np.array(annotation["cam_t_m2c"]).reshape(3) / 1000.0  # to m

                    object_annotations.append(
                        sp.ObjectAnnotation(
                            cls=str(annotation["obj_id"]),
                            object_id=annotation["obj_id"],
                            position=t,
                            quat_xyzw=R.from_matrix(rot).as_quat(),
                            bbox_visib=LineMod._get_bbox(obj_mask_visib),
                            bbox_obj=LineMod._get_bbox(obj_mask),
                            px_count_visib=px_count_visib,
                            px_count_valid=px_count_valid,
                            px_count_all=px_count_all,
                            visib_fract=visib_fract,
                        )
                    )

                rp = sp.RenderProduct(
                    rgb=rgb,
                    depth=depth,
                    mask=mask,
                    cam_position=np.array([0, 0.0, 0.0]).astype(np.float32),
                    cam_quat_xyzw=np.array([0, 0, 0, 1]).astype(np.float32),
                    intrinsics=intrinsics,
                    objs=object_annotations,
                )

                writer.write_data(i, render_product=rp)
                q.put(num_imgs)
