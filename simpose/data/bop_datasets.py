import os
import typing
import simpose as sp
from scipy.spatial.transform import Rotation as R
import json
import numpy as np
import requests
from simpose.data.tfrecord_dataset import TFRecordDataset
from pathlib import Path

if typing.TYPE_CHECKING:
    import tensorflow as tf
import shutil
from tqdm import tqdm
from PIL import Image
import multiprocessing as mp


class _BOPBase(TFRecordDataset):
    _urls: list[str] = []
    _base_dir: str = ""
    _img_dir: str = ""
    _estimated_size: str = ""
    CLASSES: dict[str, int] = {}

    @classmethod
    def get(
        cls,
        root_dir: Path,
        get_keys: None | list[str] = None,
        num_parallel_files: int = 16,
    ) -> "tf.data.Dataset":  # download and extract if tfrecord not found

        root_dir.mkdir(exist_ok=True)

        if (
            not root_dir.joinpath("subsets").exists()
            or len(list(root_dir.joinpath("subsets").glob("*"))) == 0
        ):
            # download and extract
            sp.logger.warning(
                f"Could not find pre-processed data. Downloading and extracting from https://bop.felk.cvut.cz/datasets/. Required disk space is about {cls._estimated_size}..."
            )

            cls.download_and_extract(root_dir)

            # process
            root_dir.joinpath("subsets").mkdir(exist_ok=True)

            subsets = list(root_dir.joinpath(cls._img_dir).glob("*"))

            assert cls.process_parallel(root_dir, subsets), "Could not process dataset"

            # remove unprocessed images
            shutil.rmtree(root_dir.joinpath(cls._img_dir))

        # return dict of datasets
        return TFRecordDataset.get(
            root_dir, get_keys=get_keys, num_parallel_files=num_parallel_files
        )

    @classmethod
    def process_parallel(cls, root_dir, subsets) -> bool:
        with mp.Manager() as manager:
            set_queue: dict[mp.Queue, Path] = {manager.Queue(): subset for subset in subsets}
            bars: dict[str, tqdm] = {}

            with mp.Pool() as pool:
                res = pool.starmap_async(
                    cls._process_subset,
                    [(root_dir, subset, q) for q, subset in set_queue.items()],
                )

                while not res.ready():
                    for q in set_queue.keys():
                        try:
                            num_imgs = q.get(block=False)
                            if q not in bars:
                                bars[q] = tqdm(
                                    total=num_imgs, unit="images", desc=f"{set_queue[q].name}"
                                )

                            bars[q].update(1)
                        except:
                            pass
                while not all(q.empty() for q in set_queue.keys()):
                    for q in set_queue.keys():
                        try:
                            num_imgs = q.get(block=False)
                            bars[q].update(num_imgs)
                        except:
                            pass
            for bar in bars.values():
                bar.close()

        return all(res.get())

    @classmethod
    def download_and_extract(cls, root_dir: Path):
        for url in cls._urls:
            with requests.get(url, allow_redirects=True, stream=True) as r:
                total_length = r.headers.get("content-length")
                bar = tqdm(
                    total=int(total_length),
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {url.split('/')[-1]}",
                )

                r.raise_for_status()
                with root_dir.joinpath("temp.zip").open("wb") as F:
                    # F.write(r.content)
                    for chunk in r.iter_content(chunk_size=8192):
                        F.write(chunk)
                        bar.update(len(chunk))
                bar.close()
            # unzip
            sp.logger.info("Extracting...")
            shutil.unpack_archive(root_dir.joinpath("temp.zip"), extract_dir=root_dir)

        # remove temp.zip
        root_dir.joinpath("temp.zip").unlink(missing_ok=True)

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
    def _extract_cam_intrinsics(cam_info):
        depth_scale = cam_info["depth_scale"]
        intrinsics = np.array(cam_info["cam_K"]).reshape(3, 3).astype(np.float32)
        return intrinsics, depth_scale

    @classmethod
    def _process_subset(cls, root_dir: Path, subset: Path, q) -> bool:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with open(subset.joinpath(f"scene_camera.json")) as F:
            cam_infos = json.load(F)

        img_files = list(subset.joinpath("rgb").glob("*.png"))
        img_indices = [int(x.stem) for x in img_files]
        img_indices = sorted(img_indices)

        num_imgs = len(img_files)
        output_dir = root_dir.joinpath(f"subsets/{subset.name}")
        output_dir.mkdir(exist_ok=True)

        # convert to tfrecord
        writer_config = sp.writers.WriterConfig(
            output_dir=output_dir,
            overwrite=True,
            start_index=0,
            end_index=num_imgs - 1,
        )

        with subset.joinpath("scene_gt.json").open("r") as F:
            all_annotations = json.load(F)

        id2cls_map = {v: k for k, v in cls.CLASSES.items()}

        with sp.writers.TFRecordWriter(writer_config, None) as writer:
            for img_index in img_indices:
                intrinsics, depth_scale = cls._extract_cam_intrinsics(cam_infos[str(img_index)])

                rgb = np.asarray(Image.open(subset.joinpath(f"rgb/{img_index:06d}.png"))).astype(
                    np.uint8
                )
                depth = (
                    np.asarray(Image.open(subset.joinpath(f"depth/{img_index:06d}.png")))
                    * depth_scale
                    / 1000.0
                ).astype(
                    np.float32
                )  # to m

                annotations = all_annotations[str(img_index)]

                object_annotations: list[sp.ObjectAnnotation] = []
                mask = np.zeros_like(depth, dtype=np.int32)
                for instance_index, annotation in enumerate(annotations):
                    obj_mask = np.asarray(
                        Image.open(
                            subset.joinpath(f"mask/{img_index:06d}_{instance_index:06d}.png")
                        )
                    )
                    obj_mask_visib = np.asarray(
                        Image.open(
                            subset.joinpath(f"mask_visib/{img_index:06d}_{instance_index:06d}.png")
                        )
                    )
                    mask[obj_mask_visib > 0] = instance_index + 1  # 0 is background

                    px_count_visib = np.count_nonzero(obj_mask_visib > 0)
                    px_count_all = np.count_nonzero(obj_mask > 0)
                    px_count_valid = np.count_nonzero((obj_mask_visib > 0) & (depth > 0))
                    if px_count_all != 0:
                        visib_fract = px_count_visib / px_count_all

                    rot = np.array(annotation["cam_R_m2c"]).reshape(3, 3)
                    t = np.array(annotation["cam_t_m2c"]).reshape(3) / 1000.0  # to m

                    object_annotations.append(
                        sp.ObjectAnnotation(
                            cls=str(id2cls_map[annotation["obj_id"]]),
                            object_id=instance_index + 1,
                            position=t,
                            quat_xyzw=R.from_matrix(rot).as_quat(),
                            bbox_visib=cls._get_bbox(obj_mask_visib),
                            bbox_obj=cls._get_bbox(obj_mask),
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

                writer.write_data(img_index, render_product=rp)
                q.put(num_imgs)

        return True


class LineMod(_BOPBase):
    _img_dir = "test"
    _base_dir = "lm"
    _estimated_size = "14GB"
    CLASSES = {
        "ape": 1,
        "benchvise": 2,
        "bowl": 3,
        "cam": 4,
        "can": 5,
        "cat": 6,
        "cup": 7,
        "driller": 8,
        "duck": 9,
        "eggbox": 10,
        "glue": 11,
        "holepuncher": 12,
        "iron": 13,
        "lamp": 14,
        "phone": 15,
    }
    _urls = [
        "https://bop.felk.cvut.cz/media/data/bop_datasets/lm_base.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/lm_models.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/lm_test_all.zip",
    ]


class LineModOccluded(_BOPBase):
    _img_dir = "test"
    _base_dir = "lmo"
    _estimated_size = "1GB"
    CLASSES = {
        "ape": 1,
        "benchvise": 2,
        "bowl": 3,
        "cam": 4,
        "can": 5,
        "cat": 6,
        "cup": 7,
        "driller": 8,
        "duck": 9,
        "eggbox": 10,
        "glue": 11,
        "holepuncher": 12,
        "iron": 13,
        "lamp": 14,
        "phone": 15,
    }
    _urls = [
        "https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_base.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_models.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/lmo_test_all.zip",
    ]


class TLess(_BOPBase):
    _img_dir = "test_primesense"
    _base_dir = "tless"
    _estimated_size = "9.4GB"
    CLASSES = {
        "obj_01": 1,
        "obj_02": 2,
        "obj_03": 3,
        "obj_04": 4,
        "obj_05": 5,
        "obj_06": 6,
        "obj_07": 7,
        "obj_08": 8,
        "obj_09": 9,
        "obj_10": 10,
        "obj_11": 11,
        "obj_12": 12,
        "obj_13": 13,
        "obj_14": 14,
        "obj_15": 15,
        "obj_16": 16,
        "obj_17": 17,
        "obj_18": 18,
        "obj_19": 19,
        "obj_20": 20,
        "obj_21": 21,
        "obj_22": 22,
        "obj_23": 23,
        "obj_24": 24,
        "obj_25": 25,
        "obj_26": 26,
        "obj_27": 27,
        "obj_28": 28,
        "obj_29": 29,
        "obj_30": 30,
    }
    _urls = [
        "https://bop.felk.cvut.cz/media/data/bop_datasets/tless_base.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/tless_models.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/tless_test_primesense_all.zip",
    ]


class HomeBrewedDB(_BOPBase):
    _img_dir = "val_primesense"
    _base_dir = "hb"
    _estimated_size = "2.7GB"
    CLASSES = {
        "obj_01": 1,
        "obj_02": 2,
        "obj_03": 3,
        "obj_04": 4,
        "obj_05": 5,
        "obj_06": 6,
        "obj_07": 7,
        "obj_08": 8,
        "obj_09": 9,
        "obj_10": 10,
        "obj_11": 11,
        "obj_12": 12,
        "obj_13": 13,
        "obj_14": 14,
        "obj_15": 15,
        "obj_16": 16,
        "obj_17": 17,
        "obj_18": 18,
        "obj_19": 19,
        "obj_20": 20,
        "obj_21": 21,
        "obj_22": 22,
        "obj_23": 23,
        "obj_24": 24,
        "obj_25": 25,
        "obj_26": 26,
        "obj_27": 27,
        "obj_28": 28,
        "obj_29": 29,
        "obj_30": 30,
        "obj_31": 31,
        "obj_32": 32,
        "obj_33": 33,
    }
    _urls = [
        "https://bop.felk.cvut.cz/media/data/bop_datasets/hb_base.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/hb_models.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/hb_val_primesense.zip",
    ]


class YCBV(_BOPBase):
    _img_dir = "test"
    _base_dir = "ycbv"
    _estimated_size = "17GB"
    CLASSES = {
        "002_master_chef_can": 1,
        "003_cracker_box": 2,
        "004_sugar_box": 3,
        "005_tomato_soup_can": 4,
        "006_mustard_bottle": 5,
        "007_tuna_fish_can": 6,
        "008_pudding_box": 7,
        "009_gelatin_box": 8,
        "010_potted_meat_can": 9,
        "011_banana": 10,
        "019_pitcher_base": 11,
        "021_bleach_cleanser": 12,
        "024_bowl": 13,
        "025_mug": 14,
        "035_power_drill": 15,
        "036_wood_block": 16,
        "037_scissors": 17,
        "040_large_marker": 18,
        "051_large_clamp": 19,
        "052_extra_large_clamp": 20,
        "061_foam_brick": 21,
    }
    _urls = [
        "https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_base.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_models.zip",
        "https://bop.felk.cvut.cz/media/data/bop_datasets/ycbv_test_all.zip",
    ]


class HOPE(_BOPBase):
    _img_dir = "val"
    _base_dir = "hope"
    _estimated_size = "322MB"
    CLASSES = {
        "AlphabetSoup": 1,
        "BBQSauce": 2,
        "Butter": 3,
        "Cherries": 4,
        "ChocolatePudding": 5,
        "Cookies": 6,
        "Corn": 7,
        "CreamCheese": 8,
        "GranolaBars": 9,
        "GreenBeans": 10,
        "Ketchup": 11,
        "Macaroni&Cheese": 12,
        "Mayo": 13,
        "Milk": 14,
        "Mushrooms": 15,
        "Mustard": 16,
        "OrangeJuice": 17,
        "Parmesan": 18,
        "Peaches": 19,
        "Peas&Carrots": 20,
        "Pineapple": 21,
        "Popcorn": 22,
        "Raisins": 23,
        "SaladDressing": 24,
        "Spaghetti": 25,
        "TomatoSauce": 26,
        "Tuna": 27,
        "Yogurt": 28,
    }

    _urls = [
        "https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/hope/hope_base.zip",
        "https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/hope/hope_val_realsense.zip",
    ]
