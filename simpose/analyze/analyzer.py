from pathlib import Path
import matplotlib.pyplot as plt
import json
import multiprocessing as mp
import os
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation as R
import simpose as sp
import time
import tensorflow as tf
from tqdm import tqdm


class Analyzer:
    keys = [
        sp.data.Dataset.RGB,
        sp.data.Dataset.OBJ_BBOX_VISIB,
        sp.data.Dataset.OBJ_CLASSES,
        sp.data.Dataset.CAM_LOCATION,
        sp.data.Dataset.OBJ_POS,
    ]

    def __init__(self, dataset_root: Path, fraction: float = 1.0) -> None:
        self.dataset_root = dataset_root
        self._fraction = fraction
        self._statistics: None | dict = None

    def _collect_statistics(self) -> None:
        """is called if statistics are not in cache there"""
        # t = time.perf_counter()
        self._compute_statistics()
        # print(f"Statistics computed in {time.perf_counter() - t:.2f}s")

    @staticmethod
    @tf.function
    def quat_to_matrix(quat):
        x, y, z, w = quat[0], quat[1], quat[2], quat[3]
        tx = 2.0 * x
        ty = 2.0 * y
        tz = 2.0 * z
        twx = tx * w
        twy = ty * w
        twz = tz * w
        txx = tx * x
        txy = ty * x
        txz = tz * x
        tyy = ty * y
        tyz = tz * y
        tzz = tz * z
        matrix = tf.stack(
            (
                1.0 - (tyy + tzz),
                txy - twz,
                txz + twy,
                txy + twz,
                1.0 - (txx + tzz),
                tyz - twx,
                txz - twy,
                tyz + twx,
                1.0 - (txx + tyy),
            ),
            axis=-1,
        )  # pyformat: disable
        output_shape = tf.concat((tf.shape(input=quat)[:-1], (3, 3)), axis=-1)
        return tf.reshape(matrix, shape=output_shape)

    def save_pdf(self, path: Path) -> None:
        """saves statistics as PDF (matplotlib figuers)"""
        if self._statistics is None:
            self._collect_statistics()

        fig, axs = plt.subplots(1, 2, figsize=(10, 8))

        # Plot 1
        # just not that interesting
        # ns_objects = self._statistics["ns_objects"]
        # axs[0, 0].hist(ns_objects, bins=np.arange(0, np.max(ns_objects) + 1))
        # axs[0, 0].set_title("Number of Images with n Objects")
        # axs[0, 0].set_xlabel("Number of Objects")
        # axs[0, 0].set_ylabel("Number of Images")

        # # Plot 2
        # ns_classes = self._statistics["ns_classes"]
        # axs[0, 1].hist(ns_classes, bins=np.arange(0, np.max(ns_classes) + 1))
        # axs[0, 1].set_title("Number of Images with n Classes")
        # axs[0, 1].set_xlabel("Number of Classes")
        # axs[0, 1].set_ylabel("Number of Images")

        # Plot 3
        row = 0
        mean_dist = self._statistics["mean_dist_to_camera"]
        axs[0].hist(self._statistics["distances_to_camera"], bins=np.linspace(0, 1.5, 20))
        axs[0].set_title("Distances to Camera")
        axs[0].set_xlabel("Distance to Camera [m]")
        axs[0].set_ylabel("Number of Objects")
        axs[0].axvline(mean_dist, color="r", linestyle="--", label=f"mean: {mean_dist:.2f}")

        # Plot 4
        mean_side = self._statistics["mean_longest_side"]
        axs[1].hist(self._statistics["longest_sides"], bins=np.linspace(0, 1.0, 20))
        axs[1].set_title("Longest Sides")
        axs[1].set_xlabel("Longest Side [%]")
        axs[1].set_ylabel("Number of Objects")
        axs[1].axvline(mean_side, color="r", linestyle="--", label=f"mean: {mean_side:.2f}")

        # add title to figure
        # fig.suptitle(f"Statistics for {self.dataset_root.name}")

        # set figure size to square
        fig.set_size_inches(5.5, 2)

        # add grid
        for ax in axs:
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)

    def print(self) -> None:
        """prints statistics in human-readable format to the CLI"""
        print(f"Statistics for {self.dataset_root}")
        if self._statistics is None:
            self._collect_statistics()
        stats = self._statistics

        print(f"mean distance to camera: {stats['mean_dist_to_camera']}")
        print(f"std of object distance: {np.std(stats['distances_to_camera'])}")
        print(f"mean number of classes: {stats['mean_n_classes']}")
        print(f"mean number of objects: {stats['mean_n_objects']}")
        print(f"mean object side in %: {stats['mean_longest_side']}")
        print(f"std of object side: {np.std(stats['longest_sides'])}")
        print(f"total images: {stats['total_images']}")
        print(f"total objects: {stats['total_objects']}")

    def print_full(self) -> None:
        if self._statistics is None:
            self._collect_statistics()
        stats = self._statistics
        self._plot_histogram_to_cli(stats["ns_objects"], "num_images_with_n_objects")
        self._plot_histogram_to_cli(stats["ns_classes"], "num_images_with_n_classes")
        self._plot_histogram_to_cli(stats["distances_to_camera"], "distances_to_camera")
        self._plot_histogram_to_cli(stats["longest_sides"], "longest_sides")
        self.print()

    def _plot_histogram_to_cli(self, values: np.ndarray, name: str) -> None:
        # find out current terminal width
        width = os.get_terminal_size().columns
        hist = np.histogram(values, bins="auto")
        max_bar_width = width - (6 + 2 + 4 + 2)  # for the key and value
        max_value = np.max(hist[0])
        scale_factor = max_bar_width / max_value

        print(f"{name}:")
        for i, (count, bin) in enumerate(zip(hist[0], hist[1])):

            bar_length = np.floor(scale_factor * count).astype(int)
            key = f"{bin:6.3f}" if isinstance(bin, float) else f"{int(bin):6d}"
            value = f"{count:4.1f}"

            print(f"{key}: {value} {'█' * bar_length}")
        print()

    def _compute_statistics(self) -> None:
        # parallely process the tfrecord files (only take first n samples of each file)
        tfrecords = list(self.dataset_root.glob("**/*.tfrecord"))
        # print(tfrecords)
        # print(f"Found {len(tfrecords)} tfrecords.")

        # self.process_tfrecord(tfrecords[0])
        # return

        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(self.process_tfrecord, tfrecords)

        # results = [self.process_tfrecord(tfrecord) for tfrecord in tqdm(tfrecords)]

        distances_to_camera = [x["distances_to_camera"] for x in results]
        ns_objects = [x["ns_objects"] for x in results]
        ns_classes = [x["ns_classes"] for x in results]
        longest_sides = [x["longest_sides"] for x in results]
        n_total_images = sum([x["n_total_images"] for x in results])
        n_total_objects = sum([x["n_total_objects"] for x in results])

        distances_to_camera = np.concatenate(distances_to_camera).flatten()
        longest_sides = np.concatenate(longest_sides).flatten()
        ns_objects = np.concatenate(ns_objects).flatten()
        ns_classes = np.concatenate(ns_classes).flatten()

        mean_dist = np.mean(distances_to_camera)
        mean_longest_side = np.mean(longest_sides)
        mean_n_classes = np.mean(ns_classes)
        mean_n_objects = np.mean(ns_objects)

        self._statistics = {
            "ns_objects": ns_objects,
            "ns_classes": ns_classes,
            "distances_to_camera": distances_to_camera,
            "longest_sides": longest_sides,
            "mean_dist_to_camera": mean_dist,
            "total_images": n_total_images,
            "total_objects": n_total_objects,
            "mean_n_classes": mean_n_classes,
            "mean_n_objects": mean_n_objects,
            "mean_longest_side": mean_longest_side,
        }

    # @tf.function
    def _extract_statistics(self, data: dict[str, tf.Tensor]) -> dict[str, int]:
        """extracts statistics from a single data point"""

        has_valid_bbox = tf.reduce_any(data[sp.data.Dataset.OBJ_BBOX_VISIB] > 0, axis=-1)
        visible_indices = tf.where(has_valid_bbox)[:, 0]  # (n,)

        boxes = tf.gather(data[sp.data.Dataset.OBJ_BBOX_VISIB], visible_indices)  # (n,4)
        h, w = tf.shape(data[sp.data.Dataset.RGB])[0], tf.shape(data[sp.data.Dataset.RGB])[1]

        bbox_widths = (boxes[:, 2] - boxes[:, 0]) / w
        bbox_heights = (boxes[:, 3] - boxes[:, 1]) / h
        longest_side = tf.maximum(bbox_heights, bbox_widths)

        n_visible_objects = tf.math.count_nonzero(has_valid_bbox)

        classes = data[sp.data.Dataset.OBJ_CLASSES]
        visible_classes, _ = tf.unique(tf.gather(classes, visible_indices))
        n_visible_classes = tf.shape(visible_classes)[0]

        # dist2cam
        cam_pos = data[sp.data.Dataset.CAM_LOCATION]  # (3, )
        obj_pos = data[sp.data.Dataset.OBJ_POS]  # (n_objects, 3)
        obj_pos = tf.gather(obj_pos, visible_indices)  # (n_visible_objects, 3)

        t = obj_pos - cam_pos[tf.newaxis, :]  # (n_visible_objects, 3)
        dists = tf.norm(t, axis=-1)  # (n_visible_objects, )

        return {
            "n_visible_objects": n_visible_objects,
            "n_visible_classes": n_visible_classes,
            "longest_side": longest_side,
            "dists": dists,
            "classes": visible_classes,
        }

    def process_tfrecord(self, tfrecord: Path):
        mp.get_logger().setLevel("WARN")
        stats_folder = tfrecord.parent.parent.joinpath("statistics")
        stats_folder.mkdir(exist_ok=True)
        stats_path = stats_folder.joinpath(tfrecord.stem).with_suffix(".json")

        if stats_path.exists():
            with stats_path.open("r") as f:
                return json.load(f)

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        n_total_objects = 0
        n_total_images = 0
        ns_objects = []
        ns_classes = []
        distances_to_camera = []
        longest_sides = []

        start, stop = tfrecord.stem.split("_")
        n_images = int(stop) - int(start)
        img_limit = max(1, int(self._fraction * n_images))
        print(f"[{mp.current_process()}] Processing {tfrecord.stem} [{img_limit}/{n_images}]")

        # # debug
        # data = (
        #     sp.data.TFRecordDataset.get(
        #         tfrecord.parent.parent,
        #         pattern=tfrecord.name,
        #         num_parallel_files=1,
        #         get_keys=self.keys,
        #     )
        #     .take(1)
        #     .get_single_element()
        # )
        # result = self._extract_statistics(data)
        # exit()
        # # /debug

        try:
            ds = (
                sp.data.TFRecordDataset.get(
                    tfrecord.parent.parent,  # folder of record -> parent of folder
                    pattern=tfrecord.name,
                    num_parallel_files=1,
                    get_keys=self.keys,
                )
                .take(img_limit)
                .map(self._extract_statistics, num_parallel_calls=4, deterministic=False)
            )
            # print(f"[{mp.current_process()}] DS initialized.")
            for result in ds:
                n_objects = result["n_visible_objects"].numpy()
                ns_objects.append(n_objects)

                n_total_objects += n_objects
                n_total_images += 1

                n_classes = result["n_visible_classes"].numpy()
                ns_classes.append(n_classes)

                dists_img = result["dists"].numpy()
                distances_to_camera += dists_img.tolist()

                longest_sides_img = result["longest_side"].numpy()
                longest_sides += longest_sides_img.tolist()
                # print(f"[{mp.current_process()}] {n_total_images} processed.")
        except KeyError:
            print(f"KeyError in {tfrecord.parent.parent}")

        output = {
            "n_total_images": n_total_images,
            "n_total_objects": n_total_objects,
            "ns_objects": ns_objects,
            "ns_classes": ns_classes,
            "distances_to_camera": distances_to_camera,
            "longest_sides": longest_sides,
        }

        print(f"Processed {tfrecord.stem}")

        with stats_path.open("w") as f:
            json.dump(output, f, default=lambda x: x.tolist())

        return output
