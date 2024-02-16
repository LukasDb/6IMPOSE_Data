import json
from typing import Any
import tensorflow as tf
from pathlib import Path
from .dataset import Dataset


class TFRecordDataset(Dataset):
    @staticmethod
    def get(
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "*.tfrecord",
        num_parallel_files: int = 1024,
    ) -> tf.data.Dataset:
        if root_dir.joinpath("metadata.json").exists():
            with open(root_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"version": 1.0}

        if metadata["version"] < 2:
            return _TFRecordDatasetV1.get(
                root_dir, get_keys=get_keys, pattern=pattern, num_parallel_files=num_parallel_files
            )
        elif metadata["version"] < 3:
            return _TFRecordDatasetV2.get(
                root_dir, get_keys=get_keys, pattern=pattern, num_parallel_files=num_parallel_files
            )


class _TFRecordDatasetV2(Dataset):
    # dtypes of data by key
    _key_mapping = {
        Dataset.RGB: tf.uint8,
        Dataset.RGB_R: tf.uint8,
        Dataset.DEPTH: tf.float32,
        Dataset.DEPTH_R: tf.float32,
        Dataset.MASK: tf.uint8,
        Dataset.CAM_MATRIX: tf.float32,
        Dataset.CAM_LOCATION: tf.float32,
        Dataset.CAM_ROTATION: tf.float32,
        Dataset.STEREO_BASELINE: tf.float32,
        Dataset.OBJ_CLASSES: tf.string,
        Dataset.OBJ_IDS: tf.int32,
        Dataset.OBJ_POS: tf.float32,
        Dataset.OBJ_ROT: tf.float32,
        Dataset.OBJ_BBOX_VISIB: tf.int32,
        Dataset.OBJ_VISIB_FRACT: tf.float32,
        Dataset.OBJ_PX_COUNT_VISIB: tf.int32,
        Dataset.OBJ_PX_COUNT_VALID: tf.int32,
        Dataset.OBJ_PX_COUNT_ALL: tf.int32,
        Dataset.OBJ_BBOX_OBJ: tf.int32,
    }

    @staticmethod
    def get(
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "*.tfrecord",
        num_parallel_files: int = 1024,
    ) -> tf.data.Dataset:

        @tf.function
        def read_tfrecord(record_file: tf.Tensor) -> tf.data.Dataset:
            return tf.data.TFRecordDataset(record_file, compression_type="ZLIB")

        @tf.function
        def parse(example_proto: Any) -> Any:
            proto = {
                k: tf.io.FixedLenFeature([], tf.string)
                for k in _TFRecordDatasetV2._key_mapping.keys()
            }
            serialized = tf.io.parse_single_example(example_proto, proto)
            return {
                key: tf.io.parse_tensor(serialized[key], dtype)
                for key, dtype in _TFRecordDatasetV2._key_mapping.items()
            }

        num_parallel_files = max(1, num_parallel_files)

        return (
            tf.data.Dataset.from_tensor_slices(
                tf.io.match_filenames_once(str(root_dir / "data" / pattern))  # type: ignore
            )
            .interleave(
                read_tfrecord,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
                cycle_length=num_parallel_files,
            )
            .map(parse, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        )


class _TFRecordDatasetV1(Dataset):
    # dtypes of data by key
    _key_mapping = {
        Dataset.RGB: tf.uint8,
        Dataset.RGB_R: tf.uint8,
        Dataset.DEPTH: tf.float32,
        Dataset.DEPTH_R: tf.float32,
        Dataset.MASK: tf.uint8,
        Dataset.CAM_MATRIX: tf.float32,
        Dataset.CAM_LOCATION: tf.float32,
        Dataset.CAM_ROTATION: tf.float32,
        Dataset.STEREO_BASELINE: tf.float32,
        Dataset.OBJ_CLASSES: tf.string,
        Dataset.OBJ_IDS: tf.int32,
        Dataset.OBJ_POS: tf.float32,
        Dataset.OBJ_ROT: tf.float32,
        Dataset.OBJ_BBOX_VISIB: tf.int32,
        Dataset.OBJ_VISIB_FRACT: tf.float32,
        Dataset.OBJ_PX_COUNT_VISIB: tf.int32,
        Dataset.OBJ_PX_COUNT_VALID: tf.int32,
        Dataset.OBJ_PX_COUNT_ALL: tf.int32,
        Dataset.OBJ_BBOX_OBJ: tf.int32,
    }

    # keys per file type
    all_file_keys = {
        "rgb": [Dataset.RGB, Dataset.RGB_R],
        "depth": [Dataset.DEPTH, Dataset.DEPTH_R],
        "mask": [Dataset.MASK],
        "gt": [
            Dataset.CAM_MATRIX,
            Dataset.CAM_LOCATION,
            Dataset.CAM_ROTATION,
            Dataset.STEREO_BASELINE,
            Dataset.OBJ_CLASSES,
            Dataset.OBJ_IDS,
            Dataset.OBJ_POS,
            Dataset.OBJ_ROT,
            Dataset.OBJ_BBOX_VISIB,
            Dataset.OBJ_VISIB_FRACT,
            Dataset.OBJ_PX_COUNT_VISIB,
            Dataset.OBJ_PX_COUNT_VALID,
            Dataset.OBJ_PX_COUNT_ALL,
            Dataset.OBJ_BBOX_OBJ,
        ],
    }

    @staticmethod
    def get(
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "*.tfrecord",
        num_parallel_files: int = 1024,
    ) -> tf.data.Dataset:
        @tf.function
        def read_tfrecord(record_file: tf.Tensor) -> tf.data.Dataset:
            return tf.data.TFRecordDataset(record_file, compression_type="ZLIB")

        def get_parser(keys: list[str]) -> Any:
            @tf.function
            def parse(example_proto: Any) -> Any:
                proto = {k: tf.io.FixedLenFeature([], tf.string) for k in keys}
                serialized = tf.io.parse_single_example(example_proto, proto)
                return {
                    key: tf.io.parse_tensor(serialized[key], TFRecordDataset._key_mapping[key])
                    for key in keys
                }

            return parse

        chosen_keys = get_keys if get_keys is not None else TFRecordDataset._key_mapping.keys()
        file_types = [
            x
            for x in TFRecordDataset.all_file_keys.keys()
            if any(k in chosen_keys for k in TFRecordDataset.all_file_keys[x])
        ]
        # file_types = ["rgb", "gt", "depth"]
        keys_per_file_type = [
            [x for x in chosen_keys if x in TFRecordDataset.all_file_keys[t]] for t in file_types
        ]  # [[rgb, rgb_R], [cam_matrix, obj_classes, etc], [depth]] for example

        parsers = {t: get_parser(keys) for t, keys in zip(file_types, keys_per_file_type)}
        num_parallel_files = max(1, num_parallel_files // len(file_types))
        return tf.data.Dataset.zip(
            tuple(
                tf.data.Dataset.from_tensor_slices(
                    tf.io.match_filenames_once(str(root_dir / file_type / pattern))  # type: ignore
                )
                .interleave(
                    read_tfrecord,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=True,
                    cycle_length=num_parallel_files,
                )
                .map(
                    parsers[file_type], num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
                )  # yields dicts
                for file_type in file_types
            )
        ).map(
            lambda *args: {k: v for d in args for k, v in d.items()},
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
