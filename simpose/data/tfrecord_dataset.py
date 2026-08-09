import json
from typing import Any
import tensorflow as tf
from pathlib import Path
from .dataset import Dataset
import logging
import random


class TFRecordDataset(Dataset):

    def __new__(
        cls,
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "**/*.tfrecord",
        num_parallel_files: int = 16,
        pre_shuffle: bool = True,
        safe: bool = True,
    ):

        # try to load meta info; for a concatenation root without its own
        # metadata.json, use the first dataset's metadata found below it
        metadata_paths = [root_dir / "metadata.json"] + sorted(
            root_dir.glob("*/**/metadata.json")
        )
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                break
        else:
            metadata = {"version": 2.0}

        version = metadata.get("version")
        if version is None:
            raise ValueError(f"metadata.json under {root_dir} has no 'version' key.")

        if version < 2:
            return _TFRecordDatasetV1(
                root_dir, get_keys=get_keys, pattern=pattern, num_parallel_files=num_parallel_files
            )
        elif version < 5:
            # v3 == v2 except the mask is uint8; v4 == v3 except depth is float16
            return _TFRecordDatasetV2(
                root_dir,
                get_keys=get_keys,
                pattern=pattern,
                num_parallel_files=num_parallel_files,
                shuffle_files=pre_shuffle,
                safe=safe,
                mask_dtype=tf.int32 if version < 3 else tf.uint8,
                depth_dtype=tf.float32 if version < 4 else tf.float16,
            )
        raise ValueError(f"Unsupported dataset version {version} under {root_dir}.")

    @staticmethod
    def get(
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "**/*.tfrecord",
        num_parallel_files: int = 16,
        pre_shuffle: bool = True,
        safe: bool = True,
    ) -> tf.data.Dataset:
        """legacy function"""
        return TFRecordDataset(
            root_dir, get_keys, pattern, num_parallel_files, pre_shuffle, safe=safe
        )


class _TFRecordDatasetV2(Dataset):
    # dtypes of data by key
    _key_mapping = {
        Dataset.RGB: tf.uint8,
        Dataset.RGB_R: tf.uint8,
        Dataset.DEPTH: tf.float32,
        Dataset.DEPTH_R: tf.float32,
        Dataset.DEPTH_GT: tf.float32,
        Dataset.DEPTH_GT_R: tf.float32,
        Dataset.MASK: tf.int32,
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

    def __new__(
        cls,
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "**/*.tfrecord",
        num_parallel_files: int = 16,
        shuffle_files: bool = True,
        safe: bool = True,
        mask_dtype: Any = None,
        depth_dtype: Any = None,
    ) -> tf.data.Dataset:

        key_mapping = dict(_TFRecordDatasetV2._key_mapping)
        if mask_dtype is not None:
            key_mapping[Dataset.MASK] = mask_dtype
        if depth_dtype is not None:
            for k in (Dataset.DEPTH, Dataset.DEPTH_R, Dataset.DEPTH_GT, Dataset.DEPTH_GT_R):
                key_mapping[k] = depth_dtype

        if get_keys is None:
            get_keys = list(key_mapping.keys())

        # check keys if in dataset

        # file_names = tf.io.match_filenames_once(str(root_dir / "data" / pattern))
        file_names = list(str(x) for x in root_dir.glob(pattern))
        if len(file_names) == 0:
            raise FileNotFoundError(
                f"No tfrecord files found under {root_dir} with pattern '{pattern}'."
            )

        if shuffle_files:
            file_names = tf.random.shuffle(file_names)

        if safe:
            record = (
                tf.data.TFRecordDataset(
                    file_names[0],
                    compression_type="ZLIB",
                )
                .take(1)
                .get_single_element()
            )
            to_be_removed = []
            for key in get_keys:
                try:
                    proto = {key: tf.io.FixedLenFeature([], tf.string)}
                    serialized = tf.io.parse_single_example(record, proto)
                except Exception:
                    to_be_removed.append(key)
            get_keys = [x for x in get_keys if x not in to_be_removed]

        @tf.function
        def parse(example_proto: Any) -> Any:
            proto = {k: tf.io.FixedLenFeature([], tf.string) for k in get_keys}
            serialized = tf.io.parse_single_example(example_proto, proto)
            return {
                key: tf.io.parse_tensor(serialized[key], key_mapping[key]) for key in get_keys
            }

        num_parallel_files = max(1, num_parallel_files)

        dataset = tf.data.TFRecordDataset(
            file_names,
            compression_type="ZLIB",
            num_parallel_reads=num_parallel_files,
        ).map(
            parse,
            num_parallel_calls=num_parallel_files,
            deterministic=not shuffle_files,
            synchronous=False,
            use_unbounded_threadpool=True,
        )

        return dataset

    @staticmethod
    def get(
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "**/*.tfrecord",
        num_parallel_files: int = 16,
        shuffle_files: bool = True,
    ) -> tf.data.Dataset:
        return _TFRecordDatasetV2(
            root_dir,
            get_keys=get_keys,
            pattern=pattern,
            num_parallel_files=num_parallel_files,
            shuffle_files=shuffle_files,
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

    def __new__(
        cls,
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "*.tfrecord",
        num_parallel_files: int = 16,
    ) -> tf.data.Dataset:

        if get_keys is None:
            get_keys = list(_TFRecordDatasetV1._key_mapping.keys())

        # check keys if in dataset
        to_be_removed = []
        for key in get_keys:
            file_type = [k for k, v in _TFRecordDatasetV1.all_file_keys.items() if key in v][0]
            record = (
                tf.data.TFRecordDataset(
                    tf.io.match_filenames_once(str(root_dir / file_type / pattern)),
                    compression_type="ZLIB",
                )
                .take(1)
                .get_single_element()
            )

            try:
                proto = {key: tf.io.FixedLenFeature([], tf.string)}
                serialized = tf.io.parse_single_example(record, proto)
            except Exception:
                # logging.getLogger(__name__).warning(f"Key {key} not found in dataset")
                to_be_removed.append(key)
        get_keys = [x for x in get_keys if x not in to_be_removed]

        @tf.function
        def read_tfrecord(record_file: tf.Tensor) -> tf.data.Dataset:
            return tf.data.TFRecordDataset(record_file, compression_type="ZLIB")

        def get_parser(keys: list[str]) -> Any:
            @tf.function
            def parse(example_proto: Any) -> Any:
                proto = {k: tf.io.FixedLenFeature([], tf.string) for k in keys}
                serialized = tf.io.parse_single_example(example_proto, proto)
                return {
                    key: tf.io.parse_tensor(serialized[key], _TFRecordDatasetV1._key_mapping[key])
                    for key in keys
                }

            return parse

        file_types = [
            x
            for x in _TFRecordDatasetV1.all_file_keys.keys()
            if any(k in get_keys for k in _TFRecordDatasetV1.all_file_keys[x])
        ]
        # file_types = ["rgb", "gt", "depth"]
        keys_per_file_type = [
            [x for x in get_keys if x in _TFRecordDatasetV1.all_file_keys[t]] for t in file_types
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

    @staticmethod
    def get(
        root_dir: Path,
        get_keys: None | list[str] = None,
        pattern: str = "*.tfrecord",
        num_parallel_files: int = 16,
    ) -> tf.data.Dataset:
        return _TFRecordDatasetV1(root_dir, get_keys, pattern, num_parallel_files)
