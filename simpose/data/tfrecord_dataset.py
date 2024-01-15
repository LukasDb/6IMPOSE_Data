from typing import Any
import tensorflow as tf
from pathlib import Path
from .dataset import Dataset


class TFRecordDataset(Dataset):
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
        deterministic: bool = False,
        pattern: str = "*.tfrecord",
    ) -> tf.data.Dataset:
        """create a tf.data.Dataset from 6IMPOSE tfrecord dataset. Returns a dict with the specified keys"""
        if get_keys is not None:
            # filter keymap
            key_map = {k: v for k, v in TFRecordDataset._key_mapping.items() if k in get_keys}
        else:
            key_map = TFRecordDataset._key_mapping

        per_file_keys = {
            file_key: [x for x in TFRecordDataset.all_file_keys[file_key] if x in key_map]
            for file_key in TFRecordDataset.all_file_keys
            if any(x in key_map for x in TFRecordDataset.all_file_keys[file_key])
        }

        # create a parsed dataset per file type
        ds: list[tf.data.Dataset] = []
        for name, keys in per_file_keys.items():
            files = tf.io.matching_files(str(root_dir / name / pattern))  # type: ignore
            shards = tf.data.Dataset.from_tensor_slices(files)

            def parse_tfrecord(example_proto: Any) -> Any:
                return tf.io.parse_single_example(
                    example_proto, TFRecordDataset._get_proto_from_keys(keys)
                )

            # has to be deterministic, otherwise the order of the shards is random
            tf_ds = shards.interleave(
                lambda x: tf.data.TFRecordDataset(x, compression_type="ZLIB"),
                deterministic=True,
                num_parallel_calls=tf.data.AUTOTUNE,
            ).map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

            ds.append(tf_ds)

        parse_to_tensors = TFRecordDataset._parse_to_tensors(key_map, per_file_keys)

        dataset = tf.data.Dataset.zip(tuple(ds)).map(  # type: ignore
            parse_to_tensors, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic
        )
        return dataset

    @staticmethod
    def _get_proto_from_keys(keys: list[str]) -> dict[str, tf.io.FixedLenFeature]:
        return {k: tf.io.FixedLenFeature([], tf.string) for k in keys}

    @staticmethod
    def _parse_to_tensors(
        key_map: dict[str, tf.dtypes.DType], per_file_keys: dict[str, list[str]]
    ) -> Any:
        """creates a map function that parses the tfrecord dataset to a dict of tensors"""

        def parser(*args: dict) -> dict[str, tf.Tensor]:
            return {
                key: tf.io.parse_tensor(data_dict[key], key_map[key])
                for data_dict, keys in zip(args, per_file_keys.values())
                for key in keys
            }

        return parser
