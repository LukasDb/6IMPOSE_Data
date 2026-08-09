"""RenderProduct -> TFRecordWriter -> TFRecordDataset round trip: every field must
come back with identical values, dtypes and shapes."""
import numpy as np
import pytest
import tensorflow as tf

import simpose as sp
from simpose.data.dataset import Dataset
from simpose.writers.tfrecord_writer import TFRecordWriter
from simpose.writers.writer import WriterConfig

from helpers import IMG_H, IMG_W, make_render_product


@pytest.fixture()
def written_dataset(tmp_path):
    rps = {
        0: make_render_product(n_objs=3, seed=0, index=0),
        1: make_render_product(n_objs=0, seed=1, index=1),  # empty annotation list
        2: make_render_product(n_objs=2, seed=2, index=2),
    }
    cfg = WriterConfig(output_dir=tmp_path, overwrite=True, start_index=0, end_index=2)
    with TFRecordWriter(cfg, None) as writer:
        for i, rp in rps.items():
            writer.write_data(i, render_product=rp)
    return tmp_path, rps


def _read_all(root):
    ds = sp.data.TFRecordDataset(root, num_parallel_files=1, pre_shuffle=False)
    return {int(item[Dataset.CAM_LOCATION][0].numpy()): item for item in ds}


def test_full_value_roundtrip(written_dataset):
    root, rps = written_dataset
    items = _read_all(root)
    assert set(items.keys()) == set(rps.keys())

    for i, rp in rps.items():
        item = items[i]

        np.testing.assert_array_equal(item[Dataset.RGB].numpy(), rp.rgb)
        np.testing.assert_array_equal(item[Dataset.RGB_R].numpy(), rp.rgb_R)
        # v4 stores depth as float16, so the roundtrip is exact only after the
        # writer's own narrowing -- compare against that, and assert the dtype so a
        # silent widening back to float32 (which doubled datapoint size) is caught.
        assert item[Dataset.DEPTH].dtype == tf.float16
        assert item[Dataset.DEPTH_R].dtype == tf.float16
        np.testing.assert_array_equal(item[Dataset.DEPTH].numpy(), rp.depth.astype(np.float16))
        np.testing.assert_array_equal(item[Dataset.DEPTH_R].numpy(), rp.depth_R.astype(np.float16))

        assert item[Dataset.MASK].dtype == tf.uint8
        np.testing.assert_array_equal(item[Dataset.MASK].numpy(), rp.mask.astype(np.uint8))

        np.testing.assert_allclose(item[Dataset.CAM_MATRIX].numpy(), rp.intrinsics)
        np.testing.assert_allclose(
            item[Dataset.CAM_LOCATION].numpy(), np.array(rp.cam_position, dtype=np.float32)
        )
        np.testing.assert_allclose(
            item[Dataset.CAM_ROTATION].numpy(), rp.cam_quat_xyzw.astype(np.float32)
        )
        np.testing.assert_allclose(
            item[Dataset.STEREO_BASELINE].numpy(), np.float32(rp.stereo_baseline)
        )

        classes = [b.decode() for b in item[Dataset.OBJ_CLASSES].numpy()]
        assert classes == [o.cls for o in rp.objs]
        np.testing.assert_array_equal(
            item[Dataset.OBJ_IDS].numpy(), [o.object_id for o in rp.objs]
        )
        if len(rp.objs) > 0:
            np.testing.assert_allclose(
                item[Dataset.OBJ_POS].numpy(),
                np.array([o.position for o in rp.objs], dtype=np.float32),
            )
            np.testing.assert_allclose(
                item[Dataset.OBJ_ROT].numpy(),
                np.array([o.quat_xyzw for o in rp.objs], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                item[Dataset.OBJ_BBOX_VISIB].numpy(), [o.bbox_visib for o in rp.objs]
            )
            np.testing.assert_array_equal(
                item[Dataset.OBJ_BBOX_OBJ].numpy(), [o.bbox_obj for o in rp.objs]
            )
            np.testing.assert_array_equal(
                item[Dataset.OBJ_PX_COUNT_VISIB].numpy(), [o.px_count_visib for o in rp.objs]
            )
            np.testing.assert_array_equal(
                item[Dataset.OBJ_PX_COUNT_VALID].numpy(), [o.px_count_valid for o in rp.objs]
            )
            np.testing.assert_array_equal(
                item[Dataset.OBJ_PX_COUNT_ALL].numpy(), [o.px_count_all for o in rp.objs]
            )
            np.testing.assert_allclose(
                item[Dataset.OBJ_VISIB_FRACT].numpy(),
                np.array([o.visib_fract for o in rp.objs], dtype=np.float32),
            )


def test_empty_objs_record_keeps_keys_and_shapes(written_dataset):
    root, _ = written_dataset
    item = _read_all(root)[1]  # the record written with 0 objects

    assert item[Dataset.OBJ_IDS].shape == (0,)
    assert item[Dataset.OBJ_POS].shape == (0, 3)
    assert item[Dataset.OBJ_ROT].shape == (0, 4)
    assert item[Dataset.OBJ_BBOX_VISIB].shape == (0, 4)
    assert item[Dataset.OBJ_BBOX_OBJ].shape == (0, 4)
    assert item[Dataset.OBJ_VISIB_FRACT].shape == (0,)
    # image data of the empty frame is untouched
    assert item[Dataset.RGB].shape == (IMG_H, IMG_W, 3)


def test_mask_ids_over_255_are_rejected(tmp_path):
    rp = make_render_product(n_objs=1, seed=0)
    rp.mask = np.full((IMG_H, IMG_W), 300, dtype=np.float16)
    cfg = WriterConfig(output_dir=tmp_path, overwrite=True, start_index=0, end_index=0)
    with TFRecordWriter(cfg, None) as writer:
        with pytest.raises(ValueError, match="uint8"):
            writer.write_data(0, render_product=rp)


def test_reader_rejects_unknown_version(written_dataset):
    root, _ = written_dataset
    (root / "metadata.json").write_text('{"version": 99}')
    with pytest.raises(ValueError, match="Unsupported dataset version"):
        sp.data.TFRecordDataset(root, pre_shuffle=False)


def test_reader_reports_empty_dataset(tmp_path):
    (tmp_path / "metadata.json").write_text('{"version": 3.0}')
    with pytest.raises(FileNotFoundError, match="No tfrecord files"):
        sp.data.TFRecordDataset(tmp_path, pre_shuffle=False)
