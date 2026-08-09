"""EXR write -> read round trip (single- and multi-channel, all supported dtypes)."""
import numpy as np
import pytest

from simpose.exr import EXR


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.uint32])
def test_single_channel_roundtrip(tmp_path, dtype):
    if np.issubdtype(dtype, np.integer):
        img = np.arange(32 * 48, dtype=dtype).reshape(32, 48)
    else:
        img = np.linspace(0, 1, 32 * 48, dtype=dtype).reshape(32, 48)

    exr = EXR(tmp_path / "test.exr")
    exr.write({"chan.R": img})
    out = exr.read("chan.R")

    assert out.dtype == dtype
    np.testing.assert_array_equal(out, img)


def test_multichannel_single_open_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    channels = {f"{i:04}.R": rng.random((16, 24)).astype(np.float32) for i in range(5)}

    exr = EXR(tmp_path / "multi.exr")
    exr.write(channels)
    out = exr.read_channels(list(channels.keys()))

    assert set(out.keys()) == set(channels.keys())
    for key, img in channels.items():
        np.testing.assert_array_equal(out[key], img)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EXR(tmp_path / "nope.exr").read("chan.R")


def test_missing_channel_raises(tmp_path):
    exr = EXR(tmp_path / "test.exr")
    exr.write({"a.R": np.zeros((4, 4), dtype=np.float32)})
    with pytest.raises(KeyError):
        exr.read("b.R")
