"""Shared fixtures/helpers for the round-trip integration tests."""
from pathlib import Path

import numpy as np

import simpose as sp

IMG_H, IMG_W = 32, 48


def make_render_product(
    n_objs: int, seed: int = 0, index: int | None = None, stereo: bool = True
) -> sp.RenderProduct:
    """deterministic RenderProduct; `index` is encoded into cam_position[0] so a
    record can be identified after a write -> read round trip"""
    rng = np.random.default_rng(seed)
    objs = [
        sp.ObjectAnnotation(
            cls=f"obj_{i}",
            object_id=i + 1,
            position=rng.uniform(-1, 1, 3),
            quat_xyzw=_random_quat(rng),
            bbox_visib=(1, 2, 3 + i, 4 + i),
            bbox_obj=(0, 1, 5 + i, 6 + i),
            px_count_visib=10 + i,
            px_count_valid=8 + i,
            px_count_all=12 + i,
            visib_fract=(10 + i) / (12 + i),
        )
        for i in range(n_objs)
    ]
    return sp.RenderProduct(
        rgb=rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        rgb_R=rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8) if stereo else None,
        depth=rng.random((IMG_H, IMG_W)).astype(np.float32),
        depth_R=rng.random((IMG_H, IMG_W)).astype(np.float32) if stereo else None,
        # float16, like the EXR readback in Scene.render
        mask=rng.integers(0, n_objs + 1, (IMG_H, IMG_W)).astype(np.float16),
        objs=objs,
        intrinsics=np.array([[100.0, 0, 24], [0, 100.0, 16], [0, 0, 1]], dtype=np.float32),
        cam_position=(float(index if index is not None else 0), 0.0, 1.0),
        cam_quat_xyzw=_random_quat(rng),
        stereo_baseline=0.063 if stereo else None,
    )


def _random_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def write_shard(output_dir: Path, start: int, end: int) -> None:
    """write one complete shard covering indices start..end (inclusive)"""
    from simpose.writers.tfrecord_writer import TFRecordWriter
    from simpose.writers.writer import WriterConfig

    cfg = WriterConfig(output_dir=output_dir, overwrite=True, start_index=start, end_index=end)
    with TFRecordWriter(cfg, None) as writer:
        for i in range(start, end + 1):
            writer.write_data(i, render_product=make_render_product(n_objs=2, seed=i, index=i))


CUBE_OBJ = """mtllib {mtl_name}
{vertices}
usemtl test_mat
f 1 2 4 3
f 5 7 8 6
f 1 5 6 2
f 3 4 8 7
f 1 3 7 5
f 2 6 8 4
"""

CUBE_MTL = """newmtl test_mat
Kd 0.8 0.2 0.2
Ns 250
"""


def write_cube_obj(directory: Path, name: str = "cube", offset: float = 0.0) -> Path:
    """unit cube OBJ with a material; offset shifts all vertices, so the center of
    volume (and therefore the COM handling) is non-trivial when offset != 0"""
    verts = []
    for x in (0.0, 1.0):
        for y in (0.0, 1.0):
            for z in (0.0, 1.0):
                verts.append(f"v {x + offset} {y + offset} {z + offset}")
    obj_path = directory / f"{name}.obj"
    mtl_path = directory / f"{name}.mtl"
    obj_path.write_text(CUBE_OBJ.format(mtl_name=mtl_path.name, vertices="\n".join(verts)))
    mtl_path.write_text(CUBE_MTL)
    return obj_path
