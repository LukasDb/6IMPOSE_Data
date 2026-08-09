"""The label channels are rasterized in EGL rather than rendered by Blender, so
these check them against analytically known geometry, and check the invariants
that must hold between the instance mask and the per-object annotations.

Covers the two failure modes that are silent (wrong labels, not a crash):
stale `matrix_world` after a transform, and object ids inflated past what the
uint8 mask can hold. Requires bpy; skipped otherwise."""
import contextlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

bpy = pytest.importorskip("bpy")
pytest.importorskip("moderngl")

import simpose as sp

from helpers import write_cube_obj

# level < DEBUG makes sp.redirect_stdout a no-op; its fd-level redirection would
# otherwise fight pytest's output capture
sp.logger.setLevel(1)

IMG_W, IMG_H = 320, 240


@pytest.fixture
def scene(tmp_path):
    sc = sp.Scene(img_h=IMG_H, img_w=IMG_W)
    sc.set_output_path(tmp_path)
    return sc


@pytest.fixture
def camera(scene):
    cam = scene.create_camera("Cam")
    cam.set_from_hfov(70.0, IMG_W, IMG_H, degrees=True)
    return cam


def _light(scene):
    light = scene.create_light("l", energy=200.0, type="POINT")
    light.set_location((0.5, -0.5, 0.5))
    return light


def test_silhouette_matches_analytic_projection(scene, camera, tmp_path):
    """A unit cube face-on at a known distance projects to a square of known
    side length: side_px = fx * size / distance. The unoccluded silhouette
    (bbox_obj / px_count_all) must reproduce that."""
    _light(scene)
    (tmp_path / "models").mkdir(exist_ok=True)
    obj_path = write_cube_obj(tmp_path / "models", name="cube")

    size = 1.0  # write_cube_obj spans 0..1
    obj = scene.create_object(obj_path, "cube", add_semantics=True)
    obj.set_location((-size / 2, -size / 2, -size / 2))  # centre the cube on the origin
    obj.set_rotation(R.identity())

    dist = 5.0
    camera.set_location((0.0, -dist, 0.0))
    camera.point_at(np.zeros(3))

    scene.frame_set(0)
    rp = scene.render(contextlib.nullcontext())

    K = camera.calculate_intrinsics(IMG_W, IMG_H)
    # the near face sits half a cube closer than the centre
    expected_side = K[0, 0] * size / (dist - size / 2)

    ann = rp.objs[0]
    x1, y1, x2, y2 = ann.bbox_obj
    got_w, got_h = x2 - x1 + 1, y2 - y1 + 1

    assert got_w == pytest.approx(expected_side, abs=2.0)
    assert got_h == pytest.approx(expected_side, abs=2.0)
    # a face-on cube is fully visible, so unoccluded == visible
    assert ann.px_count_all == pytest.approx(expected_side**2, rel=0.05)
    assert ann.px_count_visib == ann.px_count_all
    assert ann.visib_fract == pytest.approx(1.0, abs=1e-3)


def test_instance_mask_respects_occlusion(scene, camera, tmp_path):
    """Two cubes on the camera axis: the near one must fully occlude the far one
    in the instance mask, while the far one keeps its full unoccluded silhouette."""
    _light(scene)
    (tmp_path / "models").mkdir(exist_ok=True)
    obj_path = write_cube_obj(tmp_path / "models", name="cube")

    far = scene.create_object(obj_path, "far", add_semantics=True)
    far.set_location((-0.5, 1.0, -0.5))
    near = scene.create_object(obj_path, "near", add_semantics=True)
    near.set_location((-0.5, -1.0, -0.5))

    camera.set_location((0.0, -6.0, 0.0))
    camera.point_at(np.zeros(3))

    scene.frame_set(0)
    rp = scene.render(contextlib.nullcontext())

    anns = {a.object_id: a for a in rp.objs}
    near_ann, far_ann = anns[near.object_id], anns[far.object_id]

    # the near cube is closer, hence larger, and covers the far one entirely
    assert near_ann.px_count_visib > 0
    assert far_ann.px_count_visib == 0, "far cube should be fully occluded"
    assert far_ann.px_count_all > 0, "but its unoccluded silhouette is non-empty"
    assert far_ann.visib_fract == 0.0
    assert near_ann.visib_fract == pytest.approx(1.0, abs=1e-3)

    ids_in_mask = set(int(x) for x in np.unique(rp.mask)) - {0}
    assert ids_in_mask == {near.object_id}


def test_plane_occludes_but_labels_as_background(scene, camera, tmp_path):
    """The floor plane has pass_index 0: it must occlude geometry behind it while
    labelling as background, which is what the EEVEE visib layer used to do."""
    _light(scene)
    (tmp_path / "models").mkdir(exist_ok=True)
    obj_path = write_cube_obj(tmp_path / "models", name="cube")

    scene.create_plane(size=20.0, with_physics=False)  # plane at z=0
    # cube fully below the plane -> hidden from a camera above it
    obj = scene.create_object(obj_path, "cube", add_semantics=True)
    obj.set_location((-0.5, -0.5, -3.0))

    camera.set_location((0.0, 0.0, 5.0))
    camera.point_at(np.zeros(3))

    scene.frame_set(0)
    rp = scene.render(contextlib.nullcontext())

    ann = rp.objs[0]
    assert ann.px_count_visib == 0, "plane must occlude the cube beneath it"
    assert 0 not in {a.object_id for a in rp.objs}, "plane is not an annotated object"


def test_object_ids_are_compacted_for_the_uint8_mask(scene, camera, tmp_path):
    """`get_new_object_id` hands out ids from a monotonic counter and a removed
    object still consumes one, so rejection sampling (as Dropjects does on object
    diameter) inflates ids past the 255 the uint8 mask can hold. They must be
    renumbered to a contiguous 1..N before rendering, and the mask and the
    annotations must agree afterwards."""
    _light(scene)
    (tmp_path / "models").mkdir(exist_ok=True)
    obj_path = write_cube_obj(tmp_path / "models", name="cube")

    kept = []
    for k in range(3):
        for _ in range(100):  # burn ids the way the diameter-rejection loop does
            scene.create_object(obj_path, "tmp", add_semantics=True, hide=True).remove()
        obj = scene.create_object(obj_path, f"keep{k}", add_semantics=True)
        obj.set_location((-0.5 + 1.5 * k, -0.5, -0.5))
        kept.append(obj)

    assert max(o.object_id for o in kept) > 255, "test must reproduce the >255 condition"

    camera.set_location((1.5, -8.0, 0.0))
    camera.point_at(np.array([1.5, 0.0, 0.0]))
    scene.frame_set(0)
    rp = scene.render(contextlib.nullcontext())

    ids = sorted(o.object_id for o in scene.get_active_objects())
    assert ids == list(range(1, len(ids) + 1)), "ids must be compacted to 1..N"
    assert int(rp.mask.max()) <= 255

    # the writer rejects mask ids over 255, so this is what used to fail
    ann_ids = {a.object_id for a in rp.objs}
    mask_ids = set(int(x) for x in np.unique(rp.mask)) - {0}
    assert mask_ids <= ann_ids


def test_scale_is_reflected_in_labels_on_the_very_first_render(scene, camera, tmp_path):
    """`matrix_world` is stale until the depsgraph re-evaluates, and nothing flushes
    it between `create_object(scale=...)` and the first render. If the labels are
    built from the stale matrix the object's `scale` is silently ignored - which
    reads as a mesh 1/scale times too large (this is how the omni3d models, whose
    scale is 0.001, ended up projecting completely wrongly).

    A later render would mask the bug, because rendering flushes the depsgraph - so
    this must assert on the FIRST render of a freshly created, scaled object.
    Note there is deliberately no `frame_set` before the render either: that also
    flushes the depsgraph, and `_upload_geometry` runs before it."""
    _light(scene)
    (tmp_path / "models").mkdir(exist_ok=True)
    obj_path = write_cube_obj(tmp_path / "models", name="cube")

    scale = 0.25
    obj = scene.create_object(obj_path, "cube", add_semantics=True, scale=scale)
    # write_cube_obj spans 0..1 locally, so 0..scale in world -> centre it
    obj.set_location((-scale / 2, -scale / 2, -scale / 2))
    obj.set_rotation(R.identity())

    dist = 5.0
    camera.set_location((0.0, -dist, 0.0))
    camera.point_at(np.zeros(3))

    ann = scene.render(contextlib.nullcontext()).objs[0]

    K = camera.calculate_intrinsics(IMG_W, IMG_H)
    expected_side = K[0, 0] * scale / (dist - scale / 2)

    x1, y1, x2, y2 = ann.bbox_obj
    got_side = x2 - x1 + 1
    assert got_side == pytest.approx(expected_side, abs=2.0), (
        f"silhouette is {got_side}px but a cube scaled to {scale} at {dist}m projects "
        f"to ~{expected_side:.1f}px (a stale matrix_world would ignore the scale)"
    )


def test_render_batch_reports_each_frame_and_keeps_cycles_resident(scene, camera, tmp_path):
    """render_batch must invoke the callback per frame (so writes and progress
    happen as frames complete) and must never switch the render engine, which is
    what keeps Cycles' use_persistent_data cache warm."""
    _light(scene)
    (tmp_path / "models").mkdir(exist_ok=True)
    obj_path = write_cube_obj(tmp_path / "models", name="cube")
    obj = scene.create_object(obj_path, "cube", add_semantics=True)
    obj.set_location((-0.5, -0.5, -0.5))

    camera.set_location((0.0, -5.0, 0.0))
    camera.point_at(np.zeros(3))

    seen: list[int] = []
    indices = np.arange(3)
    products = scene.render_batch(
        indices, contextlib.nullcontext(), on_frame_rendered=lambda i, rp: seen.append(i)
    )

    assert seen == list(indices), "callback must fire once per frame, in order"
    assert len(products) == len(indices)
    assert scene._bl_scene.render.engine == "CYCLES"
    # the per-object EEVEE view layers are gone; only the beauty layer remains
    assert [l.name for l in scene._bl_scene.view_layers] == ["ViewLayer"]
    assert not (tmp_path / "mask").exists() or not list((tmp_path / "mask").iterdir())


def test_render_batch_tolerates_a_writer_without_a_semaphore(scene, camera, tmp_path):
    """A Writer built without a comm has gpu_semaphore None; that must not crash."""
    _light(scene)
    (tmp_path / "models").mkdir(exist_ok=True)
    obj_path = write_cube_obj(tmp_path / "models", name="cube")
    obj = scene.create_object(obj_path, "cube", add_semantics=True)
    obj.set_location((-0.5, -0.5, -0.5))
    camera.set_location((0.0, -5.0, 0.0))
    camera.point_at(np.zeros(3))

    products = scene.render_batch(np.array([0]), None)  # type: ignore[arg-type]
    assert products[0].mask is not None
