# Round-trip integration tests

Run with:

```bash
python -m pytest tests/ -v
```

Requires the generation environment (`tensorflow`, `bpy`, `pybullet`, `OpenEXR`) plus
`pytest`. The Blender-backed tests are skipped automatically if `bpy` is not importable.

| File | Round trip covered |
|---|---|
| `test_exr_roundtrip.py` | `EXR.write` → `EXR.read`/`read_channels` (all dtypes, multichannel single-open) |
| `test_writer_reader_roundtrip.py` | `RenderProduct` → `TFRecordWriter` (v3) → `TFRecordDataset`: exact value/dtype/shape equality per field, empty-annotation frames, >255 mask-id rejection, reader error paths |
| `test_resume_roundtrip.py` | partial run → `get_pending_indices` → resumed run → exactly one record per index; truncated and short shards detected and regenerated; `overwrite: true` bypass |
| `test_config_roundtrip.py` | `generate_template_config()` → YAML → pydantic validation for every generator in `__generators__`, plus the checked-in `dropjects.yml` |
| `test_blender_roundtrips.py` | camera K → Blender shifts/lens → K (off-center principal point); Blender pose → PyBullet → Blender (COM offset algebra); quaternion `[x,y,z,w]` convention; camera fill-ratio geometry; atomic V-HACD cache |
| `test_egl_labels.py` | scene geometry → EGL rasterizer → mask/annotations: silhouette vs analytic projection, occlusion ordering in the instance mask, the floor plane occluding as background, object-id compaction for the uint8 mask, `scale` honoured on the first render, `render_batch` per-frame callback + engine/view-layer state |

Notes:

- The `Error: Not freed memory blocks` line after a run is Blender's exit-time leak
  report, not a test failure.
- `test_blender_roundtrips.py` sets the simpose logger below DEBUG so
  `sp.redirect_stdout` no-ops; its fd-level redirection conflicts with pytest capture.
- `test_egl_labels.py` additionally needs `moderngl` and a working EGL device. Two of
  its cases guard *silent* label corruption and were verified to fail when the
  corresponding guard is removed: dropping `Scene._compact_object_ids` yields ids
  `[101, 202, 303]` instead of `1..N`, and dropping the `view_layer.update()` in
  `Scene._upload_geometry` makes a scaled object's silhouette 57 px instead of 12 px.
  Keep them assertion-tight — both bugs pass a laxer test.
