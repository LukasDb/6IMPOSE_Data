# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`6IMPOSE_Data` (import name: `simpose`) is a synthetic 6D-pose dataset pipeline. It has two halves that share one vocabulary of keys:

1. **Generation** — drives Blender (`bpy`) + PyBullet headlessly to render RGB/depth/mask + 6D-pose labels, and writes sharded ZLIB-compressed TFRecords.
2. **Consumption** — `tf.data` loaders for the generated datasets and for BOP-challenge datasets (auto-download + preprocess into the same TFRecord format).

## Commands

```bash
pip install -e .                          # OpenEXR often needs: conda install -c conda-forge openexr-python

simpose generate -i my_config.yml         # interactively scaffold a config (prompts for Generator type)
simpose generate my_config.yml -vv        # run generation; -v/-vv/-vvv → log level 20/10/1
simpose generate my_config.yml --start_index 0 --end_index 999   # hidden overrides of Writer indices
simpose view <dataset_dir>                # Streamlit dataset viewer (concatenates all datasets found)
simpose download <output_dir> {YCB,Omni3D}
simpose analyze <dataset_dir> --fraction 0.1 -o report.pdf
```

```bash
python -m pytest tests/ -v                # round-trip integration tests
python -m pytest tests/test_exr_roundtrip.py::test_name -v   # single test
```

Tests need the generation env (`tensorflow`, `bpy`, `pybullet`, `OpenEXR`) plus `pytest`; the Blender-backed ones skip themselves if `bpy` is missing. See [tests/README.md](tests/README.md) for what each file covers. `Error: Not freed memory blocks` after a run is Blender's exit-time leak report, not a failure.

There is no linter config and no CI. The code is fully type-annotated in mypy style (`# type: ignore` is used liberally around `bpy`), so match that when editing.

[AUDIT_DECISIONS.md](AUDIT_DECISIONS.md) records the reviewed-and-decided fixes (A/B/C items) behind much of the current behaviour — check it before "fixing" something that looks odd.

## Architecture

### Config-driven plugin resolution by naming convention

Everything is wired from a YAML config with three top-level sections (`Generator`, `Writer`, `Randomizers`), and classes are resolved by `getattr` on the package plus a `Config` suffix:

- `config["Generator"]["type"]` → `sp.generators.<Type>` and `sp.generators.<Type>Config`
- `config["Writer"]["type"]` → `sp.writers.<Type>`, params validated by `sp.writers.WriterConfig`
- each randomizer's `type` → `sp.random.<Type>` and `sp.random.<Type>Config`

So **adding a component means: define `FooConfig(<Base>Config)` + `Foo(<Base>)`, export both from the subpackage `__init__.py`** (and add the generator name to `sp.generators.__generators__`). All configs derive from `base_config.BaseConfig` (pydantic, `extra="forbid"`); `get_description()` supplies the inline `#` comments that `dump_with_comments()` emits when scaffolding a config, and `generate_template_config()` on each Generator assembles the full template string.

The special randomizer type `Join` recursively builds children and folds them with `+` — only `JoinableRandomizer` subclasses (currently `ModelLoader`) support this. `ModelLoader.__add__` appends to `_additional_loaders`, and `get_object` picks a sub-loader weighted by `num_models`.

### Multiprocessing / GPU arbitration ([generators/generator.py](simpose/generators/generator.py))

`Generator.start()` runs in the parent, which never imports `bpy`:

- Pending indices are split into shards of `worker_shards`; `n_workers` is clamped to `n_datapoints // worker_shards` (min 1), so small runs silently become single-worker.
- Each shard becomes a `mp.Process` running `Generator.process` (start method is forced to `spawn`). The worker sets `CUDA_VISIBLE_DEVICES` **before** touching TensorFlow, seeds numpy from its PID, builds randomizers from config, then opens the Writer as a context manager and calls the subclass's `generate_data`.
- GPU concurrency is limited by `MainSemaphore` (parent) / `RemoteSemaphore` (worker) — a queue-based semaphore with a **600 s** timeout that starts when the GPU is acquired. The parent's poll loop kills timed-out or crashed workers and **reschedules their shard**, so `generate_data` must tolerate being restarted for a given index range. A shard is retried at most `MAX_SHARD_ATTEMPTS = 3` times before the whole run aborts with the worker's last failure. Workers announce success via a PID queue and are hard-killed if they linger.
- Progress is a count queue drained by a `tqdm` bar in the parent.

Because of `spawn`, anything crossing the process boundary must be picklable, and `import simpose` must happen inside worker processes (it configures `OPENCV_IO_ENABLE_OPENEXR` and the `mp` logger at import time).

### Blender is imported lazily, always

`import bpy` / `import pybullet` appear **inside functions**, never at module top level (except under `TYPE_CHECKING`). Keep it that way — the parent process and the `tf.data` loaders must be usable without paying for Blender.

### Scene, entities, and the observer system

`sp.Scene` is an `Observable` wrapping a `bpy.types.Scene`; it owns the compositor node graph, the render-device selection, and a PyBullet `DIRECT` connection (gravity −z, fixed 1/240 s timestep, collision plane at z=0). Entities (`Object`, `Camera`, `Plane`, `Light`) are thin wrappers over `bpy` objects with state stashed in custom properties (`pb_id`, `coll_id`, `mass`, `semantics`, `sp_baseline`, …) rather than Python attributes — this is what makes them reconstructable from the Blender scene.

Randomizers are `Observer`s that `listen_to(scene)` and fire on an `Event` (`ON_SCENE_CREATED`, `ON_OBJECT_CREATED`, `BEFORE/AFTER_PHYSICS_STEP`, `BEFORE/AFTER_RENDER`, or `NONE` for manually driven ones like `ModelLoader`). The `trigger` field in each randomizer's config is that event.

`Scene.step_physics(dt)` steps PyBullet then copies poses back into Blender (`apply_pybullet_pose`). Physics only exists for objects created with a `mass`; concave collision meshes are V-HACD-decomposed and **cached next to the source mesh as `<stem>_vhacd.obj`** (the `ModelLoader` glob explicitly skips `_vhacd.obj` and `_collision.obj`).

A `6IMPOSE_META` YAML file next to a mesh overrides `mass`, `friction`, `scale`, `name`, `restitution` passed to `create_object`.

### How masks and labels are produced ([scene.py](simpose/scene.py), [egl_rasterizer.py](simpose/egl_rasterizer.py))

**Blender only renders RGB + depth.** Both label channels are rasterized separately by a headless EGL rasterizer, so the render engine is never switched.

Per frame, inside **one** GPU semaphore hold:

1. Cycles render of the right camera (if stereo), files renamed with an `_R` suffix.
2. Cycles render of the left camera → RGB (composited over the background image via an AlphaOver node) + depth EXR. Depth is a *pass of the same render call*, not a separate render.
3. `EGLRasterizer.render_instance_mask` → depth-tested id map (the `visib` instance mask).
4. `EGLRasterizer.render_silhouettes` → per-object *unoccluded* silhouette (each semantic object drawn alone, depth test off, read back within its projected bbox), giving `bbox_obj` / `px_count_all`, hence `visib_fract`.

Geometry is uploaded to the rasterizer **once per scene** in world space (`Scene._upload_geometry`); only a view-projection matrix changes per frame, so camera-only randomization costs no CPU transform work. Every visible mesh participates as an occluder — including the floor plane, which labels as background (`pass_index == 0`) but still occludes. **Only objects created with `add_semantics=True` get silhouettes** — distractors appear in `visib` but get zeroed extra annotations.

Why it is done this way: the old approach rendered the masks with EEVEE into a multilayer EXR (one view layer + compositor slot per semantic object, driven by an EEVEE-only `Material Output`). That meant flipping `scene.render.engine` between CYCLES and BLENDER_EEVEE **every frame**, which tore down Cycles' `use_persistent_data` cache (BVH/textures) and forced a full geometry resync per frame, plus a per-scene EEVEE shader-compile stall and ~51 view layers. The EGL passes were validated to reproduce the EEVEE output exactly (instance mask pixel-agreement 1.0000 / per-object IoU 1.0000 including interpenetrating geometry; silhouettes IoU 1.0000, 0 px bbox error).

Two non-obvious correctness requirements, both of which fail *silently* (wrong labels, not a crash):

- `bpy.context.view_layer.update()` must be called before reading `matrix_world` — setting `.location`/`.rotation_quaternion`/`.scale` does not update it, and a stale matrix yields silently wrong geometry.
- The moderngl context must be re-made-current (`with self.ctx:`) before every use, because Blender's own GL usage clobbers it; otherwise readbacks return the *previous* draw's buffer.

EGL device selection is **not** constrained by `CUDA_VISIBLE_DEVICES`, so the worker exports `SIMPOSE_EGL_DEVICE` to keep a worker's label passes on the same physical GPU as its Cycles render.

`Scene.render()` / `Scene.render_batch()` assemble a `RenderProduct` (a plain dataclass of optional numpy arrays + `ObjectAnnotation` list) and delete the intermediate rgb/depth files. `render_batch(indices, semaphore, on_frame_rendered=...)` renders many frames of one static scene and invokes the callback per frame, so writes and progress ticks happen as each frame completes. `RenderProduct` is the boundary between Blender and the writers — a `Writer.write_data` accepts *either* a `scene` (renders it) *or* a pre-built `render_product`, never both.

`add_semantics` is a parameter threaded from the generator through `ModelLoader.get_object` to `create_object`; `Dropjects` passes `True`. Every semantic object costs one extra EGL silhouette pass per frame (~0.15 ms at 2560×1440 with a bbox-bounded readback), so scenes with many labelled objects get proportionally more expensive — but no extra view layers.

Conventions in the labels: metres, quaternions as `[x,y,z,w]`, intrinsics in OpenCV format, bboxes as absolute pixels `[x1,y1,x2,y2]` (all-zero if invisible), class name derived from the mesh filename.

### Writers and on-disk layout

`Writer` (ABC) handles output dir, index range, GPU semaphore lifetime, cleanup-on-exception, and progress reporting; subclasses implement `get_pending_indices`, `_write_data`, `_cleanup`. `TFRecordWriter` writes one file per shard, `data/{start:06}_{end:06}.tfrecord`, ZLIB, each field `tf.io.serialize_tensor`'d into a bytes feature keyed by the `sp.data.Dataset.*` constants.

```
<output_dir>/
  config.yaml            # dumped generator config
  metadata.json          # {"version": 3.0} — drives reader dispatch
  shard_index.json       # cached per-shard record counts (resume bookkeeping)
  data/*.tfrecord
  data/*.count           # per-shard sidecar written by the worker that filled it
  meshes/*.ply           # exported labelled objects
```

With `overwrite: false`, `get_pending_indices` **resumes**: a shard counts as complete when its record count matches the range encoded in its filename; short or corrupt ones are deleted and regenerated. `overwrite: true` skips the scan entirely.

Getting that count used to mean iterating every shard with `tf.data.TFRecordDataset` in the parent — a full ZLIB inflate of the whole dataset before the first worker spawns (~31 min / 313 GB on `dropjects_lvis`, which looks exactly like a hung run) and it dragged CUDA into the parent, where TF then reserved ~22 GB on GPU 0 and starved the worker assigned to it. [writers/shard_index.py](simpose/writers/shard_index.py) replaces it with three layers, cheapest first:

1. **`shard_index.json`** — counts keyed by shard name; a resume is a few-KB read.
2. **parallel raw-zlib count** for anything the index doesn't vouch for: walks the record framing (`uint64 length | crc | payload | crc`) without materializing payloads, threaded because `zlib.decompress` drops the GIL (~7x faster than the serial tf.data scan; 31 min → ~4 min) and never touches the GPU. Reports a tqdm bar over *bytes to inflate*, so a cold scan is visibly progressing.
3. the result is folded back into the index, so a shard is counted the expensive way at most once.

The whole shard is a single zlib stream, so the lengths are interleaved with the payloads — inflating is unavoidable, only *parsing* is skipped. Shard size is **not** a usable completeness signal (complete shards range 397–774 MB here).

Two invariants worth preserving, because breaking either silently yields a dataset that looks finished but has missing datapoints:

- the index is **never trusted on its own** — each entry carries the shard's `size` + `mtime_ns`, and a mismatch voids it. Safe because a complete shard is written once and never reopened.
- the expected count is **never stored**, only derived from the filename, so an edited index range can't be mistaken for a complete shard.

Workers write a `<shard>.count` sidecar in `Writer.__exit__` instead of updating the shared index, since 16 processes read-modify-writing one JSON would silently drop entries; the parent adopts sidecars on its next scan.

### Dataset readers ([data/](simpose/data/))

`sp.data.Dataset` is just the namespace of key constants (`RGB`, `MASK`, `OBJ_POS`, …) shared by writer, reader and viewer. Add a new field in all three places plus the reader's `_key_mapping` dtype table.

`TFRecordDataset` and the BOP classes override `__new__` to **return a `tf.data.Dataset`, not an instance** — `TFRecordDataset(root, ...)` and the legacy `.get(...)` are equivalent. `__new__` reads `metadata.json` and dispatches on its `version`: `<2` → `_TFRecordDatasetV1` (separate `rgb`/`depth`/`mask`/`gt` files zipped together), `<4` → the current reader (one record per example). The only v2/v3 difference is the mask dtype — `tf.int32` in v2, `tf.uint8` in v3 (so v3 rejects mask ids >255 at write time). Requested keys missing from the first record are silently dropped. The default `pattern="**/*.tfrecord"` globbed from `root_dir` is what makes multi-dataset concatenation and BOP `subsets/` work transparently.

`num_parallel_files` dominates RAM/CPU; start at 1 when debugging. `pre_shuffle` shuffles the file list and turns off determinism.

BOP loaders (`LineMod`, `LineModOccluded`, `TLess`, `HomeBrewedDB`, `YCBV`, `HOPE`) download + convert into the layout above on first use, then delegate to `TFRecordDataset`; each exposes a `CLASSES` name→id dict.

## Gotchas

- `.gitignore` ignores `*.yaml` (but not `*.yml`) — the tracked example config is [dropjects.yml](dropjects.yml). Use `.yml` for anything meant to be committed.
- The `Dropjects` scaffolded template emits `CameraPlacementRandomizer` while the checked-in `dropjects.yml` uses `CameraSceneRandomizer`; both exist, don't "fix" one to match the other blindly.
- Several `sp.logger.error(...)` calls in the generator worker path are progress messages, not failures.
- `Scene._setup_rendering_device` hardcodes `selected_devices = [0]`; multi-GPU is handled by per-worker `CUDA_VISIBLE_DEVICES`, not here.
