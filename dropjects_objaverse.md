# dropjects_objaverse — scoping notes

High-level survey for building an Objaverse-LVIS counterpart to `dropjects_omni`.
Every number below was measured, not quoted, unless marked *(claim)*.

The metadata these numbers came from is staged at
`/media/storage/shared/datasets/objaverse_lvis_models/metadata/`
(`annotations_2024_08_16.json.gz`, `annotations_2023_07_28.json.gz`,
`lvis-annotations.json.gz`, `object-paths.json.gz`, and `lvis_tabletop_uids.json` =
the 24,773 filtered uids). Five sample GLBs used for the import test are in `../samples/`.

## 1. The object dataset

### Objaverse-LVIS itself

Measured from `lvis-annotations.json.gz` (HF `allenai/objaverse`):

| | |
|---|---|
| unique uids | **46,207** (commonly quoted as "47K") |
| LVIS categories | **1,156** |
| objects/category | min 1 (`bolo_tie`), median 30 (`rodent`), max 453 (`chair`) |
| categories with <5 objects | 20 |
| format | one `.glb` per uid, path from `object-paths.json.gz` (798,759 entries) |
| GLB size (n=40 sample) | median **1.3 MB**, mean **12.0 MB**, p90 14 MB, max 325 MB |
| est. raw download | ~550 GB for all of LVIS, ~300 GB for the tabletop subset |
| license | per-asset in `license_info` (Sketchfab CC variants — `by`, `by-sa`, `cc0`, …) |

For comparison, the current `omni3d_models` source is **5,902 models / 216 categories / 994 GB**
(full-res `Scan.obj` plus `simplified.obj`). So Objaverse-LVIS is ~8x the object count and
~5x the category count at a *smaller* footprint.

### Metric scale — this is the crux

**Objaverse GLBs have no metric scale.** Measured world-space bounding boxes after
`bpy.ops.import_scene.gltf` on 5 random LVIS assets: max dimension = 287.5, 120.0,
0.23, 26.3, 36.8 (Blender units). Over three orders of magnitude. The required
per-asset scale factor ranged from 0.0014 to 1.30.

This is the one hard difference from OmniObject3D, where every scan is in millimetres
and `ModelLoader` just applies a fixed `to_scale = 0.001`. A single `scale_range`
cannot work for Objaverse — scale must be per asset.

**Solution: the objathor annotations** (AI2, the asset pipeline behind Holodeck /
ProcTHOR / SPOC). Downloaded and verified:

```
https://pub-2619544d52bd4f35927b08d301d2aba0.r2.dev/annotations.json.gz   # 357 MB, version 2024_08_16
# or: python -m objathor.dataset.download_annotations --version 2024_08_16
```

| | |
|---|---|
| assets annotated | **625,252** |
| **covers Objaverse-LVIS** | **40,227 / 46,207 = 87.1%** |
| annotator | `gpt-4o-2024-08-06` (vision + text), per `annotation_info` |

Relevant fields, all present for 100% of the LVIS overlap:

| field | type | meaning (verified) |
|---|---|---|
| `scale` | float, **metres** | target extent along the up axis; exactly `height`/100 for 40,215/40,227 |
| `height` | int, cm | same quantity in cm |
| `max_dimension` | int, cm | longest side; median `max_dimension/height` = 1.33, p90 = 4.0 |
| `z_axis_scale` | bool | **`True` for all 40,227** → `scale` refers to the z axis |
| `mass` | float, **kg** | tabletop subset: p10 0.10, median 0.50, p90 2.5 |
| `materials` + `composition` | list + weights | `plastic` 13750, `metal` 8248, `paint` 3623, `glass` 2401, `wood` 2258, … |
| `pose_z_rot_angle` | float, rad | canonical yaw correction |
| `quality` | int | LVIS subset: 8→26855, 9→8724, ≤4→181. Usable as a cheap filter |
| `receptacle`, `onFloor`, `onObject`, `onWall`, `onCeiling` | bool | placement priors |
| `category`, `synset` | str | 4,136 free-form categories; top: toy 3202, sculpture 1835, bottle 483 |
| `license_info` | dict | license + creator |

An older `2023_07_28` release exists (8.4 MB, 39,664 entries, with a
`scale_annotated_by` field distinguishing `human_instance_level` / `human_category_level`
/ `gpt-4`) but it **only covers 5,896 LVIS uids = 12.8%**. Use `2024_08_16`.

There is also a category-level human-annotated scale sheet from AI2 *(claim, not fetched)*:
`https://docs.google.com/spreadsheets/d/1dRUKadvw5DyL44yh5nTo0Yc6cllFMSORihkEHq4QtYw` —
coarser (per LVIS category, largest bbox side) but a useful cross-check for the
~13% of LVIS with no objathor entry.

### Usable subset for a Dropjects-style tabletop scene

Filtering the 40,227 LVIS assets that have metric scale to `3 cm <= scale <= 40 cm`:

| | |
|---|---|
| objects | **24,773** |
| LVIS categories covered | **1,105 / 1,156** |
| per-category count | min 1, median 14, max 325 (`seashell`) |
| scale | p10 5 cm, median 15 cm, p90 30 cm |
| mass | p10 0.10 kg, median 0.50 kg, p90 2.5 kg |
| quality | 21,615 at ≥8; only 88 below 5 |

So **~25k objects across ~1.1k categories**, vs 5,902 objects / 216 categories today.
That is the variety win. (Widening to 2–50 cm gives 28,468; the extra 3.7k are mostly
things that will not behave well in a 40 cm drop volume.)

### Texture / appearance variety

Mixed, and lower-fidelity than OmniObject3D scans. From the 5-asset import test:
1–6 mesh objects and 1–6 materials each, 0–3 image textures. Every material had a
`Principled BSDF` fed either by an `Image Texture` (3 of 5 assets) or by nothing at all
(flat base colour — `auto_66`-style materials). OmniObject3D by contrast is uniformly
one photogrammetry-baked JPG per model. Expect Objaverse to trade photometric realism
for geometric and semantic variety — which is the point of the exercise, but it means
`AppearanceRandomizer` will be a no-op on the untextured materials (the HSV node is only
wired when Base Color already has an input).

## 2. The dropjects_omni rendering setup

Master config: `/media/storage/shared/datasets/dropjects/dropjects_omni/dropjects.yml`.
Verified against an actual record in `data/000000_000049.tfrecord`.

| | |
|---|---|
| generator | `Dropjects`, 50 objects/scene, 50 images/scene |
| resolution | 2560 x 1440, stereo, baseline 63 mm, hfov 70° → fx = fy = 1828.03, principal point exactly centred |
| dataset size | **500,000 images / 10,000 shards / 5.8 TB** (~11.6 MB per datapoint) |
| depth range | ~0–1.45 m in the sampled frame |
| randomizers | Appearance, Light (1–4 lights, 300–800 W, dist 3–10 m), Background, CameraScene |
| camera | pitch 20–90°, `fill_range` (0.1, 0.35), roll jitter 90°, pitch/yaw jitter 5°, `min_cam2obj_distance` 0.1 m |
| physics | friction 0.9, restitution 0.1, `drop_spread` 0.4 m, mass hardcoded to 0.2 kg in the generator |
| object source | `ModelLoader` `source: OMNI3D` → `models/<name>/Scan/simplified.obj`, fixed ×0.001 |

Things worth knowing before copying it:

- **`OBJ_CLASSES` are instance names, not categories** — e.g. `handbag_010`, `shampoo_020`.
  A sampled frame had 50 objects spanning 30 unique classes (the generator picks
  `n_classes` distinct models then duplicates them).
- **In the shipped omni dataset, `obj_bbox_obj`, `obj_px_count_all`, `obj_px_count_valid`
  and `obj_visib_fract` are all zero** throughout — it was generated when `Dropjects`
  hardcoded `add_semantics=False`, so no per-object EEVEE view layer existed. Only `mask`,
  `obj_bbox_visib`, `obj_px_count_visib`, pose and camera are real in it. The current code
  passes `add_semantics=True` (audit A1), so a new run **will** produce amodal boxes and
  occlusion fractions. dropjects_objaverse gets labels omni does not have.
- The per-scene class picker rejects any model whose diameter differs from the first pick
  by >70%, so scenes are size-homogeneous. With Objaverse's wider size spread this
  rejection loop will do much more work than it did for omni.
- `MainSemaphore` kills a worker that holds the GPU too long — the timeout is **600 s**
  since audit C2 (it was 100 s when omni ran). A shard that dies is retried at most
  **3 times** before the whole run aborts (C1).
- Objects drop from `highest_point + 2 * max_obj_diameter`; there is no `drop_height`
  knob any more (removed in A11).

### Deltas vs the omni run that affect comparability

The code moved since omni was generated. The biggest change is structural:

- **Labels no longer come from Blender.** omni rendered masks as a third pass (EEVEE, one
  view layer + compositor file slot per semantic object, read back from a multilayer EXR).
  That is gone: `Scene.render_batch` now uploads scene geometry once to a headless EGL
  rasterizer ([egl_rasterizer.py](simpose/egl_rasterizer.py)) and rasterizes the instance
  mask and the per-object unoccluded silhouettes directly, so only RGB + depth come from
  Cycles and the render engine is never switched. Consequence for this dataset: the "50
  semantic objects means 50 extra EEVEE view layers" cost concern is **obsolete**, and
  per-frame label cost is now a view-projection matrix update rather than a render.

Two more that change pixels:

- **`view_transform` is now `"Standard"`** (A7); omni was rendered with Blender's default
  Filmic tonemap. RGB will not be photometrically comparable to omni without accounting
  for this.
- **Depth EXR is now 32-bit** (A4); omni's depth was written at 16-bit half precision.
- **`fill_range` means something different.** The fill→distance formula was ~2x off and was
  corrected (A5). omni's `(0.1, 0.35)` produced camera distances of **0.87–4.01 x** the
  largest object diameter; under the corrected formula the same numbers give
  **2.30–8.17 x** — noticeably wider framing. A5 explicitly decided *not* to retune the
  yml, so `dropjects_objaverse.yml` keeps `(0.1, 0.35)`. To reproduce omni's actual
  framing instead, use **`(0.203, 0.854)`**. This is a real fork in the road: pick before
  committing to a long run.
- Light ranges were retuned (A8 inclusive-randint fix, A10 tightening), so the config uses
  `no_of_lights_range: [1, 3]` and `distance_range: [3, 6]` where omni had `[1, 4]` and
  `[3, 10]`.

## 3. The pipeline (built and validated)

Two scripts under `/media/storage/shared/datasets/objaverse_lvis_models/`. Both are
resumable and detached-safe; **no simpose code change was needed.**

### `download_lvis_glbs.py`

Downloads the 24,773 curated uids into `glbs/<hf-subdir>/<uid>.glb`. Resume is by
construction: each file lands as `<name>.part` and is only renamed after the full body is
received and fsynced, so any file with its final name is complete and a restart just skips
it — no HEAD requests, no size manifest. 12 workers, 5 attempts with exponential backoff,
aborts if free space drops below 200 GB.

**Done: 24,733 files / 184.0 GB / 2.75 h at ~20–34 MB/s, zero permanent failures.**
(My 300 GB estimate from a 40-file sample was high; largest single asset was 346 MB.)

### `convert_glbs.py`

Per asset: import GLB → bake the transform hierarchy → join meshes → scale to the objathor
metric size → centre bbox → strip non-surface geometry → decimate to 10k verts → export OBJ
+ textures + `6IMPOSE_META`. Writes `objects.json` mapping
`"<lvis_category>_<NNNN>" -> "models/<uid>/model.obj"`, which is exactly what
`ModelSource.FILE` reads. 32 workers, ~2.9 assets/s, ~96% yield (the rest are genuinely
degenerate: <100 verts, no faces, or decimation that won't converge).

**Two bugs found by checking the output distribution rather than the samples** — both would
have quietly poisoned the dataset, so keep an eye on these if the converter is changed:

1. **Scaling blowup on near-planar assets.** Dividing by `bbox_z` explodes when z is
   near-zero: one asset with `extent_raw = [100, 100, 3e-5]` became **165 km** across, which
   would have wrecked every scene it appeared in (and `CameraSceneRandomizer` sizes its
   distance off `max_object_diameter`). Fixed by taking
   `factor = min(scale/bbox_z, (max_dimension/100)/bbox_max)` — the annotated
   `max_dimension` is the cross-check — plus a hard reject above 1 m.
2. **Decimation silently stalling at ~50%.** Some assets carry millions of vertices joined
   by *edges only*. `ratio=0.01` collapsed faces exactly (2,003,567 → 20,035) but left
   2.55M orphaned vertices, so a vertex-count-driven loop never converged and assets came
   out at up to **3.9M verts** against a 10k budget. `delete_loose` doesn't catch them (they
   have edges). Fixed by `delete_loose(use_verts=True, use_edges=True, use_faces=False)`
   before and after each pass, 6 passes, ratio floor 0.002, and a reject above 40k.

Verified on the offenders: 165,046 m → 1.00 m, 8.3 m → 0.15 m, 3.9M verts → 9,922 verts.

Why offline conversion rather than teaching simpose `.glb`: it quarantines every Objaverse
quirk in one place, and it does the expensive work (join, decimate) once instead of per
scene setup. For the record, loading `.glb` directly would have needed three fixes —
`create_object`'s suffix dispatch doesn't cover `.glb`, `ModelSource.GENERIC_GLTF` globs
`**/*.gltf` which never matches, and `from_gltf` takes `selected_objects[0]` which for all
5 test assets was `Sketchfab_model`, an **EMPTY**, not a mesh.

### Axis convention — resolved, no bug

Worth writing down because it looks broken and is not. `wm.obj_import` applies its Y-up
conversion as an **object-level `rotation_euler` of +90° about X**, not baked into mesh
data — and `_initialize_bl_object` immediately calls `set_rotation(identity)`, discarding
it. So Blender ends up using the raw *file* coordinates, which is also exactly what
PyBullet reads for the collision mesh. Measured on `biscuit_067`:

```
LOCAL bbox (mesh data)               [52.7 13.6 50.7]
WORLD after import (rot_euler 90° X) [52.7 50.7 13.6]
WORLD after set_rotation(identity)   [52.7 13.6 50.7]  == PyBullet AABB ✓
```

The converter therefore exports with `up_axis='Z', forward_axis='Y'` (identity
passthrough), so the file holds upright Z-up coords and both consumers agree.

### End-to-end validation

Converted models loaded back through the real `sp.entities.Object.from_obj`:

- **Geometry/scale round-trips exactly** — Blender world extent == the converter's target
  metric extent to <0.3 mm.
- **PyBullet agrees** — collision AABB matches within a constant +4 mm, which is Bullet's
  default 2 mm-per-side collision margin. Visual and collision meshes are aligned.
- **Metric scale is baked into the mesh**, so `6IMPOSE_META` carries no `scale`, the loader
  stays at 1.0, and PyBullet's `meshScale` stays 1.0 and consistent.

### Appearance: what survives the OBJ round trip

Surveyed 150 materials across 60 random assets for what drives Principled Base Color:

| Base Color driven by | share of materials | survives OBJ? |
|---|---|---|
| unlinked (flat colour) | 56.7% | **yes** — exact, e.g. `flat(0.077, 0.021, 0.004)` |
| direct `TEX_IMAGE` | 32.0% | **yes** — comes back through simpose's HSV node |
| `VERTEX_COLOR` | 5.3% | **no** — `from_obj` doesn't wire a Color attribute (only `from_ply` does) |
| no Principled BSDF | 5.3% | no — exports as default grey |
| via `MIX` (baseColorFactor x texture) | 0.7% | **yes, after the fix below** |

The headline is that Objaverse-LVIS is *substantially untextured*: 57% of materials are flat
colour, and 48% of assets have no textured base colour at all. That is a real fidelity step
down from OmniObject3D's uniform photogrammetry-baked JPGs, and it is a property of the
source data, not of the conversion.

OBJ/MTL only emits `map_Kd` when Base Color links **directly** to an image, so the glTF
`MIX` wrapper silently exported as flat grey. The converter now walks upstream from Base
Color to find the image (or a constant colour) and relinks it directly. Also note GLB
textures are *packed*, so `path_mode="COPY"` had nothing on disk to copy — they are written
out explicitly first.

Two known, accepted losses: PBR roughness/metallic/normal maps (MTL can't express them, and
`AppearanceRandomizer` overwrites roughness/metallic with uniform values anyway, exactly as
for omni3d), and vertex colours on ~5% of materials.

#### The survey above under-counted, and the first full conversion shipped broken (fixed)

**The "no Principled BSDF, 5.3%" row was the whole iceberg.** That survey only asked what
drives *Principled* Base Color, so it recorded materials with no Principled node as a single
small bucket and never looked at what they actually contained. Measured across all 23,850
converted assets rather than a 60-asset sample:

| | assets | share |
|---|---|---|
| ≥1 material that lost its colour | 5,921 | **24.8%** |
| no textured material at all (and ≥1 lost) | 4,801 | **20.1%** |
| texture PNG on disk that the mtl never references | 2,335 | 9.8% |
| every material exported as default grey | 983 | 4.1% |

At 50 objects per scene that is ~12 objects per frame rendering flat white or black. It was
caught by eye — a red flower atlas rendering as a white blob — not by the pipeline.

Three glTF material shapes reach the exporter without a readable Base Color. Only the third
was handled:

1. **`KHR_materials_unlit`** — the dominant case, 3,509 materials. Blender's glTF importer
   builds `TEX_IMAGE → EMISSION → MIX_SHADER (+ BSDF_TRANSPARENT, LIGHT_PATH) → OUTPUT` with
   **no Principled node at all**, so the flatten pass hit `if not pbs: continue`. The PNG is
   still written next to the obj (the converter dumps all of `bpy.data.images`), which is why
   the symptom is an *unreferenced* texture rather than a missing file.
2. **Colour on Emission Color, Base Color unlinked** (usually black) — `bc.links` is empty so
   the pass `continue`d at its first guard. Exports `Kd 0 0 0` → renders black.
3. **Base Color behind a `MIX`** — the case the original fix addressed.

`convert_glbs.py` now synthesizes a Principled BSDF and wires it to Material Output for case
1 (taking the colour from the `EMISSION` node specifically — a BFS from `Surface` walks into
the transparent half of the unlit mix and picks up its default white), and falls back to the
Emission Color texture for case 2, guarded on `Emission Strength > 0` so that `Lights_off`-style
materials aren't turned white. Validated on 30 re-downloaded GLBs: 13 gained textures, 0 lost,
0 broken refs, 0 conversion failures, and **exported geometry is byte-identical** — which is
what lets the V-HACD cache survive a re-conversion (~60 CPU-hours).

**[audit_textures.py](/media/storage/shared/datasets/objaverse_lvis_models/audit_textures.py)**
now checks the converted set for exactly this and exits non-zero if it recurs. Run it after
every conversion: per-asset the failure is invisible, and it only shows up in the aggregate.

Still not fixed: **vertex colours** (~980 assets export as flat `Kd 0.8` grey). `wm.obj_export`
has `export_colors`, but `wm.obj_import` has no matching option and `Object.from_obj` doesn't
wire a Color Attribute node the way `from_ply` does — so it needs a simpose-side change too,
and turning `export_colors` on unilaterally risks PyBullet's obj parser choking on the
extended `v x y z r g b` lines. Deliberately left as a separate decision.

## 4. Generation smoke test

10 images / one 50-object scene at full 2560x1440 stereo, run against 10,073 converted
models. **Passed** — exit 0, no errors, one v3 shard, `meshes/` exported with clean LVIS
names (`water_gun_0014`, `piggy_bank_0017`, `toothbrush_0001`).

Labels from the EGL rasterizer are sound on Objaverse geometry. All 19 keys present,
including the four that are all-zero in the shipped omni dataset:

- 39–50 of 50 objects visible per frame, mask ids 0..50, `uint8`
- `visib_fract` never exceeds 1.0
- `px_count_all >= px_count_visib` for **every object in every frame**
- depth 0.52–1.42 m, 100% valid

**Two findings that change the plan.**

### RGB was silently misaligned with the labels (fixed)

The composited RGB came out at a *different size per frame*, while depth and mask were
always exact:

```
frame 0: rgb=(1441, 2560)   depth=(1440, 2560)  mask=(1440, 2560)
frame 1: rgb=(1440, 2562)   depth=(1440, 2560)  mask=(1440, 2560)
frame 2: rgb=(1441, 2560)   depth=(1440, 2560)  mask=(1440, 2560)
```

Cause: the background `Scale` node in `"Render Size"` mode scales an arbitrary-aspect
background to *cover* the render, and the composited canvas ends up 1–2 px larger than
`resolution_x/y` by an amount that depends on the individual image's scale factor. All
16,873 backgrounds are odd sizes (256x256, 640x480, 1504x1000 — none 16:9), so this fires
constantly. It is a bpy 5.0 compositor-port regression: omni's records are exactly
1440x2560.

This is worse than it looks — RGB was off from the mask, the depth *and* the intrinsics
(`cy = 720` assumes 1440 rows) by a per-frame amount, which would quietly degrade any
model trained on it. Fixed defensively in `Scene._read_rgb`: crop to
`resolution_y/resolution_x` on read, and hard-error if the image is ever *smaller*. That
holds regardless of what the compositor does. Worth also fixing at the node level.

### The 10x slowdown was a cold V-HACD cache, not the models

First LVIS scenes ran ~10x slower than omni3d. It is **not** rendering, and the models
are **not** heavier — measured per-frame, LVIS is slightly *faster*:

| mean of 5 frames | omni3d | LVIS |
|---|---|---|
| `cycles` | 4.47 s | **3.95 s** |
| `labels(egl)` | 0.33 s | 0.17 s |
| `readback` | 0.66 s | 0.68 s |

And per model, LVIS geometry is smaller (n=25 each):

| | LVIS | omni3d |
|---|---|---|
| vertices mean | 6,860 | 8,490 |
| **faces mean** | **7,137** | **17,120** |
| texture megapixels mean | 2.0 | 2.5 |
| materials mean (max) | 2.0 (6) | 1.0 (1) |
| texture MB on disk | 2.42 | 0.69 |

Only texture *bytes* are inflated, from writing lossless PNG where omni3d has JPG — that
is disk, not render cost, since Cycles decodes to the same pixels.

The real cause: `Object._initialize_bl_object` runs `pybullet.vhacd` on **first use** of a
mesh and caches `<stem>_vhacd.obj` beside it. **omni3d is 5,902/5,902 cached** — warm for
ages. LVIS was cold, and V-HACD costs a measured **mean 10.2 s per model** (range
0.2–27 s). Dropjects loads 50 models per scene, so a cold cache is **~8.5 min of scene
setup**, paid before a single frame renders.

Fixed by `pregen_vhacd.py`, which builds the whole cache offline in parallel (resumable,
atomic temp+rename matching the runtime's own convention, so a concurrent generation run
can never see a truncated file). ~21k models x 10.2 s is ~60 CPU-hours, about an hour of
wall clock at 64 workers, paid once. **Run it before any timing measurement or long run** —
otherwise the first pass through the model set silently pays it instead.

### Storage: depth reverted to float16 (dataset format v4)

The first smoke run measured **28.6 MB per datapoint** vs omni's 11.6 MB, which would have
made 500k images ~14.3 TB. Cause: audit A4 had widened depth to 32-bit, and float32 depth
both doubles the bytes and compresses much worse.

Reverted to half float on request — **this supersedes A4.** float16 gives ~1 mm
quantization over this pipeline's 0.5–2.5 m working range, comparable to a real depth
sensor. Changed in three places, which must stay in step:

- `scene.py`: depth EXR slot `color_depth = "16"`, and `_read_depth` returns float16
  (clamping still done in float32 to dodge half's 65504 overflow)
- `tfrecord_writer.py`: `depth*` stored as `np.float16`, `metadata.json` version → **4.0**
- `tfrecord_dataset.py`: dispatch widened to `version < 5` with a `depth_dtype` argument —
  v2/v3 keep reading float32, v4 reads float16, so existing datasets (omni is v2) are
  unaffected

`tests/test_writer_reader_roundtrip.py` now asserts `tf.float16` on the depth channels, so
a silent widening back to float32 fails the suite rather than quietly doubling dataset size.

## Resolved

- **objathor's z is Blender's z** after `import_scene.gltf` — tested across 40 assets by
  comparing the annotated `max_dimension/height` aspect against measured bbox aspect ratios:
  median |log ratio| error 0.163 for z vs 0.405 (y) and 0.488 (x). Scaling by bbox z is
  correct. The residual spread is because `height`/`max_dimension` are GPT-4o estimates of
  the *real object*, not measurements of the mesh, so the prior is coarse — some assets end
  up with a max dimension well above the annotated one (e.g. a "sign" at 0.3 m tall but
  0.54 m wide). See the clamp question below.
- **OBJ axis convention and visual/collision alignment** — see section 3. No bug, no code
  change; export identity-passthrough Z-up.
- **`BLENDER_EEVEE` is correct.** The env is bpy **5.0.1**, and 5.0 reverted the engine id
  back to `BLENDER_EEVEE` (only 4.2–4.x used `BLENDER_EEVEE_NEXT`). Non-issue.
- **`add_semantics`** is already `True` in the current generator (audit A1), so the new
  dataset gets amodal boxes and `visib_fract`.

## Open questions for the deep dives

1. **`fill_range`: omni framing or corrected semantics?** See the deltas in section 2 —
   the single decision with the biggest visible effect on the dataset.
2. **Per-frame wall clock, and does the EGL rasterizer like Objaverse geometry?** The old
   "50 extra EEVEE view layers" worry is gone with the rasterizer rewrite, but nothing has
   been measured on this asset set, and Objaverse meshes are messier than OmniObject3D
   scans (non-manifold, multi-material, decimated, with non-surface geometry stripped).
   Needs: a timing run against the 600 s semaphore budget at 2560x1440 stereo, and a check
   that `visib_fract`/`bbox_obj` come out sane. Note the v3 mask is `uint8`, so a scene may
   not exceed 255 object ids — 50 objects plus copies stayed under 100 in omni, so there is
   headroom, but it is now a hard limit rather than a soft one.
3. **Instance classes or LVIS categories?** `objects.json` currently keys on
   `<lvis_category>_<NNNN>`, so `OBJ_CLASSES` will read `mug_0007` and the category is
   recoverable by stripping `_NNNN` — parallel to omni's `handbag_010`. Fine for
   instance-level work; decide now if category-level labels are wanted.
4. **Should max dimension be clamped?** Scaling by height alone lets some assets exceed
   their annotated `max_dimension` (median ratio 1.33, p90 4.0). An optional second pass
   could clamp so that no object's largest side exceeds, say, 0.5 m.
5. **Quality/filter policy.** `quality`, `receptacle`, `onFloor` and the material list are
   available; only the 3–40 cm size filter is applied so far. Worth deciding what a
   "droppable object" is — e.g. excluding `onWall`/`onCeiling` items.
6. **Per-asset friction/restitution from `materials`.** objathor gives a material list plus
   `composition` weights; the converter writes one global friction 0.9 / restitution 0.1,
   as omni did. `mass` *is* per asset (clamped to 0.02–5 kg; objathor runs to 5000 kg).
7. **GLBs get deleted once conversion is verified** (user's call). Measured ~2.4 MB per
   converted model, so the OBJ+PNG tree lands around **57 GB** against **172 GB** of source
   GLBs. Sequencing matters: verify the converted output distribution *first*, because a
   re-conversion has already been necessary once (see the two bugs above) and re-downloading
   is 2.75 h. `download_lvis_glbs.py` can always refetch from
   `metadata/lvis_tabletop_uids.json` + `object-paths.json.gz`.
8. **Vertex-coloured assets** (~5% of materials) currently render flat. Could be handled by
   baking vertex colours to a texture in the converter, or by wiring a Color attribute node
   the way `from_ply` does.
