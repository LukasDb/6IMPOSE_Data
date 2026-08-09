# Audit decisions — 2026-08-03

Consolidated from five audits (hard bugs, silent correctness, 6D-pose config sanity,
performance, robustness). Every item was reviewed; decisions below. Checkboxes track
implementation progress.

## A. Label & data correctness (dataset content)

- [x] **A1 — add_semantics hardcoded False** (`model_loader.py:158`): make `add_semantics` a
  parameter on `ModelLoader.get_object`; Dropjects passes **True for main/target objects and
  their copies, False for distractors**. Restores `visib_fract`, `bbox_obj`, `px_count_all`,
  `px_count_valid`, and `meshes/` export.
- [x] **A2 — intrinsics setter/getter inconsistent** (`camera.py:130,182`): fix both to Blender's
  convention (shift_y in sensor-**width** units under HORIZONTAL fit, correct signs), re-enable
  the round-trip assert at `camera.py:188-190`.
- [x] **A3 — BOP `visib_fract` stale carry-over** (`bop_datasets.py:217-219`): initialize
  `visib_fract = 0.0` per instance.
- [x] **A4 — depth GT quantized to float16** (`scene.py:666`): depth EXR slot `color_depth="32"`.
- [x] **A5 — fill→distance formula ~2× off** (`camera_scene_randomizer.py:60`,
  `camera_placement_randomizer.py`): correct tan-space math `(d/2)/tan(fill·fov/2)`.
  **No yml retune** — new behavior is the intended meaning of `fill_range`.
- [x] **A6 — dead rejection-sampling guard**: reset `loc = None` on every failed check in both
  camera randomizers so the 10k-iteration RuntimeError actually fires.
- [x] **A7 — no color management**: set `view_settings.view_transform = "Standard"`.
- [x] **A8 — LightRandomizer exclusive randint** (`light_randomizer.py:45`): change to
  `randint(low, high+1)` **and reduce `no_of_lights_range` upper bound by 1 in dropjects.yml**
  (code fix, no behavior change for existing config).
- [x] **A9 — AppearanceRandomizer docs say σ, code is uniform**: keep uniform, rewrite config
  descriptions to "uniform half-range"; change the `ReferenceError` `break` → `continue`.
- [x] **A10 — dropjects.yml light energy/distance spread**: tighten ranges (e.g. cap max light
  distance) but **keep a healthy spread** — some exposure diversity is wanted.
- [x] **A11 — dead `drop_height` config**: remove the stale key from DropjectsConfig,
  dropjects.yml and the template (hardcoded `2*max_obj_diameter` rule stays — finetuned).
- [x] **A12 — SimpleGen linear vfov** (`simple_gen.py:127`): tan-space `2*atan(tan(hfov/2)*h/w)`.
  *(superseded: `SimpleGen` has since been deleted.)*
- [x] **A13 — CameraFrustumRandomizer inverse rotation + linear vfov**
  (`camera_frustum_randomizer.py:37,55`): use `cam_rot.apply(...)`, tan-space vfov.

## B. Hard crash bugs

- [x] **B1 — scene-reset ReferenceError → infinite reschedule** (`dropjects.py:200`): reset
  model loaders **before** creating the new `sp.Scene`; additionally make
  `Object.remove`/`ModelLoader.reset` tolerate `ReferenceError` as a safety net.
- [x] **B2 — Join sampling drops a loader** (`model_loader.py:135-143`): sample from
  `[self] + _additional_loaders` weighted by `num_models`, any join arity.
- [x] **B3 — empty annotation list crashes writer and deletes shard**
  (`tfrecord_writer.py:80`): guard `objs == []` by writing empty tensors; `_cleanup` must not
  delete the whole shard on per-frame errors.
- [x] **B4 — `Light.remove` finally defeats guard** (`light.py:41-46`): restructure control flow.
- [x] **B5 — `Object.remove` reads coll_id after removal** (`object.py:459`): read pb_id/coll_id
  before removing the Blender object; fixes PyBullet collision-shape leak.

## C. Orchestration & robustness

- [x] **C1 — no retry cap on rescheduled shards** (`generator.py:178-201`): cap at 3 attempts
  per shard, then **abort the run** surfacing the worker's last failure.
- [x] **C2 — GPU semaphore timeout**: raise hardcoded constant 100s → **600s** (no config knob).
- [x] **C3 — stale `s` semaphore release** (`generator.py:182-191`): look up the semaphore the
  dead worker actually holds (restore intent of commented-out line 179).
- [x] **C4 — success race**: drain `success_accumulator` before classifying a dead worker as
  crashed.
- [x] **C5 — config.yaml clobbered before overwrite refusal** (`generator.py:59-62`): call
  `get_pending_indices()` (validation) **before** `dump_config()`.
- [x] **C6 — implement real resume** (`tfrecord_writer.py:39-43`): when `overwrite: false` and
  dataset exists, scan `data/*.tfrecord`, verify each shard is readable, schedule only
  missing/corrupt index ranges.
- [x] **C7 — Ctrl-C truncated shards** (`writer.py:73`): restore the deferred-KeyboardInterrupt
  mechanism around `write_data`.
- [x] **C8 — parent fd exhaustion**: create semaphore comm queues lazily at worker launch; reap
  comms of dead workers.
- [x] **C9 — V-HACD cache poisoning** (`object.py:563-575`): write to temp path +
  `os.replace()` atomic rename.
- [x] **C10 — fail-fast config validation**: validate randomizer inputs (dirs exist, globs
  non-empty, image extensions — incl. widening the `*.jpg`-only background glob) **in the
  parent at startup**, abort with a clear message.

## D. Reader

- [x] **D1 — parse map**: `num_parallel_calls=tf.data.AUTOTUNE` (V1+V2). **No explicit
  `.prefetch()`** per decision.
- [x] **D2 — `get()` pattern mismatch**: align `_TFRecordDatasetV2.get` default pattern with
  `__new__` (`**/*.tfrecord`).
- [x] **D3 — unknown/missing metadata version**: raise a clear `ValueError` instead of silently
  returning `None` / bare `KeyError`.
- [x] **D4 — empty glob**: raise "no tfrecords found under <path> with pattern <p>" instead of
  `IndexError`.

## E. Performance

- [x] **E1 — depth EXR 3-channel → BW** (`scene.py:663`).
- [x] **E2 — annotation loop**: one `np.nonzero(mask==id)` per object (derive count/bbox/valid);
  open the mask EXR once for the whole loop.
- [x] **E3 — semaphore handshake latency** (`remote_semaphore.py`): drain each comm fully per
  `run()`, shorten parent poll sleeps (~0.05s) and the 0.5s confirm sleep.
- [x] **E4 — ZLIB level**: `compression_level=1` in `TFRecordOptions`.
- [x] **E5 — drop-loop O(n²) syncs** (`dropjects.py:238-251`): `step_physics(dt, apply=False)`
  during settling, heights via `p.getBasePositionAndOrientation`, single `_apply_simulation()`
  after the loop.
- [x] **E6 — mask dtype int32 → uint8** with **metadata version bump to 3** and reader dispatch
  (old v2 datasets keep parsing as int32).

## Won't fix (explicit decisions)

- Denoiser / Cycles samples stay hardcoded (64 + denoise is the chosen operating point).
- BOP partial-conversion detection (delete `subsets/` manually if interrupted).
- Downloader hardening (datasets already downloaded).
- Deterministic reader key probe (accept probe-file nondeterminism).
- Per-object amodal mask storage (revisit if amodal training is adopted).
- Physics settling / restitution / mass defaults (finetuned, only the dead `drop_height` key is removed).
- V1 legacy reader uint8 mask wrap; Scene(bl_scene) reconstruction gaps; mask-EXR float16
  id>2048 trap.
