# Stage contracts

The pipeline's stages exchange data through files in a run directory. This page specifies those files; `src/v2hoi/contracts.py` implements their formats. The current exporter writes an internal artifact, not a verified official submission, and validates only the refined human, motion, and object mesh. How the team splits the stages is in [workflow.md](workflow.md).

**Data boundary:** Develop, tune, self-check, and validate with Track 1-derived artifacts or self-created synthetic tests. Submission assets must come from Track 1 inputs and the reconstruction built from them; synthetic fixtures are tests, not submission assets. Do not use Track 2 data, assets, trajectories, reference meshes, or labels for any of these purposes. Track 1 has no public ground truth, so an official ground-truth-based local score is unavailable.

## Run directory

```
runs/<run_id>/
  run.json                          dataset, upstream run, history of invocations
  inputs/<episode>/camera.json      Camera          inputs    1 platform & perception
  inputs/<episode>/masks.npz        Masks           inputs    1 platform & perception
  inputs/<episode>/depth.npz        Depth           inputs    1 platform & perception
  human/<episode>/human.npz         Human           human     2 human
  human/<episode>/depth_scale.json  DepthScale      human     2 human
  objects/<object>/object.json      ObjectAsset     objects   3 object
  objects/<object>/mesh.glb         the mesh        objects   3 object
  motion/<episode>/motion.npz       Motion          motion    3 object
  refine/<episode>/human.npz        RefinedHuman    refine    4 temporal & physics
  refine/<episode>/motion.npz       RefinedMotion   refine    4 temporal & physics
  export/                           internal schema export    1 platform & perception
```

Stages run in this order: inputs, human, objects, motion, refine, export.

`<episode>` is the six-digit episode index, as in the datasets. `runs/` is not committed.

## Conventions

- **Frame.** Everything is in the clip's camera frame: OpenCV axes (x right, y down, z forward), metres. The camera is static within a clip, so the camera frame is also the submission's world frame. The official evaluation fits one Sim(3) on the first frame, which makes every rigid choice of world equivalent.
- **Human convention.** SOMA-X parameters are stored in the SOMA convention (y up). `body.SomaBody` output multiplied by `diag(1, -1, -1)` is in the camera frame. This conversion is part of the internal schema and does not require Track 2 labels.
- **Time.** One entry per video frame. Every per-frame array covers every frame of the clip and holds no NaN or inf. Occluded frames are filled, never left out.
- **Units.** Metres, radians, pixels.
- **Version.** Every file records `CONTRACT_VERSION`. Loading a file with another version fails.

## Artifacts

### Camera: `inputs/<episode>/camera.json`

| Field | Meaning |
|---|---|
| `width`, `height` | video size in pixels |
| `fx`, `fy`, `cx`, `cy` | pinhole intrinsics in pixels; the principal point lies inside the image |
| `camera` | physical camera name, when the Track 1 metadata gives it |

A physical camera gets one set of intrinsics, merged over its clips (`stages.inputs.merge_intrinsics`).

### Masks: `inputs/<episode>/masks.npz`

`human` and `obj`: `(n_frames, height, ceil(width / 8))` uint8, one bit per pixel packed along the width. Build with `Masks.pack(human_bool, obj_bool)` and read one frame with `masks.frame("obj", t)`.

### Depth: `inputs/<episode>/depth.npz`

`depth`: `(n_frames, ceil(height / stride), ceil(width / stride))` float16, in the depth model's own metres, before any scaling to the human. Video pixel `(u, v)` maps to `depth[:, v // stride, u // stride]`; 0 means unknown. `stride` is stored with it; the producer picks it to keep files manageable (a full-resolution clip is about 3 GB).

### Human: `human/<episode>/human.npz`

| Field | Per-frame shape | Meaning |
|---|---|---|
| `pose` | 77 × 3 | SOMA-X local joint rotations (rotation vectors) |
| `transl` | 3 | SOMA-X root translation, SOMA convention |
| `identity` | 45 | MHR identity coefficients |
| `scale` | 68 | SOMA-X scale parameters |
| `bone_flex` | 6 | SOMA-X bone length flexibles |
| `mhr_global_rot` | 3 | MHR global rotation, Euler ZYX |
| `mhr_body_pose` | 133 | MHR body pose parameters |
| `mhr_hand_pose` | 108 | MHR hand pose parameters |
| `mhr_scale` | 28 | MHR scale parameters |
| `mhr_shape` | 45 | MHR shape parameters |
| `mhr_transl` | 3 | MHR translation, camera frame |

The SOMA-X fields feed the current internal parquet serializer and can support self-created synthetic checks. The MHR fields are intended for the official submission; their shapes follow SAM 3D Body's output (toolkit `v2d_sam3d_body`) until the official format is published. This schema compatibility does not authorize reading Track 2 data. A clip has one person, so the identity should be one vector repeated over frames (`stages.human.lock_identity`).

### DepthScale: `human/<episode>/depth_scale.json`

| Field | Meaning |
|---|---|
| `scale` | metric depth = `scale` × `Depth.depth`; one value per clip, positive |
| `method` | how it was estimated |

The human is the only metric anchor (design 3.2): object mesh scales and tracking use the scaled depth, so objects land at the same depth as the hands.

### ObjectAsset: `objects/<object>/object.json` and `mesh.glb`

One per object, shared by all of its clips. `mesh.glb` is in metres, in the object's canonical frame; the largest side of its bounding box is between 2 cm and 3 m.

| Field | Meaning |
|---|---|
| `name` | object name from the dataset |
| `scale` | scale already applied to the mesh, kept for provenance; the object's clips agree on one value (`stages.objects.merge_scale`) |
| `source` | generator and candidate that produced the mesh |
| `symmetry` | e.g. `continuous-z`, or null; free text until a consumer needs more |
| `episodes` | clips that use this mesh |

### Motion: `motion/<episode>/motion.npz`

| Field | Per-frame shape | Meaning |
|---|---|---|
| `T_cam_obj` | 4 × 4 | maps mesh coordinates to the camera frame; proper rotation, last row `[0, 0, 0, 1]` |
| `confidence` | scalar | 0–1, how much the tracker trusts the frame; 0 marks a frame filled without evidence |

Every frame has a pose, occluded ones included. The tracker fills frames it cannot see simply (holding the last pose, `stages.motion.hold_missing`) with confidence 0; the refine stage decides how to fill them properly.

### RefinedHuman and RefinedMotion: `refine/<episode>/human.npz`, `refine/<episode>/motion.npz`

Same fields as Human and Motion, after temporal and contact refinement: smoothing, static segments, low-confidence frames, contact. Export reads these. It detects a raw human or motion artifact from a nearer upstream run than the refined artifact, but it does not yet detect a same-run overwrite or all changed dependencies. Re-run refine whenever its inputs change.

### Internal export schema: `export/`

The current backend writes a parquet layout that `v2hoi.score` can read, plus `mhr/episode_XXXXXX.npz` with MHR fields. The name `tier1` in the backend is a legacy serialization label, not permission to read Track 2 or a claim that this is the official submission format:

```
export/meta/info.json
export/meta/episodes_metadata.jsonl
export/data/chunk-000/episode_XXXXXX.parquet
export/mesh/<object>/<object>.glb
export/mhr/episode_XXXXXX.npz
```

Poses are copied unchanged, since the camera frame is the world frame. The official submission exporter is not implemented. A ground-truth-based score of Track 1 is unavailable locally because Track 1 has no public ground truth.

## Upstream runs

A run can name an upstream run (`--upstream`). Reads look in the run first, then in its upstream, then in that run's upstream. Start a Track 1 run, then run a replacement stage against that snapshot. The commands below use only registered fake backends and exercise orchestration, not reconstruction quality. These placeholders use Track 1 metadata to select clips but do not inspect the video:

```bash
python -m v2hoi.run --run-id track1-smoke-base --dataset track1 --episodes 0 6 9 16 24
python -m v2hoi.run --run-id motion-smoke-1 --dataset track1 --episodes 0 6 9 16 24 \
    --upstream runs/track1-smoke-base --stages motion refine export
```

Once a real backend is registered, select it with `--backend motion=<registered-name>` in a new Track 1 run. Re-running stages in an existing run overwrites their outputs and keeps the rest; dependency invalidation is incomplete, so re-run downstream stages after changing any input. `run.json` records every invocation: stages, backends, episodes, and git revision.

## Legacy Track 2 entry points: not permitted

The runner now rejects `--dataset tier1`, `human=tier2`, `motion=tier2`, and `objects=reference`; the legacy data-reader classes also refuse execution. These initial restrictions await remote runtime verification. Historical outputs from those paths are **not permitted** for development, tuning, self-checks, validation, or submission provenance. Do not use them as upstream runs. Full artifact-source validation remains to be implemented.

Use Track 1-derived artifacts to develop a real stage, and self-created synthetic fixtures to exercise its contracts. The current fake backends provide only a file-format and orchestration smoke test. Real reconstruction backends remain to be implemented.

## Adding a backend

A backend is a class with `run(run, clips)`, listed in its stage module's `BACKENDS`:

```python
class MyTracker:
    def run(self, run: Run, clips: list[Clip]) -> None:
        import heavy_dependency  # inside, so CI and the fake pipeline stay light

        for clip in clips:
            masks = run.load(Masks, episode=clip.episode)
            ...
            motion = Motion(T_cam_obj=poses, confidence=conf)
            motion.validate(clip.n_frames)
            run.save(motion, episode=clip.episode)


BACKENDS = {"fake": FakeMotion, "mytracker": MyTracker}
```

Select it with `--backend motion=mytracker`. A model that needs its own environment (Docker, conda) can run as a subprocess; the backend then converts its output into these files.

## Changing a contract

Changing a field's meaning or shape bumps `CONTRACT_VERSION`. Do it in a small PR of its own, before the code that depends on it, and have it approved by the owners of the stages that read the field. Runs saved under the old version stop loading, which is intended: rerun or convert them.

## Known gaps

- Resuming by input hash (design section 3) is not implemented yet. For now, rerun the stages you changed.
- Depth has no real producer yet. The fake one writes a flat wall at stride 8.
- There is no public Track 1 ground truth for local official scoring. Test metric primitives with independently generated geometry using the test command in the README. The full legacy scorer still uses Tier 2-derived normalization values, so its CLI is outside this workflow even with synthetic reference files.
- Masks are stored whole-clip: about 0.4 MB compressed for an empty 900-frame clip, and about 400 MB in memory once loaded. If real masks turn out too large, they will move to per-frame chunks under a new contract version.
