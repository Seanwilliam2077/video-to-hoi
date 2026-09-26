# Stage contracts

The pipeline's stages exchange data only through files in a run directory. This page specifies those files; `src/v2hoi/contracts.py` implements them, and export validates every artifact before writing a submission. How the team splits the stages is in [workflow.md](workflow.md).

## Run directory

```
runs/<run_id>/
  run.json                          dataset, upstream run, history of invocations
  inputs/<episode>/camera.json      Camera                  inputs   (platform)
  inputs/<episode>/masks.npz        Masks                   inputs   (platform)
  human/<episode>/human.npz         Human                   human
  objects/<object>/object.json      ObjectAsset             objects
  objects/<object>/mesh.glb         the object's mesh       objects
  motion/<episode>/motion.npz       Motion                  motion
  motion/<episode>/human.npz        RefinedHuman, optional  motion
  export/                           Tier 1 layout           export   (platform)
  score.json, summary.json          written by --score
```

`<episode>` is the six-digit episode index, as in the datasets. `runs/` is not committed.

## Conventions

- **Frame.** Everything is in the clip's camera frame: OpenCV axes (x right, y down, z forward), metres. The camera is static within a clip, so the camera frame is also the submission's world frame. The official evaluation fits one Sim(3) on the first frame, which makes every rigid choice of world equivalent.
- **Human convention.** SOMA-X parameters are stored in the SOMA convention (y up). `body.SomaBody` output multiplied by `diag(1, -1, -1)` is in the camera frame. The Tier 1 layout relates its human and its objects the same way.
- **Time.** One entry per video frame. Every per-frame array covers every frame of the clip and holds no NaN or inf. Occluded frames are filled, never left out.
- **Units.** Metres, radians, pixels.
- **Version.** Every file records `CONTRACT_VERSION`. Loading a file with another version fails.

## Artifacts

### Camera: `inputs/<episode>/camera.json`

| Field | Meaning |
|---|---|
| `width`, `height` | video size in pixels |
| `fx`, `fy`, `cx`, `cy` | pinhole intrinsics in pixels; the principal point lies inside the image |
| `camera` | physical camera name, when the dataset gives it (Track 1 does, Tier 1 does not) |

A physical camera gets one set of intrinsics, merged over its clips (`stages.inputs.merge_intrinsics`).

### Masks: `inputs/<episode>/masks.npz`

`human` and `obj`: `(n_frames, height, ceil(width / 8))` uint8, one bit per pixel packed along the width. Build with `Masks.pack(human_bool, obj_bool)` and read one frame with `masks.frame("obj", t)`.

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

The SOMA-X fields match the Tier 1 columns and feed export and local scoring. The MHR fields are what the official submission asks for. Their shapes follow SAM 3D Body's output (toolkit `v2d_sam3d_body`) until the official format is published. A clip has one person, so the identity should be one vector repeated over frames (`stages.human.lock_identity`).

### RefinedHuman: `motion/<episode>/human.npz` (optional)

Same fields as Human. The motion stage writes it when joint human–object refinement changes the human; export uses it instead of Human.

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
| `confidence` | scalar | 0–1, how much the tracker trusts the frame |

Every frame has a pose, occluded ones included.

### Export: `export/`

The Tier 1 layout that `v2hoi.score` reads, plus `mhr/episode_XXXXXX.npz` with the MHR fields for the official format:

```
export/meta/info.json
export/meta/episodes_metadata.jsonl
export/data/chunk-000/episode_XXXXXX.parquet
export/mesh/<object>/<object>.glb
export/mhr/episode_XXXXXX.npz
```

Poses are copied unchanged, since the camera frame is the world frame. Once `eval_reconstruction.py` is published, a second export backend will write the official artifact from the same run.

## Upstream runs

A run can name an upstream run (`--upstream`). Reads look in the run first, then in its upstream, then in that run's upstream. So a person working on one stage runs only that stage (and export) against a fixed snapshot of the others:

```bash
python -m v2hoi.run --run-id motion-fp-1 --dataset tier1 --episodes 7 9 10 19 21 \
    --upstream runs/baseline-v1 --stages motion export --backend motion=<name> --score
```

Re-running stages in an existing run overwrites their outputs and keeps the rest. `run.json` records every invocation: stages, backends, episodes, and git revision.

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
- Masks are stored whole-clip: about 0.4 MB compressed for an empty 900-frame clip, and about 400 MB in memory once loaded. If real masks turn out too large, they will move to per-frame chunks under a new contract version.
