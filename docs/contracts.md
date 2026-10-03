# Stage contracts

The pipeline's stages exchange data through files in a run directory. This page specifies those files; `src/v2hoi/contracts.py` implements their formats. [track1-requirements.md](track1-requirements.md) is the single active requirement specification. The current exporter writes an internal artifact, not a verified official submission, and validates only the refined human, motion, and object mesh. How the team splits the stages is in [workflow.md](workflow.md).

**Implementation status, 2026-10-03:** the directory layout and artifact tables below describe implemented **contract v1**. The official evaluation kit is now available and exposes incompatibilities with v1; documenting its format does not migrate existing artifacts. The proposed v2 section at the end is a future contract change, not a claim that `CONTRACT_VERSION`, adapters or backends have changed. See [evaluation.md](evaluation.md) and gate G0 in [implementation-plan.md](implementation-plan.md).

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

- **Frame.** V1 stores the scene in the clip's OpenCV camera frame (x right, y down, z forward), in metres. [R1](track1-requirements.md#r1--one-human-derived-alignment-for-the-complete-scene) requires one Sim(3) fit using only human geometry at the first reference frame's explicit original-video ID and the same transform applied to human and object for all frames. Later scored subsets retain that alignment. External row indexing is an adapter concern, not an alternative rule. Whole-clip, per-frame and separate object fitting are not permitted replacements. Alignment cannot repair relative scale or coordinate mistakes; verify units, axes and parameter semantics explicitly.
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

V1 can merge intrinsics over a physical-camera group (`stages.inputs.merge_intrinsics`). A shared camera name alone does not verify unchanged crop, resolution or intrinsics; proposed real backends must test that assumption and retain an independent per-clip fallback.

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

The SOMA-X fields feed the current internal parquet serializer and can support self-created synthetic checks. The MHR fields are historical placeholders, not a verified official representation. In particular, v1's separate pose arrays and `mhr_scale[28]` must not be renamed or padded into the official native `pose[T,136]`, `scales[68]`, `shape[45]` contract. Implement a validated producer/converter before export. A clip has one person, so the v1 identity helper repeats a clip-level vector (`stages.human.lock_identity`); the proposed native contract stores static shape/scales once. None of this authorizes reading Track 2 data.

### DepthScale: `human/<episode>/depth_scale.json`

| Field | Meaning |
|---|---|
| `scale` | metric depth = `scale` × `Depth.depth`; one value per clip, positive |
| `method` | how it was estimated |

The intended v1 convention uses human-derived scale to align depth before object mesh scaling and tracking. Real producers and consumers of this scale are not implemented; the fake human writes a fixed value. A common scale convention would not guarantee metric accuracy or contact. Proposed refinement treats human scale as an uncertain prior and checks it against independent image and support evidence.

### ObjectAsset: `objects/<object>/object.json` and `mesh.glb`

V1 stores one asset per object name, so its serializer shares that file across the object's clips. This is a repository design choice, not a confirmed official requirement or permission to pool evidence across episodes. Keep cross-episode reconstruction disabled pending organizer clarification; the proposed v2 distinguishes asset identity from the episodes allowed to contribute evidence. `mesh.glb` is in metres, in the object's canonical frame; v1 checks that its largest bounding-box side lies between 2 cm and 3 m.

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

The contract requires a pose on every frame, occluded ones included. The `stages.motion.hold_missing` helper can hold the preceding pose, or the first available pose for leading gaps, with confidence 0 on filled frames. No registered real tracker currently uses this helper, and real occlusion repair remains to be implemented.

V1 validates finite values and proper transforms but does not enforce the documented confidence interval. The scalar is not a calibrated probability and does not distinguish translation, orientation and visibility uncertainty. The registered refine backend currently passes inputs through unchanged.

### RefinedHuman and RefinedMotion: `refine/<episode>/human.npz`, `refine/<episode>/motion.npz`

Same fields as Human and Motion, reserved for the refinement output. The registered backend currently copies its inputs unchanged; temporal, occlusion and contact repairs are planned. Export reads these artifacts. It detects a raw human or motion artifact from a nearer upstream run than the refined artifact, but it does not yet detect a same-run overwrite or all changed dependencies. Re-run refine whenever its inputs change.

### Internal export schema: `export/`

The current backend writes an internal parquet layout plus `mhr/episode_XXXXXX.npz` with decomposed MHR fields. This is not the native-MHR input schema of the reference-based diagnostic scorer and is not the official submission payload. The name `tier1` in the backend is a legacy serialization label, not permission to read Track 2:

```
export/meta/info.json
export/meta/episodes_metadata.jsonl
export/data/chunk-000/episode_XXXXXX.parquet
export/mesh/<object>/<object>.glb
export/mhr/episode_XXXXXX.npz
```

Poses are copied unchanged in this internal format. The official submission exporter is not implemented, although the official kit is available. Reference-dependent CD/ACC cannot be evaluated locally on public Track 1 ground truth because none is released. Current official PEN is predicted-hand penetration into the submitted object, not a reference difference, but its final value uses the reference-derived alignment scale. Unaligned local penetration is a diagnostic, not final official PEN. The native-MHR reference diagnostic described in [evaluation.md](evaluation.md) requires separately validated inputs, an authorized decoder and a permitted reference; it does not convert this v1 export or establish official equivalence.

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
- There is no public Track 1 ground truth for local official scoring. Test metric primitives and the native-MHR reference diagnostic with independently generated geometry using the explicit CI allowlist in [development.md](development.md#tests). Do not restore historical Track 2 normalization, reference defaults or scorer modes. A permitted synthetic reference validates mathematics, not challenge quality or official equivalence.
- Masks are stored whole-clip: about 0.4 MB compressed for an empty 900-frame clip, and about 400 MB in memory once loaded. If real masks turn out too large, they will move to per-frame chunks under a new contract version.

## Proposed contract v2: not implemented

The migration is a separate reviewed contract PR under G0. It must retain source v1 files unchanged, declare which fields can be converted, and fail explicitly where a trustworthy native reconstruction cannot be recovered. Array dimensions alone do not establish equivalent MHR semantics.

| Proposed record | Required meaning and validation |
|---|---|
| `NativeHuman` / `NativeMHR204` | Proposed authoritative native MHR `pose[T,136]`, static `scales[68]`, static `shape[45]`, original frame IDs/timestamps and exact rig/decoder identity. Native parameter units are not uniformly metres/radians; follow the pinned converter, particularly for root translation. Preserve the upstream native artifact. Derive SOMA diagnostics from this authority and check forward geometry and export/reload consistency; never maintain independently optimized MHR and SOMA truths. |
| `Observations` | Per-source masks/keypoints/depth/flow as applicable, validity and visibility, frame IDs, producer/model hashes and uncertainty. Record whether evidence is predicted, independently reviewed or held out for evaluation. Estimator confidence, fit residuals and posterior uncertainty are separate fields. Optimizers must not freely lower observation weights to hide errors. |
| `ObjectGeometry` | Mesh hash, canonical frame, unit/scale convention, topology/collision representation, generating source episodes and their permitted use. The official transform has rotation, translation and a separate scale; record whether scale is baked into vertices or applied explicitly, exactly once. Freeze canonical origin across comparisons and verify the compiled mesh after official welding, simplification and padding. Per-episode references may share geometry only under the approved data-use setting. |
| `Symmetry` | Typed finite or continuous symmetry group with axis/origin or transforms, source evidence and ambiguity. Pose comparison/refinement must account for equivalent orientations; a continuous representative for export is not an observed orientation. |
| `MotionState` | Every frame's object transform, timestamps, observed/inferred/invalid evidence status, uncertainty by observable component, candidate/hypothesis identity and accepted lineage. Complete finite arrays alone do not establish reconstruction quality. |
| `SupportAndContact` | Scene-plane/patch evidence and uncertainty; typed edges between body, object and scene patches, with time intervals and normal/tangential constraints. Simultaneous support, grasp and sliding are permitted; no-contact is a hypothesis. Do not apply no-slip constraints to sliding edges. |
| `ArtifactManifest` | Immutable input/output hashes, code, configuration, model, runtime and coordinate-adapter identities; source policy and derivation links; status distinguishing interface fixtures, model execution and accepted reconstruction. Unknown provenance or fake artifacts cannot qualify as a real baseline. |
| `TrialAndExport` | One coherent candidate containing human, geometry, motion and diagnostics, with an explicit expected object/episode manifest and all source-frame IDs, including occlusions. Metrics and overlays must reload this exact export. Record the single shared alignment and its reference-frame index; world-posed object CD and reference-relative ACC must not be replaced by shape-only or self-smoothness diagnostics. Write to a temporary location, validate, then promote atomically while retaining the previous accepted trial. |

The proposed dependency graph includes camera, observations, depth/scale, native human, mesh, support/contact, motion and refinement. Changing any consumed hash invalidates all descendants, including affected clips when sharing is enabled. Upstream hop counts and file existence are insufficient, including for same-run overwrites. Each iteration has a fixed input manifest, bounded variables, a declared acceptance test and a rollback target.

Migration acceptance requires independently generated transform, scale, time-index and representation fixtures; a native real Track 1 clip; source-policy checks; export/reload/render consistency; and validation with the pinned official kit's supported interfaces. No Track 2 data or derived values may be used to pass these checks. Contract validation, successful execution and reconstruction fidelity remain separate gates.

The owner's 2026-10-03 final requirements also make camera provenance explicit: Track 2 camera calibration and iron/bowl/table meshes are prohibited, as are transformed or cached derivatives. V1's dataset label, path checks and free-text asset `source` do not establish this provenance. Production acceptance must verify the derivation chain without opening prohibited assets for comparison. See [workflow.md](workflow.md#final-owner-requirements-2026-10-03) for the dated conversation source and all six requirements.
