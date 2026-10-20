# Standalone Track 1 Human batch execution

This runner connects the received Body input package to real SAM 3D Body,
fixed-identity native MHR fitting, independent MediaPipe image observations,
and native finger/wrist refinement. It does not use HandFlow, object contact
annotations, hidden references, the fake pipeline backend, or Track 2 assets.
Execution completion is separate from quality acceptance and official export.

## Prerequisites and two-episode pilot

Follow [the receiving guide](body-input-handoff.md) and preserve the successful
receiver and pinned native-reader reports. The batch runner requires an all-30
receiver gate: `passed: true`, `all_30_body_inputs_verified: true`, and
`videos_verified: 30`. A subset or producer receipt is insufficient. It rechecks
the selected input identity, video and artifact hashes before execution; the
SAM input adapter also verifies the mask/camera bindings it consumes.

Use the complete extracted package root, which contains both `body-inputs/`
and `data/`, rather than the inner `body-inputs/` directory. The two Python
executables can belong to separate installed environments. Supply all real
model/source paths explicitly; the runner neither downloads weights nor treats
the input archive as a model runtime.

Freeze the executing `tools/` and `src/` source tree before long jobs. Record
the actual model/environment inventory and execution device alongside receiver
evidence. Run EP16 (360 original frames) and EP11 (877 original frames, the
longest clip) first:

```powershell
& <sam-python> tools/run_human_batch.py run `
  --input-root <extracted-package-root> `
  --receiver-checks <successful-all30-receiver-report.json> `
  --output <new-result-root> --episodes 16 11 `
  --sam-python <sam-python> --mediapipe-python <mediapipe-python> `
  --checkpoint <verified-sam-model.ckpt> --mhr-model <verified-mhr-model.pt> `
  --converter <reviewed-mesh_to_mhr_params.py> `
  --sam-source <verified-sam-runtime-source> `
  --mediapipe-model <verified-hand_landmarker.task> `
  --inference-device cuda --fit-device cpu --pose-workers 4 `
  --fit-model-batch 1 --hand-device cpu
```

The device settings are explicit execution choices, not performance claims.
GPU fitting requires `--pose-workers 1`; validate its memory, numerical behavior
and measured speed before selecting it. Compare the two pilot videos, native
checks, crop observations, missing-side evidence and abrupt motion manually.
Only after that interface/effect review should the same command use
`--episodes all`. The episode selection does not invalidate matching completed
stages. `--continue-on-error` records a failed episode and continues independent
episodes; failure never becomes a successful result.

For a driver-only check correction, `--stage-runtime-root <previous-verified-snapshot>`
can retain unchanged external model tools and native modules while the new driver
runs its own `hands` and `check` wrappers. The selected tree must have a
`snapshot.json` containing every tools/src file hash and the required Human tools,
modules and viewer template. Changed, missing, escaping or undeclared files are
rejected; generated `__pycache__` files are excluded from that source inventory.
The source tree is selected before native modules are imported. The actual driver
and stage snapshot identities are recorded separately in the environment receipt.
The driver identity is not inserted into every model-stage cache key: existing
command paths, source hashes, model identities and verified output bytes still
determine reuse. Changed driver wrappers/checks receive their own new attempts;
there is no hash override or relabeling of previous evidence.

For a rendering-only correction, `--hand-comparison-tool <new-renderer.py>`
selects a different hand renderer while leaving model-stage tools in the chosen
stage snapshot. The renderer's adjacent `review_human.py` and
`human_review/template.html` are required and fingerprinted with it. Its command
and source hashes invalidate the hand-comparison cache; the override identity is
recorded separately from common model-stage inputs. The default renderer path
and its source identity remain unchanged when the option is omitted.

## Stages and evidence

Each episode keeps independent attempt directories, start/finish JSON receipts,
stdout/stderr logs, commands, timestamps, source/model/input identities and full
output SHA-256 inventories. A completed stage is reused only after verifying
every saved output identity. A changed configuration/input/source creates a new
attempt and preserves previous evidence. Inference can continue an interrupted
per-frame cache when the signature matches; other tools get fresh directories.

1. **Inference:** every original frame receives the delivered human mask and
   per-episode estimated intrinsics. Whole-image and manual-box fallbacks are
   excluded by the selected input profile.
2. **Baseline:** the reviewed native converter fits one identity and skeleton
   scale vector, then refits pose and exports native geometry for every frame.
3. **Observations:** SAM's original 70-keypoint ordering locates stable crops;
   MediaPipe supplies independent image targets. Mirror consistency, wrist/side
   association, duplicate rejection and weak color support screen targets.
   These binary weights are not measured visibility or calibrated probabilities.
   The selfie handedness classifier is recorded rather than globally forced
   without a per-clip interpretation check.
4. **Refinement:** fingers and bounded arm/wrist controls are fitted to supported
   observations while identity, skeleton scale and camera stay fixed. Default
   missing-hand behavior preserves the baseline outside supported intervals;
   bounded internal gaps of at most eight frames may receive temporal updates,
   without invented image targets. No unverified object contact or copied EP16
   phase labels are applied to another video. If no training-frame
   evidence supports fitting, the optimizer abstains and preserves the baseline
   byte-for-byte, with a separate abstention receipt.
5. **Visual review:** both baseline and hand candidate receive full-frame videos
   and original-frame viewers. A synchronized close-up video uses identical crops
   for source RGB, accepted observations and both native geometries. Each anatomical
   row draws only its own accepted observations, even when its crop overlaps the
   other hand. Side/count and variant-role headers stay within their image cell.
   Non-EP16 snapshot sets sample thirteen evenly spaced original frames including
   both endpoints; EP16 retains its previously reviewed grasp/release QA frames.
6. **Checks:** full parameter/frame coverage, one identity/scale vector, finite
   complete meshes, camera/frame consistency and native model identity are
   checked. Four deterministic original frames are independently decoded and
   compared with saved native geometry. Held-out image non-regression is reported
   per side. Every observed side, including a side with only one to four observed
   frames, can veto a regression. Fewer than five held-out observed frames still
   leaves that side insufficient to demonstrate improvement. Counts must agree
   and all image metrics must be finite and nonnegative.

The local candidate selector requires at least one sufficiently observed side
and no observed side failing the image non-regression screen.
Otherwise it selects the preserved fixed-body baseline. This is a reproducible
diagnostic selection, **not** a claim that unobserved fingers, real 3D contact,
global motion, metric scale or competition accuracy have passed.

The latest navigation files are `batch-progress.json`, each episode's
`current-result.json`, and `episode_XXXXXX/report/index.html`. Attempt receipts
are immutable evidence; navigation files are replaceable derived indexes. An
exclusive `batch.lock` prevents overlapping writes. After an unclean process
termination, inspect its recorded PID and actual process state before removing
a stale lock. Never delete it while the recorded job is still running.

## Dashboard and receiver handoff

Every local report links the baseline/refined videos, synchronized hand close-ups,
full-frame viewers and exact native checks. `dashboard-local-draft.json` describes
the handoff but intentionally creates no fake public URLs or accepted records.
The native Human NPZ remains authoritative; review meshes are derived products.

Publish the result-specific package and a depth-scale record with actual hashes
to a receiver-accessible release/team store before adding dashboard declarations.
Record actual executor, environment and reviewer. The depth-scale record must
state the monocular uncertainty and whether shared human/object scale has been
verified; writing `1` is not verification. Bind the exact upstream input record,
and use `submitted` while structural/quality or dependency acceptance is pending.
The standalone Body-ready input package does not by itself complete the board's
broader Inputs stage. Existing dashboard predecessor and acceptance rules remain
in force. No SOMA placeholder, copied scale, fabricated URL, or synthetic pass may
be used to mark a native result checked or accepted.

Large outputs stay outside Git. A delivery note under `deliveries/human/` should
link exact package identities, receiver checks, pilot review, all-30 coverage,
abstentions, selected candidates and unresolved failures. A merged code change
or completed execution does not establish [R1–R6](track1-requirements.md)
submission compliance.
