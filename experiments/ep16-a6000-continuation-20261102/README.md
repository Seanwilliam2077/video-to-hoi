# EP16 A6000 continuation — 2026-11-02

This is a development snapshot, not a completed hand reconstruction or an official submission. The object turn in direction candidate v6 was visually accepted by the owner. The owner rejected the hand result in candidate v11 because penetration and finger poses remain wrong. The hand result must be rebuilt.

## Current evidence and limits

- **Direction v6:** source material motion and approximate image geometry guided the object pose correction across the full 360-frame sequence. The hand candidate preserves every accepted v6 object pose exactly. Visual approval of turning direction does not establish complete 3D accuracy.
- **Hand v11: rejected.** Native MHR arm and hand controls were optimized with geometric human–object penalties. No physics engine was used to produce this candidate. The previous all-frame human–box surface check passed at a 0.1 mm interior tolerance, but it omitted hand self-collision and did not establish finger accuracy or physical grasp.
- The follow-up audit found 49 proper nonadjacent triangle intersections inside hand 0 at frame 115 and 51 at frame 130. Fresh native decoding agreed with the rendering cache within 0.001 mm at six sampled frames. Positive self-intersections are established; zero counts in the other sampled frames are not a full-sequence collision certificate.
- No observed 21-keypoint finger fitting was performed. Bounds on native PCA coefficients are not anatomical joint limits and can retain an incorrect starting hand pose.
- The historical MuJoCo pilot did run 5,744 simulation steps, with prescribed mocap hand parts, zero gravity and only the object dynamic. It was rejected, did not correct articulated fingers, and was not used for v11. Its code and report are retained as evidence.
- Repository synthetic tests check engineering behavior. They do not establish source grasp accuracy, official penetration scores, or submission compliance.

The authoritative current decision is [REVIEW-AND-NEXT-STEPS.json](a6000-runs/hand-review-20261102/REVIEW-AND-NEXT-STEPS.json), with its [audit](a6000-runs/hand-review-20261102/audit.json). Historical reports and script names containing “fixed”, “final”, “pass”, or “selected” do not override the user rejection. `local-session/` preserves contemporaneous preparation and earlier intermediate reports; the remote review above takes precedence.

## Included code and results

- `a6000-runs/*.py`: the actual A6000 optimization, checking, rendering, preparation and review scripts, including failed iterations for traceability.
- `a6000-runs/direction-correction-v1/`, `hand-contact-20261102-v1/`, `matching-optimization-v2/`: reports, source-derived observations and compact diagnostic arrays.
- `assets/direction-v6-refined.pth.gz`: native checkpoint for the accepted object turning reference; its original hand quality is not accepted.
- `assets/hand-v11-rejected-refined.pth.gz`: the rejected hand checkpoint, retained to reproduce and diagnose the failure.
- `assets/object.glb`: the authored object used by both candidates.
- `assets/*full-scene.mp4`: full-timeline, 360-frame, 30 fps human–object comparison previews. Zoom videos supplement those scenes; they are not replacements for them. “Rejected” filenames explicitly identify the unaccepted hand result.
- `assets/runtime-source.tar.gz`: filtered runtime source allowlist from the A6000 migration, captured at upload time. It includes the project and runtime dependency sources needed to inspect the experiment, without foundation-model weights, environment packages or challenge dataset payloads. Original licenses are retained. The experiment scripts are stored separately above.
- `local-session/`: migration, proxy policy, matching qualification, direction and hand iteration preparation records and code.

Large result files are stored with **Git LFS**, in the same branch. Install Git LFS before cloning or run `git lfs pull` afterward. A source ZIP or a clone with LFS downloading disabled may contain only pointer files. Check [publication-manifest.json](publication-manifest.json) for sizes and SHA256 hashes.

## Restore and inspect

1. Obtain this branch with Git LFS enabled. Verify result sizes and hashes against the manifest before loading checkpoints.
2. Extract `assets/runtime-source.tar.gz` into an independent working directory. Copy this snapshot's `a6000-runs/` there. Preserve the original data and model assets separately under their original licenses; this upload does not mirror foundation-model weights or Track 2 data.
3. The scripts preserve their actual A6000 absolute paths under `/data1/defu/zly/nvda/continuation-20261102-v1`. Reproduce in that layout or explicitly adjust path constants in a separate checkout. Check dependencies and input paths before running a script. Do not run historical download/setup/finalizer scripts merely to view the result.
4. Decompress each `.pth.gz` with `gzip -d -k` (the decompressed checkpoint must match the manifest SHA256). Copy the direction checkpoint to `a6000-runs/direction-correction-v1/source-material-candidate-v6/refined.pth`, and the rejected hand checkpoint to `a6000-runs/hand-contact-20261102-v1/native-contact-candidate-v11/refined.pth` in that independent tree. The source video, original baseline checkpoints and existing licensed foundation-model assets remain prerequisites for rerunning optimization or rendering comparisons. Their original paths and existing release provenance are preserved in the migration/handoff records.
5. Use the original Track 1 source frame identity `000000` and the original 360-frame timeline. The comparison videos are available for review without rerunning inference. No official submission conversion or packing is included.

All external retrieval must use a proxy or a mirror. Network policy and pinned model provenance are recorded under `local-session/continuation-20261102/`. No new model inference or optimization was launched for this publication.

## Publication checks

The workflow’s explicit 14-file synthetic test allowlist was rerun on the clean source copy: **273 passed**. The deployed runtime copy initially reported 265 passed, one failure and seven setup errors because its existing model-cache symlink is intentionally rejected by asset path safety checks. The clean-source rerun passed without changing source code or weakening those checks. Both logs are retained. All 148 included experiment/session Python files passed a syntax parse. The two compressed checkpoints decompress to SHA256-identical copies of the A6000 originals. These checks do not validate finger accuracy or physical grasp.

## Required next work

1. Re-estimate visible fingers from original RGB crops, with confidence-weighted finger landmarks, silhouettes and temporal association. Treat hidden digits as uncertain rather than fabricated observations.
2. Verify hand identity and native joint order, axes and units. Fit anatomical native controls; any MANO initialization needs validated retargeting to MHR, not parameter copying.
3. Optimize source agreement and hand–object, hand–hand, finger–palm and hand–body collision constraints together. Preserve the accepted object turn as the observed reference.
4. Build articulated hands and forearms with finite tracking actuation, joint limits and calibrated per-link collision proxies. Validate the proxies against the full decoded skin geometry.
5. Demonstrate object dynamics under gravity/support and check contact forces, slip, image agreement and all collision categories on a representative short interval before a full-sequence run. Prescribed hand motion or merely calling `mj_step` is insufficient evidence of a recovered grasp.
