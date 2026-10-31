# Archived execution recipes

These files preserve the recipes actually executed for the two controlled
pilots, including their original machine-local paths. They are an audit and
reproduction reference, not a one-command portable installation, native importer,
new all-30 launcher or replacement for the installed project tools.

## Layout and required dependencies

Keep task recipes at one task root, beside `inputs/`, `weights-receipt.json`
and `runtime01/`. The included `runtime01` is the exact small native fitting,
adapter, evaluator and renderer source snapshot. It is bound by `snapshot.json`.
The inference source and weight revisions, dependency versions, actual bootstrap
commands, crop/camera inputs, output receipts and license notices are retained
in the adjacent records. Large dependencies are intentionally absent:

- The exact generic EgoHand checkpoint and SOMA-X rig/neutral geometry, including
  the recorded MHR LOD1 compatibility assets. Fetch only the file allowlists in
  the weight plans at their pinned revisions and verify their object identities.
- The official IsaacCapture and DINOv3 implementation at recorded commits.
  Recreate only the recorded code/configuration allowlist. Bundled example
  videos, images, dataset calibration or challenge assets are not experiment
  dependencies and must not be fetched as a shortcut.
- The exact permitted full-body MHR decoder and original Track 1 EP16/EP11
  videos, manifests, frozen SAM baseline, original MP observations and bounded
  MP candidate, all identified by input hashes.
- A Python environment matching the environment records. The original Windows
  run reuses an existing Torch installation read-only through `.pth` files.
  Those paths are evidence, not automatically valid paths on another host.

Source copies preserve original bytes. Recreate the original task layout before
using a recipe; moving scripts into a separate subdirectory changes their
`__file__`-relative inputs. Absolute paths in the input/job records and recipes
must also be relocated. Preserve input bytes, record the new paths and manifest
hashes, and create fresh output/job/receipt locations. Existing completed or
failed attempts must not be overwritten. Relocation has not been independently
verified as a portable package import.

## Executed order

1. Verify source/video/camera/crop/weight identities and the environment. Use only
   the two authorized Track 1 episodes. Do not use example or Track 2 data.
2. `run_controlled_egohand.py` performs complete inference and official adaptive
   filtering. Actual commands used CPU float32, 12 threads, hand batch 8,
   eight source frames per chunk and `--with-smoothing`. No `--frame-ids` subset
   is used for either complete run. The process-local CPU geometry wrapper does
   not change learned weights; the separate backend-parity receipt checks it.
3. `prepare_after_inference.py` verifies complete inference and binds new target
   and fitting jobs. The reusable project entry point is
   `tools/adapt_egohand_observations.py`, separately applied to raw and filtered
   outputs. The exact native fitting commands are in the fit-finished records.
4. `tools/evaluate_egohand_pilots.py` freshly decodes every native frame and keeps
   MP-reference evidence separate from generated-target consistency. Original
   fixed thresholds and development masks remain unchanged.
5. `render_egohand_comparison.py` produces synchronized videos, one-third-speed
   critical windows and sampled images. Its final summary is written after
   complete decoding of every output video. `grasp_motion_diagnostics.py`
   records opening and motion curves without new fitting or acceptance.
6. `summarize_completed_pilots.py` and `package_local_reviews.py` produce the
   compact evidence and self-contained offline video packages after both clips
   complete. They do not publish, change catalog selections or launch all 30.

The archived `run_postprocessing.py` and job receipts describe the actual local
coordination. They contain completed-attempt paths, so executing them unchanged
is not a safe way to request a new experiment. Use the project CLIs with a
documented fresh attempt and explicit dependencies instead.

Full-clip smoothing observes all raw frame estimates, including future and
downstream stride-5 frames. Excluding those targets in the native fitter does
not create an end-to-end unseen-frame test. Passing geometry, coverage or
checksum checks does not establish true hidden-hand, contact, scale or official
competition accuracy.
