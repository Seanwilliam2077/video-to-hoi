# Legacy scale-search comparison (2026-10-11)

This is a controlled Track 1 preprocessing experiment on episode 16, not the
default provided-mesh MHR release test or an official accuracy score. The owner
requested the original legacy NVIDIA candidate search while preserving the
current MHR backend, weights and settings.

## Control and changed input

The control is `runs/windows-episode16/calibrated-budget4096-v4`. Its native
videos cover all 360 original frames with the colored 4096-face object. Its
input maximum dimension is 0.163564 m after the previously recorded SRT fit.

The comparison is `runs/windows-episode16/legacy-scale4096-comparison-inference`.
It uses the control's exact 2052 vertices, 4096 triangles and vertex colors;
one uniform scale is the only surface change. No second decimation, new mesh
generation, mask modification or trajectory smoothing is performed.

The old generation scale and SRT scale are divided out to express this same
surface in SAM3D's original normalized units. This allows the original search's
fixed candidate range to retain its intended meaning. The chosen normalized
scale is 0.4329458128078818, or 1.2725218491985497 times the control mesh.
The resulting extents are approximately 0.208111 × 0.205351 × 0.208191 m.
These are estimated dimensions, not measurements of the real object.

## Original algorithm and I/O adaptation

Source: NVlabs/CARI4D revision `71fa7cbe46081467edadd11ab534b0c14aa9d913`,
`tools/estimate_scale.py` and `tools/chamfer_dist_np.py`. Only these two code files
were copied; no legacy datasets, weights or environment are loaded.

The complete original `estimate_metric_scale` function is extracted and executed:
30 coarse candidates from 0.03 to 3.0; eight fine candidates spanning 0.8 times
the smallest to 1.2 times the largest of the top three, plus those three retained
candidates; five FoundationPose registration iterations for every candidate;
10,000 sampled surface points; the original erosion/bilateral depth filters;
the original bidirectional unsquared Chamfer distance divided by raster mask IoU.

The only function edits replace its shared debug folder and final UV-only OBJ
writer. The actual vertex-color mesh is preserved. The search's math, sample
count, candidate ranges, registration calls and selection are unchanged. The
current MHR FoundationPose implementation and the existing scorer/refiner weights
are used, so the experiment changes preprocessing while retaining the backend.
Full source hashes, adapted code and raw coarse/fine results are saved alongside
the preparation. The original search's cache is reused after I/O interruptions.

The generation frame is original frame zero. Its lossless RGB, object mask,
estimated intrinsics and human-anchored MoGe2 depth come from the control.
Depth is expressed in integer millimetres for the legacy PNG-equivalent input
contract. No ground-truth depth, pose or size is available or consumed.

## Settings and comparison criteria

All eight MHR stages retain the control settings: rotation optimization enabled,
postopt learning rate 0.005, 300 requested optimization steps over all 360 frames,
FoundationPose render/refine/depth batches 256/512/32, preview render batch 8,
and the same original checkpoint and permitted Kaolin fallback. Both videos use
the same CPU preview encoding; encoding changes are excluded from this comparison.

Check the original FoundationPose visible-mask IoU across the whole clip and
outside search frame zero; report its segments separately. Compare per-stage
translation/rotation increments and inspect the video at the largest changes.
These diagnostics cannot establish true size, correct hidden motion or an
official score. Inference execution and quality acceptance are reported separately.

## Export policy

The owner asked to perform expensive submission conversion only after explicit
submission confirmation. Iteration stops at native inference, videos and motion
diagnostics. The already mostly completed control conversion was allowed to
finish: 360 frames, 757 seconds in the original converter, 769.39 seconds for
the complete export process. Its automatic packed-export rendering was cancelled.
The legacy search comparison does not run submission conversion or packing.

## Reproduce

Use the root `.venv` after dot-sourcing `tools/windows/activate.ps1`:

```powershell
python tools/windows/run_legacy_scale_comparison.py `
  --control-run runs/windows-episode16/calibrated-budget4096-v4 `
  --preparation runs/windows-episode16/calibrated-budget4096-preparation-v4/preparation.json `
  --output runs/windows-episode16/legacy-scale4096-comparison-new
```

Use a new output directory. `--resume` reuses the original saved candidate search
after an I/O interruption. It does not silently skip the full native video run.
No packages or model weights were downloaded for this experiment.

The source-donor option currently defaults to `D:/Projects/CARI4D_Win`: the tool
reads the two pinned Git blobs from that checkout, then preserves them under
`third_party/cari4d-legacy-scale`. The donor supplies code only. Keep it available
for a fresh search invocation until the source-loading tool supports verified
cache-only loading; a completed native run does not require the donor environment.

## Complete clip results

**Owner review correction (2026-10-11):** the latest candidate's visible scale
is reasonable, but its first-frame orientation is wrong: the grass face should
be on top, while the reconstruction puts it on a side. Early mask agreement is
not orientation acceptance. The next experiment freezes this scale and geometry
and repairs orientation only; see the fixed-scale review in the Windows README.

All eight native stages and their pinned-launcher command audit passed. Both
variants retain the same official source tree, model options, video and masks.
They cover 360 frames at 30 fps. No new environment, packages or weights were
needed. The comparison is complete; it is **not accepted as an improvement**.

| Prediction diagnostic | v4 control | Legacy scale search |
|---|---:|---:|
| Approximate maximum mesh dimension | 16.4 cm | 20.8 cm |
| FP visible-mask IoU, whole clip mean | 0.474 | 0.365 |
| FP visible-mask IoU, frames 0–119 mean | 0.535 | 0.648 |
| FP visible-mask IoU, frames 120–239 mean | 0.487 | 0.049 |
| FP visible-mask IoU, frames 240–359 mean | 0.399 | 0.399 |
| Final maximum adjacent rotation change | 52.65 degrees | 22.06 degrees |
| Final maximum adjacent translation change | 2.22 cm | 3.50 cm |

The larger candidate improves early FP mask fit but loses tracking severely in
the middle of the clip. Final optimization reduces its angular jumps, while
translation increments worsen. The sampled final overlays still show orientation
and placement differences from the source. Smaller angular increments alone do
not establish accurate motion, and later refinement does not erase the raw FP
failure as evidence. These diagnostics are not ground-truth errors or competition
scores. The control is preserved, also without a quality-acceptance claim.

The derived review video is
`runs/windows-episode16/legacy-scale4096-comparison-inference/review/v4-vs-legacy-scale.mp4`:
source on the left, final v4 in the middle, final legacy-search output on the right.
It is encoded with NVENC at 2304 × 576, and all 360 frames decode correctly.
This GPU encoding applies only to the derived review; the two original native
preview encoders and model inputs are unchanged. Nine sampled frames were visually
inspected (0, 60, 120, 160, 180, 213, 224, 282, 359); this is not full temporal
acceptance of every frame. The FP IoU/depth plot accompanies the video.

The first search executed all 41 candidates in the 104.92-second process, which
then failed at the final UV-only writer. Original candidate caches were preserved.
The corrected vertex-color I/O adapter reused these completed candidates; its
0.37-second receipt is cache finalization time, **not full search runtime**.
The sampled whole-device search peak was about 15,519/16,303 MiB despite a 75%
PyTorch allocation cap. That cap does not limit driver, desktop and other GPU
allocations; this attempt did not meet a 75% whole-device ceiling.

The compact durable evidence is [the comparison record](legacy-scale4096-comparison-result.json).
The existing online team pages cannot be republished from this session.
