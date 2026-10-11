# Complete flow audit and input repairs (2026-10-11)

The owner flagged an undersized gray object and visible rotation jumps. The
previous dense run proves core execution and serialization, not an accepted
end-to-end official baseline. Its input had not passed mesh autoscaling and its
default postopt froze object rotation. No output-quality claim is retained.

The pinned NVIDIA [input instructions](https://github.com/nvidia-isaac/video_to_data/blob/193382a184308f6800331f3db829f76dc4294cfe/reconstruction/modules/v2d_cari4d/README.md#object-mesh)
require a physical-scale input mesh. Generated candidates are separate optional
preparation, and the README explicitly warns that autoscaling must precede
CARI4D. The core does not consume the separate SAM3D scale JSON or estimate mesh
scale during its eight released stages. Baking the predicted scale alone did
not satisfy the missing independent calibration/acceptance check.

| Item | Audit and action | Evidence / limits |
|---|---|---|
| Data and camera | Original Track 1 video, all 360 consecutive frames, estimated MoGe2 camera/depth; no reference poses/assets | Input, generation and native manifests pin hashes; original-video frame IDs retained |
| Segmentation | Original NVIDIA entry, original-video decoding, SAM2 CUDA hole filling area 8 | Complete H5 stream and pinned masks; no JPEG/cache substitution |
| Generated shape | Original NVIDIA SAM3D entry, both 25-step samplers and default generation flags | CPU residency adapter, unchanged sampling/precision methods; external human-anchored depth input explicitly recorded |
| Metric mesh scale | **Previously incomplete.** Original 9.2 cm input already projected too small before reduction | Generation projection about 38x31 pixels versus mask 82x72; alignment/export preserve original scale |
| Scale repair | Original NVIDIA `scale_mesh_srt.optimize_srt(mode=str)` and constant scale application | Explicit monocular adapter uses pilot relative object/camera views and estimated depth, not upstream stereo/SfM inputs. Drifted pilot views are excluded by recorded 0.15 m depth-consistency gate. Four fit frames and three separate held-out frames; fitted factor 1.438745, about 13.3 cm. Held-out fit improves but true dimensions remain unknown |
| Reduction for inference | Owner-selected original organizer geometry budget, 4096 faces and vertex budget | Canonical coordinates preserved; original vertex colors now transferred barycentrically on the full surface. Geometry and colors checked after GLB readback. Initial geometry-only preparation is superseded and rejected |
| Eight core stage calls | Every recorded dense-run stage compared with the pinned original launcher using identical resource options | `episode16-stage-audit.json`; audit intercepts execution and report writes, so it does not rerun models or alter results. New variant checked separately after completion |
| FoundationPose | Original first-frame registration, subsequent tracking, no GT rotation oracle or fallback trajectory replacement | Dense run: one registration, 359 tracking calls; translation increment up to 6.7 cm, rotation up to 30.5 degrees. Its reliability declaration does not prove observed fidelity |
| CoCoNet | Original checkpoint, all original frames and unchanged core inference | Dense output increased the maximum adjacent rotation change to 130.5 degrees (282 to 283). This happens before export |
| Postopt | Original whole-clip loss, optimizer/schedule, 300 requested steps | Dense default froze rotation, retaining jumps exactly. New variant enables the original `--optimize_object_rotation` option; no trajectory smoothing or replacement. Resource raster checkpointing retained; Kaolin fallback remains owner-permitted and unproven equivalent |
| Human conversion | Complete original float64 converter, original initialization/keyframes/refits/shared identity | Native decoder mapping and fit discrepancies recorded; no custom identity solver |
| Submission geometry | Original organizer packer/readback and single global object scale | Dense full and budgeted export extents both about 9.2 cm: packing did not cause the size error. Submission rows omit appearance, so packed readback video is gray; inference must retain appearance |
| Motion and video acceptance | Per-stage motion increments and exact refined/export equality, plus actual video inspection | These are diagnostics, not official scores. File/array success alone is insufficient; new run and videos require review before any quality acceptance |

The original scale-only broad-view attempt produced an excessive 61 cm candidate
with almost no loss improvement; it is excluded. A subsequent scale-only filtered
candidate was preliminary. Both are preserved as rejected/superseded diagnostics;
the subsequent v3 run used only `calibrated-preparation-v3` and its color-preserving budget.

The current control is `calibrated-budget4096-v4`: SRT factor 1.7660748798466102,
approximately 16.4 cm maximum dimension, rotation optimization enabled and
postopt learning rate 0.005. Its full native preview and original converter
completed. The earlier 13.3 cm row describes v3, not the current video.

The owner's [legacy-scale comparison](scale_search_comparison.md) retained v4's
exact 4096-face colored surface and all eight native-stage options. The original
30+11 candidate search selected a constant 1.2725218491985497 multiplier (about
20.8 cm). Source, model-option, video and mask equality checks passed. Both full
360-frame clips are available. Final angular jumps decrease, but FP middle-clip
mask fit collapses and translation increments worsen. This comparison does not
replace the control or establish true dimensions. Inherited SRT calibration
receipts describe the control-input chain; they do not validate the subsequent
legacy-selected scale against physical ground truth.

Per owner instruction, new comparisons stop at native videos and diagnostics.
The already mostly completed v4 conversion finished in 769.39 seconds; automatic
packed rendering was cancelled. No legacy-comparison submission export ran.

New launch guards require calibration evidence, disjoint held-out observations,
preserved appearance, original generation/mask hashes, and an explicit budget
opt-in. Historical uncalibrated diagnostics require a separate override. The
release's provided-mesh test condition, these generated-input adapters and the
optional rotation setting are distinct; this variant is not advertised as a
bit-identical original release reproduction.

No new dependencies or model downloads were needed. Windows analytic color
transfer checks cover corners, barycentric face colors, nearest outside points,
alpha and immutable target geometry. Calibration tests reject bad scale,
evidence, video, appearance and overlapping/insufficient held-out observations.
Local documentation cannot publish the existing Claude online pages in this
session; their copies remain out of date.
