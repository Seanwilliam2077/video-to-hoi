# Official-method fidelity and Windows resource changes

The [completed legacy scale-search comparison](scale_search_comparison.md) is an
explicit preprocessing experiment, not an unmodified MHR release reproduction.
It uses the same colored 4096-face surface, current FoundationPose weights and
all v4 MHR settings. All 30+11 original candidates were computed. Whole-video
tracking is worse in the middle despite smaller final angular jumps, so quality
is not accepted. No trajectory smoothing, replacement or extra registration was
introduced. Submission conversion is deferred until explicit owner confirmation.

The original Windows experiment in `episode16-result.json` is retained and its
official-baseline label is retracted. The corrected run uses Track 1 episode 16
only. Execution is not a claim of Linux output equivalence or reconstruction accuracy.

**Audit correction:** the dense generated input did not complete independent
mesh autoscaling and its default frozen rotation retained visible jumps. Core
execution/packing passes are preserved; baseline quality is unaccepted. The
[complete flow audit](pipeline_audit.md) records all stages, input repairs and
the explicit monocular calibration adapter. The new reduced variant preserves
appearance and enables the original optional rotation optimizer.

| Component | Corrected behavior | Windows change and evidence |
|---|---|---|
| Video segmentation | NVIDIA `video_to_masks`, original MP4/Decord, default SAM2 hole area 8 | Original CUDA connected-components extension compiled into the preserved Windows wheel; all 360 masks regenerated. No JPEG inference cache. |
| Object geometry | NVIDIA `image_to_mesh`, both official 25-step samplers, original guidance, precision, decoders and default postprocessing flags | Strict checkpoint loads staged between CPU and CUDA; no hull, voxel/step override or inference-mesh simplification. 176,178 vertices / 352,348 faces generated. |
| Object grounding | Supported external-depth input with MoGe2 depth, official human-anchored depth scaling and supplied intrinsics | Lossless RGB/mask/depth inputs; only the official returned scale is baked into canonical geometry. This explicitly supplied conditioning input is recorded; it is not a claim that optional depth arguments remained unset. |
| FoundationPose | Original registration/tracking policy and complete original scorer forward | Independent render/refine batches 64/128 and depth warp 32. Dense geometry is preserved. Original 512/1024 batches exceeded memory with the generated surface. |
| Coupled prediction | Original CARI4D checkpoint and sliding-window inference on every original frame | MoGe/SAM3D/render batches 4, SAM3D refit batch 64. Input/output/source identities recorded. |
| Contact refinement | Whole-clip objective, original optimizer/schedule, 300 requested steps, original contact and temporal methods | Independent silhouette raster calls checkpointed in batches 16. Original module AST is preserved after removing this wrapper. Synthetic loss difference 0; maximum gradient difference 8.94e-8, including occlusion and a partial final batch. |
| Penetration | Owner permits the existing Open3D/PyTorch3D fallback | **Kaolin equivalence is unproven.** Analytic value/gradient checks establish only the tested operations. This remains an explicit exception. |
| Native human export | Complete official `mesh_to_mhr_params.convert()` flow, float64 defaults, original initialization/keyframes/refits/iterations | Supported model-call batch 64 only. No custom native warm start or median-identity fit. Fitting to constant shape/scales can change geometry; its discrepancy is reported separately. |
| Submission packing | Unmodified official packer and `budget_mesh` | Organizer-required weld/decimation/padding to 4,096 face and vertex rows happens after inference. Readback is compared exactly to the official budget output, and sampled surface change is recorded separately. |

`sources.json`, compatibility patches, weight receipts and run manifests identify
the actual local materials. Optional simplification/texture imports are lazy;
their packages are excluded from default setup. The isolated legacy MoGe/utils3d
API sources affect only the object-generation process in the same project venv.

The silhouette fixture also preserves the upstream constant-white rendering
case's zero vertex gradients. No antialias operation or altered objective is
silently introduced to improve it.

Monocular shape, grounding and contact accuracy remain empirical questions.

## Owner-selected pre-inference 4096-face variant

The dense reference above is preserved. A separate, explicitly selected variant
uses the organizer's unmodified budget code before tracking and optimization,
rather than only at submission packing. The prepared episode 16 mesh has 4,096
active faces and 2,052 vertices; its full source has 352,348 faces. Reduction
took 0.67 seconds without new dependencies or downloads. Original coordinates
and scale are preserved; only zero-area padding and unused vertices are removed.
This changes the input surface and can change tracking, contact and penetration.
It is recorded as an algorithm deviation, requires `--allow-budgeted-mesh`,
and has not established reconstruction equivalence or end-to-end speedup.
The unchanged distance/min/square/backward kernel on 184,320 seeded synthetic
queries took 0.987 seconds on the dense surface and 0.0117 seconds on the budget
surface (84.7x). Query distances also changed by 0.199 mm mean absolute difference.
This supports kernel-level performance only; full variant inference remains
unvalidated. [The variant record](mesh-budget-episode16-result.json) preserves
both mesh identities, source/function hashes, query shape and timing samples.

The initial generated-pose projection differs from the observed object mask;
`object-generation/projection-diagnostic.json` records that diagnostic without
rescaling the model output to hide it. No challenge answers or official scores
are available locally, and one episode is not a complete 30-episode submission.
