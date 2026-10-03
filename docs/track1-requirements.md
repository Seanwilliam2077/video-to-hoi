# Current Track 1 requirements

**Effective 2026-10-03. Source: the project owner's latest explicit six requirements in this conversation.** This is the single authoritative statement of current Track 1 acceptance requirements for this repository. It replaces conflicting earlier repository rules, organizer-answer summaries, scorer defaults and planning assumptions. Historical material may explain past decisions but is not an alternative active specification.

The requirements below govern implementation and review. They do not claim that the current reconstruction pipeline or exporter already complies. [track1-compliance.md](track1-compliance.md) records implementation gaps; [evaluation.md](evaluation.md) records source-pinned external mechanics and diagnostic limitations; [implementation-plan.md](implementation-plan.md) orders the remaining work.

## R1 — One human-derived alignment for the complete scene

Fit one Sim(3) using only corresponding predicted and reference human geometry at the **first frame of the reference trajectory**. Apply that same constant scale, rotation and translation to both the human and every object throughout the sequence. Human and object must retain a consistent coordinate frame and relative scale.

Do not fit an independent object alignment, use whole-sequence fitting to replace the required first-frame fit, select a more convenient frame, or refit later frames to conceal drift. Alignment is an evaluation operation that requires a permitted reference; reconstruction and export must not invent or retrieve an unavailable reference.

The reference's first frame must have an explicit original-video frame ID. Prediction and reference frame arrays must map to those original IDs. A local slice offset or the first row selected for scoring must not silently replace this ID. If a scoring subset begins later, retain the alignment from the first reference frame. Mapping an external kit's first-scored-array entry to the required original reference frame is an adapter task, not a competing requirement or a reason to reopen R1.

## R2 — Native MHR human trajectories

Submit the human trajectory in native MHR parameters with a verified decoder/converter contract. SOMA-X parameters, arbitrary arrays named `mhr`, and per-frame human meshes do not replace the required representation. Derived human meshes are useful for rendering and evaluation, but the native parameters remain authoritative.

The external packer's parameter dimensions and units are documented in [evaluation.md](evaluation.md#official-packer-inputs-and-final-payload). A valid-looking array shape alone does not establish a correct native mapping or official export.

## R3 — Object error on posed world-space geometry

Evaluate object Chamfer on the object mesh after its actual scale, rotation and translation put it in the shared world/scene frame, followed by the R1 human-derived alignment. For canonical vertices `V`, the scene geometry is `P[t] = s_obj * R[t] * V + translation[t]`.

A canonical-shape comparison or independently aligned object comparison cannot replace this quantity. Verify the mesh/pose pairing, canonical origin and exactly-once scale application. Exported or compiled geometry must preserve the intended world-space reconstruction.

## R4 — Reference-relative acceleration error

For trajectory `x`, define `D2(x)[t] = x[t+1] - 2*x[t] + x[t-1]`. ACC compares prediction and reference second differences after the common alignment: `norm(D2(prediction) - D2(reference))`.

Prediction-only acceleration is a separately named diagnostic. A frozen prediction can have zero self-acceleration and still be wrong. Use original frame identities to define consecutive differences; do not differentiate across omitted-frame gaps. Exact evaluated joints, object trajectory, units and reductions are documented with their source/version in [evaluation.md](evaluation.md#acceleration-error).

## R5 — Every object, every frame, including occlusion

Produce a continuous pose trajectory for every required object on every input-video frame, including full occlusion and out-of-view intervals. Validate the complete expected episode/object/frame manifest against decoded source videos. A successful subset is not a complete submission.

Invisibility or low confidence does not authorize deleting a frame. Preserve uncertainty internally and export an explicit estimate. Coverage alone does not prove motion fidelity: a repeated stationary fallback is not evidence that the hidden object was stationary.

## R6 — Track 1-only reconstruction assets and estimates

Reconstruct all challenge object assets and estimate all challenge parameters, including camera intrinsics, from the permitted Track 1 inputs and metadata. Do not reuse Track 2 assets or parameters, including iron, bowl, table or round-table meshes, camera calibrations, trajectories, renamed/rescaled copies, or cached derivatives.

The repository's broader prohibition remains in force: never download, load or use Track 2 data or derivatives for development, tuning, testing, scorer checks, normalization, comparison, ranking, demonstrations or submission. There is no evaluation-only exception. Unknown provenance must be resolved before use. Independently constructed synthetic fixtures may test mathematics and interfaces; they are not challenge submission assets. An injected decoder or a provenance label does not authorize prohibited data or establish provenance by itself.

## Applying the requirements

All active documents, code interfaces, tests and review gates must follow R1–R6. Keep external-source observations and additional engineering proposals clearly identified; neither supersedes these six requirements. Record real results, synthetic tests and unimplemented work separately. Local diagnostics must state their decoder, roles, sampling, reference provenance and equivalence limits. They must not imply access to hidden Track 1 reference data or an official score.
