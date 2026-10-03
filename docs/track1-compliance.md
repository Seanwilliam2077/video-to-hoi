# Final Track 1 requirements: implementation audit

Updated 2026-10-03 for the current reconstruction pipeline and replacement reference diagnostic. The single active requirements are [R1–R6](track1-requirements.md), from the owner's latest explicit instruction. External-kit mechanics are separately pinned in [evaluation.md](evaluation.md). Historical audit statements about the former scorer are superseded.

**Verdict: the current pipeline is not ready for a compliant final submission.** The native-MHR reference diagnostic implements the required evaluation structure with explicit permitted inputs. The reconstruction runtime still uses fake stages, a pass-through refiner and an internal v1 exporter. Diagnostic code and synthetic tests do not establish real reconstruction, official export or official equivalence.

## Requirement-by-requirement findings

| Final requirement | Current implementation | Verdict and required evidence |
| --- | --- | --- |
| R1: one first-reference-frame human Sim(3), shared by the complete scene | The reference diagnostic fits the human at the first expected original reference frame and retains the transform for human/object and later scored subsets. Pipeline validators still check primarily shape, finiteness and rigid transforms. | **Diagnostic structure implemented; real geometry unverified.** Validate actual human/object units, axes, image transforms, exactly-once scale, initial placement and later drift. Reference alignment belongs to evaluation, not an exporter without a reference. |
| R2: native MHR human trajectories | The diagnostic accepts native pose 136 / structural scales 68 / shape 45 and decodes through an explicitly supplied callable. The pipeline `Human` and `Tier1Export` still retain SOMA-X plus decomposed wrapper MHR arrays; no official packer or verified native bridge is registered. | **Diagnostic input contract implemented; submission path missing.** Validate a real authorized native decoder and bridge through geometry, fixed identity, serialization and read-back. A synthetic decoder or an NPZ named MHR does not establish this. |
| R3: posed world-space object Chamfer after common alignment | The diagnostic constructs geometry from canonical mesh samples, object scale and per-frame pose, then uses the shared human alignment and sum-of-directed-means Chamfer. Its CLI samples the declared canonical mesh; it offers no independent object fit. | **Local diagnostic implemented; official integration missing.** Verify the actual compiled submission mesh, host roles/sampling and native decode. A generic point-cloud helper or diagnostic report cannot establish official equivalence. |
| R4: reference-relative second differences | The diagnostic compares second differences of aligned human body joints and object translations against the permitted reference, respecting original frame IDs and gaps. Self-acceleration remains separately named. | **Diagnostic mathematics implemented; no public official score.** Validate real decoder semantics and pinned host roles/reductions. The pass-through refiner still performs no real motion correction. |
| R5: every object and frame, including invisibility | The diagnostic requires an explicit expected episode/object/frame manifest and does not drop invisible frames. Pipeline `Motion` requires finite transforms but fake motion is stationary; export still uses metadata counts and labels every prediction visible. | **Diagnostic coverage guards implemented; real tracking absent.** Verify decoded-video coverage, actual required objects, continuity and reacquisition. A held stationary estimate or caller-supplied manifest does not prove motion fidelity or full challenge coverage. |
| R6: Track 1-only assets and estimates, including cameras | Runner/backend/download guards reject prohibited entry points. Diagnostic inputs require provenance declarations and reject known prohibited paths/tokens before payload or decoder loading. The fake camera still uses a fixed FOV. | **Partial guards only.** Declarations are not provenance proof. Estimate cameras/meshes from permitted Track 1 inputs and record content hashes and derivation edges. Reject unknown provenance; do not open prohibited assets to compare them. |

Current code anchors: [reference diagnostic](../src/v2hoi/score.py), [contracts](../src/v2hoi/contracts.py), [export](../src/v2hoi/stages/export.py), [metrics](../src/v2hoi/metrics.py), [inputs](../src/v2hoi/stages/inputs.py), [motion](../src/v2hoi/stages/motion.py), [stage registration](../src/v2hoi/stages/__init__.py), [runner](../src/v2hoi/run.py), [clip loading](../src/v2hoi/clips.py). The historical reference scorer and prohibited benchmark tests are replaced; their defaults and statistics must not be recovered for validation.

## Consequences for the design

One constant global similarity transform cannot repair human/object scale mismatch or later drift. If both streams are transformed together by a single constant scene change, their relative geometry is preserved; scaling the object independently is not a harmless coordinate choice. Keep native MHR parameter units separate from the metric coordinates obtained after model conversion.

For object vertices `V`, construct `P[t] = s_obj * R[t] * V + t[t]`. The evaluator applies the shared human-derived similarity to `P[t]`. A good-looking canonical mesh with the wrong size, position or rotation still fails this requirement. Freeze its origin and scale before tracking, and verify the compiled submission mesh as well as the reconstruction mesh.

Let `D2(x)[t] = x[t+1] - 2*x[t] + x[t-1]`. The acceleration objective concerns `norm(D2(pred) - D2(ref))`, not `norm(D2(pred))`. A constant prediction has zero self-acceleration but nonzero error whenever the real motion accelerates. Repairs must preserve observed event timing and amplitude; frozen tracks, lagged motion and detached hands are explicit rejection controls.

An invisible frame remains an output frame. Carry pose hypotheses and uncertainty internally, use visible evidence before and after the interval, and choose a complete trajectory for export. Do not delete low-confidence frames, mark invisible input as observed, or replace every missing interval by a static pose and count that as quality validation.

The alignment frame is settled by R1: **the first reference frame, identified in original-video coordinates**. Scored windows are separate and cannot reset it. External-kit indexing must be adapted and verified; it is not another permissible alignment policy.

## Ordered acceptance work

1. **G0 — representation, evaluation and provenance:** pin the official code/sample identities, implement native MHR conversion and official packing/read-back, define shared frame/scale/index contracts and artifact provenance. Test with independent synthetic transforms, known second differences, wrong scale, wrong axis, later drift and missing-frame controls. Do not use Track 2 or unavailable references.
2. **G1 — real baseline:** register and run the real approved reconstruction path on Track 1 episodes 16 and 12; preserve immutable inputs, outputs, settings and failure evidence. A default fake-stage invocation must never be described as this baseline.
3. **G2/G3 — reliable observations and bounded repair:** verify camera/scale evidence, posed geometry, contact and complete hidden trajectories. Accept a repair only when independent video evidence improves without harming timing, amplitude or coverage. Preserve the baseline for rollback.
4. **G4 — complete submission rehearsal:** verify the full expected episode/object/frame manifest, freeze one candidate, compile and reload that exact submission, and render/review its outputs. Link its generating commit and provenance; keep official receipts and scores separate from local proxies.

Owners, acceptance conditions and rollback details are in [implementation-plan.md](implementation-plan.md). No final compliance gate is marked passed by this audit.

## Verification performed for this change

- Reviewed the active requirements, reconstruction interfaces, replacement diagnostic and pinned external code evidence.
- Diagnostic tests use independently constructed synthetic native-parameter records, injected synthetic decoders and analytic geometry. The authoritative test list is `.github/workflows/tests.yml`; the current run result is recorded in the delivery report, not inferred from an earlier test count.
- The complete explicit CPU test allowlist passed **231 tests** on 2026-10-03, including **37 replacement-scorer tests**. A real `python -m v2hoi.score` subprocess decoded an independent synthetic fixture and sampled a generated GLB, then produced the five analytically expected diagnostics. This did not execute an actual MHR model or access challenge reference data.
- Scoring behavior changed: historical reference defaults, prohibited normalization and alternative alignment modes are removed. Reconstruction backends, native export and real model execution remain separate incomplete work.

Synthetic regression evidence does not close real reconstruction, native-export, provenance-truth or official-equivalence gaps. No Track 2 data or derivatives, real reconstruction model run, server deployment or competition submission is part of this diagnostic validation.
