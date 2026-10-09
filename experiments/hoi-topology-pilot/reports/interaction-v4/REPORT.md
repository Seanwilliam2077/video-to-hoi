# Dynamic contact topology pilot v4

Track 1 episode 16, original frames 90-209. Diagnostic development experiment; no official score or 3D/contact reference truth.

Four matched conditions use one common v3 no-mesh warm start, the same native decoder, 16-knot corrections, observations, bounds, optimizer, 300-step budget, and sampled collision terms. Contact schedules are inferred once from warm-start estimated geometry and frozen during optimization. Static schedules use their temporal means, preserving total per-point contact exposure. All four receive an additional refinement budget, so differences from v3 cannot be attributed solely to the new mechanism.

| Diagnostic | Static, no mesh | Static, mesh | Dynamic, no mesh | Dynamic, mesh |
|---|---:|---:|---:|---:|
| Held-out texture error (px) | 4.201 | 4.200 | 4.220 | 4.223 |
| Mean frame max sampled penetration (mm) | 26.337 | 26.379 | 19.541 | 19.737 |
| Max sampled penetration (mm) | 55.942 | 54.658 | 60.772 | 59.795 |
| Object translation D2 (mm/frame^2) | 1.887 | 1.883 | 2.558 | 2.602 |
| Hand D2 RMS (mm/frame^2) | 3.981 | 3.982 | 3.975 | 3.966 |
| Object center motion range (mm) | 276.507 | 274.961 | 470.183 | 471.006 |
| Mean candidate surface distance (mm) | 25.795 | 25.621 | 45.004 | 45.247 |

## Interpretation

Dynamic mode switching reduces mean sampled penetration while worsening object translation second differences and increasing estimated object motion range. Maximum penetration also worsens relative to the matched static conditions. Selected hand smoothness changes very little. Held-out pixel consistency is similar, but it is reused development validation and does not establish correct 3D motion.

Adding the 3D interaction mesh to dynamic contact does not provide a clear independent gain: mean penetration, object D2 and held-out reprojection are slightly worse, despite a slightly smaller maximum penetration. This experiment fails to demonstrate the requested overall smooth reconstruction.

Contact-mode counts: {"separated": 566, "sliding": 355, "sticking": 22, "uncertain_geometry": 17}. These count candidate-point/frame labels, not video frames or verified contact events. The 566 separated labels may contain failed estimated contact, not true release. The few inspected source frames show the actor holding the block; they do not independently verify the inferred state timeline.

The synthetic check obtains 100% mode accuracy away from designed transition boundaries and removes the release constraint penalty. This only checks inference and constraint behavior on a simple known-truth fixture; it is not an optimized reconstruction benchmark.

The graph is a shared 3D geometric neighborhood Laplacian with mode-gated cross-object edges. It does not weld human and object materials, estimate friction forces, prove non-penetration, or reconstruct object surface topology. Geometry-based neighborhood changes alone do not add independent contact evidence.

## Next evidence needed

Validate at least one interval containing visually confirmed real contact change, using frozen source-side annotations of whether each hand is touching and whether sliding is visible. Contact timing labels must be separate from the held-out motion evaluation. Improve the hand/object geometry alignment and use contact observations to estimate the mode sequence, rather than treating a geometric mismatch as release. Repeat the same four-condition comparison before adding models or scaling training.

## Artifacts

- comparison-1x.mp4 and comparison-half-speed.mp4: four conditions, original camera overlay plus fixed 3D view/common scale.
- contact-modes.png: frozen geometric pseudo-label timeline.
- protocol.json: pre-run settings and selection rules.
- metrics.json: full diagnostics and state provenance.
- verification.json: finite arrays, proper rotations, original-frame coverage and video decoding.

The E-drive folder is a live remote mount. No bulk result download was performed.
