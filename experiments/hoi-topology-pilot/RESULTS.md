# Contact-structure pilot: completed results

Date: 2026-10-09. Server: a6000dove. Remote directory: `/data1/dove/zly/hoi-topology-pilot`.

## Outcome

The fixed-grasp mechanism works in the independently authored synthetic fixture. The real-video pilot does **not** establish a successful reconstruction: the fixed-contact formulation reduces the assumed attachment residual and sampled penetration depth, but worsens image consistency and leaves penetration in every frame. This result should not be treated as evidence that topology alone solves intersection or that the method is ready for a larger training run.

## Real video: Track 1 episode 16, original frames 90–209

All methods use the same 120 frames, camera, cuboid scale, initializer and observation tracks. The primary independent control uses the same spline representation, optimization variables, priors and iteration budget as the coupled methods. The original Gaussian-smoothed reference is also retained in the video.

| Diagnostic (lower is better) | Original Gaussian | Matched independent | Soft joint | Fixed attachment + collision |
|---|---:|---:|---:|---:|
| Tracked-corner mean reprojection, px | 7.226 | 7.279 | 12.110 | 16.126 |
| Sampled maximum penetration, mm | 87.364 | 87.364 | 26.022 | 17.860 |
| Mean frame maximum penetration, mm | 68.007 | 69.108 | 8.653 | 9.582 |
| Frames with sampled penetration >0.5 mm | 111/120 | 112/120 | 118/120 | 120/120 |
| Mean distance to contact surface, mm | 32.828 | 28.053 | 35.080 | 32.332 |
| Assumed fixed-attachment residual, mm | 93.997 | 86.194 | 119.624 | 41.607 |
| Prediction-only object second-difference magnitude, mm | 4.420 | 1.587 | 3.350 | 0.675 |

The smaller second-difference magnitude cannot establish more faithful motion without reference motion. The corner tracks are fitting observations, not independent annotations. Penetration is sampled on the estimated human mesh against a simplified cuboid; it is neither an exact continuous intersection test nor the official PEN metric. Reference-relative ACC is unavailable.

The final native MHR parameters reproduce the optimized human geometry within 0.001 mm. This verifies representation consistency, not correctness of the human reconstruction. The two coupled methods move human vertices approximately 32–34 mm on average relative to initialization.

## Follow-up: whole-reconstruction smoothness review

The original temporal number measured only object-center translation. It cannot establish smoothness of rotation, hand deformation or hand-object relative motion. Saved outputs were reviewed without rerunning or changing optimization.

| Prediction-only mean temporal diagnostic | Matched independent | Fixed attachment + collision |
|---|---:|---:|
| Object translation second difference, mm/frame^2 | 1.587 | 0.675 |
| Change of world-frame rotation increment, degrees/frame^2 | 0.165 | 0.216 |
| Eight selected hand material vertices second difference RMS, mm/frame^2 | 3.249 | 3.274 |
| Hand material motion in object coordinates RMS, mm/frame | 10.678 | 4.316 |

Object translation changes and relative hand-object motion are smaller, while selected hand motion is nearly unchanged and rotation-step changes are larger. These measurements do not demonstrate an overall perceptually smoother or more faithful reconstruction. Relative material motion may include legitimate sliding; eight assumed contact vertices do not cover all body or hand deformation. Natural acceleration can produce large second differences without reconstruction jitter.

`smoothness-closeup-1x.mp4` and `smoothness-closeup-half-speed.mp4` provide synchronized, enlarged original-view and 3D comparisons at original speed and half speed. The 3D camera and scale are fixed and shared. `smoothness-curves.png` plots the temporal diagnostics, and `smoothness-review.json` records their definitions and distributions. Both videos were decoded and verified to contain 120 frames at 30 fps and 15 fps respectively.

## Synthetic fixture with known truth

| Method | Trajectory RMSE, mm | Contact closure RMS, mm | Maximum penetration, mm |
|---|---:|---:|---:|
| Independent | 5.257 | 7.267 | 20.592 |
| Soft joint | 3.300 | 0.852 | 2.645 |
| Shared structure | 3.242 | approximately 0 | approximately 0 |

The shared formulation has exact closure by construction with known, correct, fixed attachments. Its near-zero penetration is a property of this designed fixture, not a general collision guarantee. A separate sliding fixture gives 7.417 mm error under the wrong fixed binding, illustrating the assumption's failure mode.

## What the real result suggests

Likely limitations include uncertain monocular depth/scale, a rounded object represented as a cuboid, uncertain hand geometry and fixed material correspondences inferred in one frame. Their individual contributions have not been isolated. The next useful experiment should improve geometric and contact evidence and allow sliding or release where supported. Increasing constraint strength alone risks moving a plausible-looking reconstruction away from the observed video.

## Review artifacts

- `results/real/comparison.mp4`: original view, four columns in table order.
- `results/real/interaction_3d.mp4`: the same four methods from a novel viewpoint.
- `results/real/metrics.json`: unrounded diagnostics and complete original frame IDs.
- `results/synthetic/comparison.mp4` and `metrics.json`: separate synthetic experiment.
- `verification.json`: decoded video counts, finite diagnostics, native consistency and synthetic checks.
- `README.md`: assumptions, acquisition identities, reproduction and limitations.

The native arrays and full model assets remain in the isolated server experiment directory. Downloads used hf-mirror, gh-proxy and the Tsinghua Python package mirror. Track 2 was not used. No official challenge scores or submission files are produced.
