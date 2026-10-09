# Hand-object contact-structure pilot

For the latest dynamic contact v4 experiment, start with
[REPRODUCE.md](REPRODUCE.md) and its linked report. The sections below retain
the earlier fixed-contact experiment's protocol and limitations.

Experiment date: 2026-10-09 (Asia/Shanghai).

Remote root: `/data1/dove/zly/hoi-topology-pilot` on `a6000dove`.
One RTX 6000 Ada GPU is selected with `CUDA_VISIBLE_DEVICES=0`.

## Purpose and scope

Test whether fixed hand-object contact correspondences and jointly optimized
continuous scene motion improve a small reconstruction. This is a diagnostic
prototype. It does not implement a topology-changing reconstruction network,
globally injective deformation, IPC, or official challenge export.

Only Track 1 episode 16 is used: original frames 90–209, 120 frames at 30 fps.
The source video, masks/cameras from other tracks, and reference trajectories
from Track 2 are never used. The synthetic fixture is authored independently.

## Input and initialization

`prepare_assets.py` restores SAM3D Body from the owner's pinned public release,
checking both part and assembled archive SHA-256 against the existing manifest.
It retrieves the pinned Track 1 original through `hf-mirror.com` and sources
through `gh-proxy.com`. Python dependencies are installed from the Tsinghua
mirror. `assets-receipt.json` and `extra-assets-receipt.json` record identities.

The official SAM3D Body initializer retains native MHR parameters, identity
coefficients, face coefficients, camera translation, and decoded geometry.
The explicit camera-axis conversion is validated by independently decoding the
native parameters through the official TorchScript MHR model. Intrinsics are
estimated with pinned MoGe2 from frames 90, 150 and 209 and fixed for the clip.
A manually inspected actor bounding box excludes background people.

The object is a simplified cuboid fitted to six manually inspected image
corners in frame 150 and their optical-flow tracks. Its ratio is 1:0.85:1.
Scale is estimated from the native-human hand depth; it is not known true scale.
The rounded/textured foam object is not exactly a cuboid. The fit and tracking
details are in `results/real/object-fit.json`.

## Comparisons

- `independent`: Gaussian smoothing, retained as the original reference diagnostic.
- `matched_independent`: independent smoothing/fitting with the same 16-knot
  natural cubic-spline representation, native controls, image/human priors,
  optimizer and iteration budget as the coupled methods; contact and collision
  weights are zero. This is the primary control for temporal representation.
- `soft_joint`: joint native-human/object optimization with surface-distance
  contact and sampled penetration penalties.
- `attachment_collision`: joint optimization with eight fixed material
  correspondences in the object's coordinate frame and the same penetration
  penalties. This tests a no-slip contact-structure hypothesis.

The object shape/scale, camera, initializer, source frame IDs and observation
tracks remain common. Native shape/scale coefficients are retained. A measured
native-parameter translation conversion bounds root changes to approximately
30 mm per coordinate before spline/parameter bounds; local arm/hand controls
are chosen by their measured contact-versus-body influence.

The solver directly differentiates the original MHR decoder. A linearized
deformation attempt was rejected because final decoding disagreed with its
approximation; its outputs are preserved in `results/rejected-linearization`.
Other development diagnostics are kept separately from the final `results/real`.

## Measurements and limitations

The synthetic fixture has known motion and fixed grasp attachments. Its zero
contact error is a consequence of exact shared representation, not evidence
of recovering real contact from pixels. A separate slip control demonstrates
failure when the fixed-binding assumption is wrong.

Real-video measurements are diagnostics on estimated geometry:

- Image-corner reprojection error measures consistency with the tracks used in
  fitting; it is not an independent test annotation or a 3D accuracy score.
- Fixed-attachment residual measures the assumed material correspondences.
  Those pairs are inferred at frame 150 and are not known 3D contact truth.
- Penetration is sampled at all human mesh vertices, unique edge midpoints and
  face centroids against the fitted cuboid. It is not a guaranteed continuous
  triangle-intersection test, not a volume estimate, and not official PEN.
- Object second-difference magnitude measures prediction-only smoothness.
  It is not reference-relative ACC and cannot establish faithful acceleration.
- Human displacement and decode consistency prevent reporting an inconsistent
  optimized mesh as a valid native-parameter result.

Reducing penetration or second differences at the cost of image consistency
does not establish successful reconstruction. A positive synthetic result and
a negative or mixed real result must both be retained.

## Outputs

`results/real/comparison.mp4`: four columns in the method order above, with
original-video background, green cuboid and magenta inside-surface samples.

`results/real/interaction_3d.mp4`: the same comparison from a novel viewpoint.
This is a wireframe diagnostic, not photorealistic or occlusion-correct rendering.

`results/real/metrics.json`: all measurements, assumptions and original frame IDs.
Each method's NPZ retains native MHR state and the corresponding scene geometry.
These are experimental records, not submission files.

`results/synthetic/comparison.mp4` and `metrics.json`: separate known-truth fixture.

## Reproduce on this server

The experiment venv inherits PyTorch from `/data1/dove/anaconda3/envs/rag` and
installs additional packages only inside this experiment. `environment.txt`
records versions. This environment is not a portable installation package.

From the remote experiment root:

```sh
CUDA_VISIBLE_DEVICES=0 env/bin/python infer_body.py
env/bin/python synthetic_trial.py
CUDA_VISIBLE_DEVICES=0 env/bin/python real_trial.py
```

The model/source acquisition scripts are restartable for completed downloads.
Inference keeps completed per-frame results. Rerunning the trials overwrites
their final result directories; copy prior results first when comparing changes.

Model and source licenses are retained in the assets/source trees. No private
tokens, SSH keys or other credentials are included in experiment outputs.
