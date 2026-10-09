# Dynamic contact topology v4

This directory preserves the standalone Physics module pilot, including its
earlier initialization and controls. It is not connected to the production
pipeline or an official submission exporter. Only Track 1 episode 16, original
frames 90–209, is used. Independent synthetic fixtures have no challenge-data
provenance.

## Result

Read [the v4 report](reports/interaction-v4/REPORT.md) and
[the pre-run protocol](reports/interaction-v4/protocol.json). Dynamic contact
gates reduce average sampled penetration but increase object translation second
differences and motion range. The interaction mesh adds no clear independent
benefit. This is a mixed result, not demonstrated reconstruction improvement.
Second differences are prediction diagnostics, not reference-relative ACC.

The four conditions share the v3 no-graph warm start and receive the same
additional optimization budget. Static gates preserve total per-point contact
exposure, but not necessarily pairwise sticking exposure. Contact pseudo-labels
are frozen estimates, not independently verified touching/sliding annotations.
The recorded `near_m` setting is unused; `distance_scale_m` controls proximity.

## Environment and inputs

The completed run used Linux, Python 3.12.12, torch 2.9.0, torchvision 0.24.0,
numpy 2.2.6, scipy 1.16.3, opencv-python 4.12.0.88, matplotlib 3.10.7 and
roma 1.6.1 on one RTX 6000 Ada. Match the PyTorch build to the host CUDA driver.
The initializer also needs the pinned upstream SAM3D Body, DINOv3, MoGe and
utils3d requirements (including timm, hydra-core, omegaconf, einops and yacs).
Follow their installation instructions and retain upstream licenses. This is
an experiment source package, not a portable preconfigured runtime.

`hf-models-manifest.json` supplies pinned release identities and SHA-256 checks;
only its SAM3D Body entry is downloaded by `prepare_assets.py`. The two acquisition
receipts preserve input provenance. Acquisition uses hf-mirror.com and
gh-proxy.com; install Python packages from the Tsinghua mirror if needed.
No model files, source videos, credentials or external source trees are in Git.
Do not substitute Track 2 assets or reference trajectories.

## Run from this directory

Use a dedicated environment with the dependencies above. The example `python`
means that environment's interpreter. Model inference and real optimization
require CUDA. Each command depends on the preceding outputs.

```sh
python prepare_assets.py
python prepare_extra.py
CUDA_VISIBLE_DEVICES=0 python infer_body.py
python synthetic_trial.py
CUDA_VISIBLE_DEVICES=0 python real_trial.py
CUDA_VISIBLE_DEVICES=0 python interaction_trial.py --texture
CUDA_VISIBLE_DEVICES=0 python dynamic_trial.py
python render_dynamic.py
```

`interaction_trial.py --texture` produces the v3 warm start, texture split and
interaction structure consumed by v4. `render_dynamic.py` validates native
arrays, rotations, frame coverage and video decoding and writes verification
records. `verify_outputs.py` checks the earlier real/synthetic controls.
The acquisition scripts download several GB. Existing per-frame initialization
is reused, while trial commands overwrite their named result directories;
preserve old outputs before rerunning.

The original [README](README.md) and [RESULTS](RESULTS.md) describe the earlier
fixed-contact experiment. They do not supersede the v4 report.

## Stored records versus full artifacts

`reports/interaction-v4/` contains copied reports, protocol, JSON metrics,
optimizer diagnostics and the contact-mode timeline from the completed run.
These files can be reviewed without downloading models or running inference.
The full native states and comparison videos remain on the experiment server
under `/data1/dove/zly/hoi-topology-pilot/results/interaction-v4`.
On the owner's Windows machine, the read-only remote mount exposes them under
`E:\video-to-hoi\outputs\hoi-topology-pilot\interaction-v4`.
The mount and its credentials are machine-specific and are not part of this branch.

Next: establish independently inspected real contact transitions, correct the
hand/object geometry alignment, and repeat the same four-condition comparison.
