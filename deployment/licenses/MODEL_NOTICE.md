# Model redistribution notices

Reviewed: 2026-09-27.

The model files remain third-party works under their respective licenses. The
video-to-hoi repository's Apache-2.0 code license does not replace or relicense
these weights. Each independently distributed model package must include the
license and notice files listed for that asset in `model-redistribution.json`.
The selected model files are redistributed without model modifications; packaging
and transport splitting do not claim a newly trained or validated model.

## SAM 3D Body

- Provider: Meta Platforms, Inc. and affiliates.
- Source: https://huggingface.co/facebook/sam-3d-body-dinov3
- Model revision: `11aaa346c7204874a1cbafe3d39a979080b2c55a`.
- License: SAM License, last updated November 19, 2025; complete copy:
  `SAM_LICENSE.txt`.
- Pinned license:
  https://huggingface.co/facebook/sam-3d-body-dinov3/blob/11aaa346c7204874a1cbafe3d39a979080b2c55a/LICENSE

The included `assets/mhr_model.pt` is the Momentum Human Rig component. The
pinned Body model card links MHR and states the SAM License for the Body model;
it does not specify an additional MHR redistribution procedure. The official
MHR repository separately identifies MHR as Apache-2.0, so its full upstream
license is also preserved as `MHR_LICENSE.txt` for that component. This does not
change the SAM License applying to the Body model as distributed.

- MHR upstream source: https://github.com/facebookresearch/MHR
- MHR license reference revision: `d96fafa33bbf018647c70c3525e91f53e79d2a14` (license reference only;
  the bundled model bytes are identified by the pinned Body asset manifest).
- MHR license evidence:
  https://github.com/facebookresearch/MHR/blob/d96fafa33bbf018647c70c3525e91f53e79d2a14/LICENSE
- MHR model/README evidence:
  https://github.com/facebookresearch/MHR/blob/d96fafa33bbf018647c70c3525e91f53e79d2a14/README.md

## SAM 3D Objects

- Provider: Meta Platforms, Inc. and affiliates.
- Source: https://huggingface.co/facebook/sam-3d-objects
- Model revision: `2e73555018d2741ccd486e56c24fac41155a1dc6`.
- License: SAM License, last updated November 19, 2025; complete copy:
  `SAM_LICENSE.txt`.
- Pinned license:
  https://huggingface.co/facebook/sam-3d-objects/blob/2e73555018d2741ccd486e56c24fac41155a1dc6/LICENSE

For both SAM models, redistribution remains subject to the SAM License and
includes a copy of that agreement. Publications reporting research performed
using SAM Materials must acknowledge their use. The agreement's applicable
use restrictions remain in force; this notice does not replace the full terms.

## CARI4D CoCoNet Native MHR

Licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement.

- Provider: NVIDIA Corporation and affiliates.
- Source: https://huggingface.co/nvidia/cari4d_commercial
- Model revision: `1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9`.
- Selected model: `2026-08-25-09-35-57/step200000.pth`.
- Weight size: `2084484133` bytes.
- Weight SHA-256:
  `78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3`.
- License: NVIDIA Open Model Agreement, released April 2, 2026
  (document version March 9, 2026); complete original PDF:
  `NVIDIA_OPEN_MODEL_AGREEMENT_2026-04-02.pdf`.
- Official governing terms:
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
- Original PDF source:
  https://www.nvidia.com/content/dam/en-zz/Solutions/license-agreements/enterprise-services/nvidia-open-model-agreement-2026-04-02.pdf

The original model revision had no README or LICENSE file. The preserved
`cari4d-model-card.md` is the official model card from revision
`6e064f8b261599a4e563d72cb8457256929d7455`. It names the same Aug-25 step-200000 model and identifies
the governing NVIDIA Open Model Agreement. Hugging Face LFS metadata confirms
the same selected weight path, size and SHA-256 at both revisions. The model
weight revision has not been changed by including that later license evidence.

Model-card source:
https://huggingface.co/nvidia/cari4d_commercial/blob/6e064f8b261599a4e563d72cb8457256929d7455/README.md

The NVIDIA terms require a license copy with redistributed works and retention
of applicable notices. This attribution is retained with the model package.
Names are used to describe upstream origin; this project is not an official
NVIDIA or Meta release and does not claim their endorsement.

## Scope

This review covers the three model assets named above. Other models, source
repositories, dependencies and datasets retain their own licenses and are not
relicensed by this notice. No challenge data or model execution results are
included in this licensing review. Downloads and release packaging do not
establish deployment readiness or successful inference.
