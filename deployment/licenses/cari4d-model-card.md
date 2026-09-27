---
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
---

# CARI4D CoCoNet Native MHR Overview

## Description

CARI4D CoCoNet Native Momentum Human Rig (MHR) refines one human's native MHR parameters and one rigid object's pose over video and predicts one contact logit per hand. It supports 4D human-object reconstruction, motion analysis, visualization, data curation, and robotics perception without direct autonomous actuation.

CARI4D CoCoNet Native MHR was developed by NVIDIA Research.

_This model is ready for commercial or non-commercial use._

### License/Terms of Use

GOVERNING DOWNLOAD TERMS: Use of the model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).

### Deployment Geography

Global.

### Use Case

For developers and researchers building commercial or noncommercial computer-vision systems for 3D/4D human-object pose estimation, motion analysis, visualization, data curation, and robotics perception. The model is not intended for direct autonomous actuation or life-critical decisions.

### Expected Release Date, Platform, and URL

| Field | Value |
|---|---|
| Expected Release Date | Hugging Face 08/30/2026 via https://huggingface.co/nvidia/cari4d_commercial |
| Release Platform | Hugging Face |
| Release URL | https://huggingface.co/nvidia/cari4d_commercial |

## Model Architecture

**Architecture Type:** Transformer.

**Network Architecture:** Dual observed/initialized branches use a frozen Vision Transformer (ViT)-B/14 red-green-blue (RGB) encoder from DINOv2 (self-distillation with no labels, version 2) and a DINOv2 ViT-S/14-initialized six-channel three-dimensional Cartesian (XYZ)/mask encoder that is fine-tuned for this model. Fused features pass through two spatial-temporal attention blocks and per-frame object-pose, native-MHR, and hand-contact heads.


**Number of Model Parameters:** 231M measured from the step-84,000 model state after excluding registered buffers.

## Input

**Input Type(s):** Video.

**Input Format(s):** MPEG-4 (MP4) red-green-blue (RGB) source video; Tensor after input materialization.

**Input Parameters:** Video - N-dimensional `[batch, 96, channels, 224, 224]`; geometry/conditioning - N-dimensional per-frame tensors.

**Other Properties Related to Input:** Each sample contains two float32 tensors shaped `[batch, 96, 9, 224, 224]`: an observed branch and an initialized-pose branch. Each branch contains RGB, normalized per-frame MHR-root-joint-centered camera-space XYZ, and human, visible-object, and full-object masks. Native MHR and rigid-object initializations provide additional conditioning.

## Output

**Output Type(s):** Tabular.

**Output Format:** Tensor.

**Output Parameters:** Tabular output - N-dimensional per-frame tensors for object pose, native-MHR parameter blocks, and hand contact.

**Other Properties Related to Output:** Per-frame outputs contain rigid-object rotation and translation residuals; native-MHR global rotation, translation, 260-dimensional body pose, 108-dimensional hand, and 45-dimensional shape residuals; and two hand-contact logits. Unsupervised output blocks are frozen by the checkpoint-bound supervision contract. Inference bundles serialize the tensor dictionary as Python pickle files.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration

**Runtime Engine(s):** PyTorch.

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA Ampere

**Validated Hardware:** NVIDIA A100.

**Supported Operating System(s):** Linux.

Integration into an AI system requires use-case-specific data and unit/system testing to verify technical, functional, safety, ethical, and legal requirements before deployment.

## Model Version

| Field | Value |
|---|---|
| Version | `2026-08-25-09-35-57`, step 200,000 |
| Training configuration | `learning/configs/mhr-daniel-commercial-moge2-behave79-val-fp16.yml` |
| Status | Training completed|

## Training, Testing, and Evaluation Datasets

### Training Dataset

| Field | Value |
|---|---|
| Link | [FORM-HOI](https://huggingface.co/datasets/nvidia/form-hoi) |
| Data Modality | Video, RGB, depth, masks, calibration, and 3D human/object annotations |
| Video Training Data Size | Less than 10,000 hours; 2,126 internal Daniel human-object interaction sequences with four camera streams |
| Data Collection Method | Hybrid: manually collected and automatic/sensors |
| Labeling Method | Hybrid: automatic/sensors, automated processing, and manually reviewed annotations |
| Properties | Count: 2,126 sequences and four camera streams per sequence. Modalities: RGB video, sensor/monocular depth, masks, calibration, and 3D annotations. Content: people performing object interactions, including personal data. Linguistic characteristics: Not Applicable. The internal Daniel corpus is not distributed by this repository. |

### Testing Dataset

| Field | Value |
|---|---|
| Data Collection Method | Not Applicable |
| Labeling Method | Not Applicable |
| Properties | Count: 0. Modalities, content nature, and linguistic characteristics: Not Applicable. No separate testing dataset is defined. |

### Evaluation Dataset

| Field | Value |
|---|---|
| Link | [Demo instructions](README.md#run-demo); [published CARI4D demo bundle](https://huggingface.co/nvidia/CARI4D/blob/main/cari4d-demo.zip) |
| Data Modality | RGB video and preprocessed inference inputs, including human/object masks, estimated human initialization, and reconstructed textured object meshes |
| Data Collection Method | Hybrid |
| Labeling Method | Not Applicable |
| Properties | Count: two demonstration inputs: one self-contained BEHAVE video and one in-the-wild gas-cylinder example whose source RGB video is obtained separately as documented in the README. The archive's masks, estimated human initialization, and reconstructed object meshes are inference inputs rather than ground-truth labels. No ground-truth human pose, object pose, contact, or quantitative evaluation labels are included; the separately published `behave-test-gt.zip` archive is not part of this evaluation dataset. This dataset is used for end-to-end pipeline execution and qualitative output inspection; no quantitative model-quality metric is computed from it. Linguistic characteristics: Not Applicable. |

## Inference

**Acceleration Engine:** PyTorch with CUDA.

**Hardware Requirements (GPU Architecture, Model):** NVIDIA Ampere A100 GPU validated; other targets require separate validation.

**Test Hardware:** NVIDIA Ampere A100 GPU.

## Ethical Considerations

NVIDIA believes Trustworthy AI is a shared responsibility and has established policies and practices to support development for a wide array of AI applications. Developers must verify that this model meets requirements for the relevant industry and use case and addresses foreseeable misuse.

Users must have proper rights and permissions for all input image and video content and must apply appropriate controls for personal data, health information, and intellectual property.

For more detail, see the Model Card++ [Bias](#bias-subcard), [Explainability](#explainability-subcard), [Privacy](#privacy-subcard), and [Safety & Security](#safety--security-subcard) subcards below. Report model quality, risk, security vulnerabilities, or NVIDIA AI concerns through the [NVIDIA security and AI concerns form](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

# Bias Subcard

| Field | Response |
|---|---|
| Participation considerations from adversely impacted groups and protected classes in model design and testing | No documented participation. Demographic and interaction coverage is uncharacterized, and the two published demos do not establish broad deployment generalization. |
| Measures taken to mitigate against unwanted bias | No demographic-specific mitigation is currently verified. |
| Bias Metric (If Measured) | Not measured. |

# Explainability Subcard

| Field | Response |
|---|---|
| Intended Task/Domain | Computer vision: 3D human-object pose and contact estimation. |
| Model Type | Spatiotemporal Transformer. |
| Intended Users | Developers and researchers building commercial or noncommercial human-object reconstruction, motion-analysis, visualization, data-curation, and robotics-perception systems. |
| Output | Tabular Tensor: per-frame rigid-object pose residuals, native-MHR parameter residuals, and two hand-contact logits. |
| Describe how the model works | The model compares observed RGB/XYZ/mask features with rendered initialization features, fuses them across space and time, and predicts per-frame corrections and contact logits. |
| Adversely impacted groups tested for comparable outcomes | Age, disability, ethnicity, gender, race, skin color, and body-morphology groups may be adversely impacted; comparable outcomes have not been established. |
| Technical Limitations and Mitigation | Performance can degrade with occlusion, poor masks or depth, inaccurate initialization, unseen objects/interactions, motion blur, poor lighting, or domain shift. Contact represents proximity, not force or physical feasibility. Use overlays and quantitative metrics to inspect outputs before downstream use. |
| Verified to meet prescribed NVIDIA quality standards | Yes |
| Performance Metrics | Validation loss; object translation error; symmetry-aware rotation geodesic error; native-MHR vertex and joint errors. |
| Potential Known Risks | Incorrect pose/contact estimates, privacy exposure from person videos or reconstructed motion, and unsafe downstream decisions if outputs are used without independent validation. |
| Terms of Use/Licensing | GOVERNING DOWNLOAD TERMS: Use of the model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). |

# Privacy Subcard

| Field | Response |
|---|---|
| Generatable or reverse engineerable personal data? | Yes; reconstructed body geometry and motion may relate to identifiable people, although the model does not synthesize source imagery. |
| Personal data used to create this model? | Yes |
| Personal data context | Training and evaluation videos depict people. |
| Categories of personal data used | Visual appearance, body geometry, pose/motion, and environment context associated with people. |
| Was consent obtained for personal data? | Yes |
| Are data-subject rights supported? | Yes |
| Was data collected by NVIDIA? | Yes; Daniel was collected internally and BEHAVE is externally sourced. |
| Is data minimized to what is required? | Yes; model training uses task-required cropped/masked RGB, depth, calibration, and annotations, and this repository does not distribute the internal videos. |
| Can data subjects access applicable disclosures? | Yes. |
| How often is the dataset reviewed? | Before Every Release. |
| Was data from user interactions with the AI model used to train the model? | No. |
| Is there provenance for all datasets used in training? | Yes; source identities and preprocessing manifests are tracked. |
| Does data labeling comply with privacy laws? | Yes |
| Can data-subject correction or deletion requests be applied? | Yes |
| Applicable Privacy Policy | [NVIDIA Privacy Policy](https://www.nvidia.com/en-us/about-nvidia/privacy-policy/) |

# Safety & Security Subcard

| Field | Response |
|---|---|
| Model Application Field(s) | Other - Computer Vision: 3D human-object pose estimation, motion analysis, visualization, data curation, and robotics perception. |
| Describe the life-critical impact (if present) | Not Applicable; the model is not intended for direct autonomous actuation or life-critical decisions. |
| Use Case Restrictions | GOVERNING DOWNLOAD TERMS: Use of the model is governed by the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/). |
| Model and dataset restrictions | Apply least-privilege access to person videos and annotations, honor dataset and third-party license restrictions, and independently validate outputs before deployment. |
