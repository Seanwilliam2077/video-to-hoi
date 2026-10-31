---
license: openmdw-1.1
library_name: egohand
---

# EgoHand v1.0 Overview

## Description:  
EgoHand predicts SOMA-hand parameters (pose, 27D MHR shape/scale coefficients) and a full-perspective camera translation (conditioned on input camera intrinsics) from a single image cropped from a 2D hand bounding box and resized to 256×256, using a DINOv3 vision backbone (initialized from [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3)) feeding a TransformerDecoder head (SOMAHandTransformerDecoderHeadV2) with learnable init pose/shape/camera queries. The predicted shape/scale coefficients parameterize MHR's rest-pose hand identity, which SOMA's rig then deforms via forward kinematics using the predicted pose to produce the final 3D hand mesh. It is trained on synthetic and real world hand data with keypoint-2D, keypoint-3D, and MHR/SOMA parameter losses computed in the crop frame, and is evaluated on the HOT3D real-world egocentric benchmark.  
EgoHand-1.0 was developed by NVIDIA as a part of EgoHand.  

_This model is ready for commercial or non-commercial use._

### License/Terms of Use:
Use of the associated model is governed by the [OpenMDW License Agreement, version 1.1](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1).  

ADDITIONAL INFORMATION: [Apache License, Version 2.0.](https://www.apache.org/licenses/LICENSE-2.0)

### Deployment Geography:  
Global

### Use Case:  
EgoHand could be used by researchers and engineers working on egocentric hand pose estimation to produce accurate 3D hand keypoints and MHR/SOMA-hand mesh parameters from single RGB crops. The resulting 3D hand pose/keypoint data serves as a training signal for downstream robotics applications — particularly robotic foundation models and manipulation policy training, where accurate hand trajectories human demonstration video are needed to supervise or bootstrap policy learning.  


## References:
[EgoHand inference pipeline](https://github.com/NVIDIA/IsaacCapture/tree/main/src/python/isaaccapture_egopost)  
[EgoHand model finetuning](https://github.com/NVlabs/GEM-X)  

## Model Architecture:   
**Architecture Type:** Transformer  
**Network Architecture:** DINOv3 (Vision Transformer backbone) + TransformerDecoder head  
**This model was developed based on DINOv3.**  
**Number of model parameters:** 883.5M  

 
## Input:  
**Input Type(s):**

* Image
* Other: Camera intrinsics

**Input Format(s):**

* Image: Red, Green, Blue (RGB)
* Camera intrinsics: Float

**Input Parameters:**

* Image: Two-Dimensional (2D)
* Camera intrinsics: Other: 3x3 matrix (focal length + principal point)

**Other Properties Related to Input:** Image cropped from 2D hand bounding box (squared, with padding) and resized to 256×256; camera intrinsics (focal length + principal point)  
  

## Output:  
**Output Type(s):**

* Other: SOMA parameters (hand pose and shape)
* Other: Camera pose

**Output Format:**

* SOMA parameters: Float
* Camera pose: Float

**Output Parameters:**

* SOMA parameters: Other: parameter vector (pose rotations + shape/scale coefficients)
* Camera pose: Three-Dimensional (3D)

**Other Properties Related to Output:** 3D hand joints and mesh  
   

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** PyTorch  
**Supported Hardware Microarchitecture Compatibility:** NVIDIA Ampere  
**Supported Operating System(s):** Linux  

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  


## Model Version(s): 
EgoHand v1.0 is a monocular model that takes a single RGB image as input. Users can post-train the model on their own data, and the resulting 3D hand joint outputs can be used to train robot foundation models.
 
### Training Dataset:

**Data Modality:**
* Image  
* Video  

**Image Training Data Size:** \~14M images (\~9M in-the-wild and MOCAP data; \~5M synthetic)  
**Video Training Data Size:** \~126 hours  
**Data count:** \~14M images  
**Data Collection Method by dataset:** Synthetic  
**Labeling Method by dataset:** Synthetic  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** The model is trained on a mix of synthetic and real-world hand motion image/video data, annotated with 3D hand keypoints and camera pose.  
* In-house purchased real-world egocentric hand data: RGB image/video of hand motion captured with head-mounted egocentric cameras, sourced from 3rd-party data vendors. 
* Synthetic data generated via the in-house RigPlay/DigitalHuman rendering pipeline: photorealistic renders of hand motion with exact ground-truth 3D keypoints, MHR/SOMA parameters, and camera pose.  
* In-house purchased motion capture data: real hand motion captured with a multi-camera, marker-based (reflective-marker) motion-capture rig for high-precision ground-truth 3D hand keypoints and pose. \~16 hours total across two capture formats: 1360×1360 at 30 FPS and 2464×2064 at 60 FPS.

### Testing Dataset:

**Data Collection Method by dataset:** Manually-Collected  
**Labeling Method by dataset:** Automatic/Sensors  
**Image Evaluation Data Size:** 100K Images  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* We tested on videos of hand movements and gestures captured in realistic scenes. Videos are collected by cameras or XR devices (e.g., Apple Vision Pro) and chunked into 30 secs long at 30 FPS. We only used the data to evaluate model qualitative performance.  
* EgoDex: 829 hours of 1080p, 30 FPS egocentric video (338K task-demonstration episodes, ~90M frames) across 194 diverse tabletop manipulation tasks, collected on Apple Vision Pro. Paired 3D hand/finger and upper-body joint pose annotations (25 joints per hand) and camera extrinsics are estimated on-device via ARKit/SLAM at recording time; we only used the RGB frames for qualitative evaluation.  


### Evaluation Dataset:
**Benchmark Score:** Undisclosed  

**Data Collection Method by dataset:** Manually-Collected  
**Labeling Method by dataset:** Automatic/Sensors  
**Image Evaluation Data Size:** 231K Images  
**Properties (Quantity, Dataset Descriptions, Sensor(s)):** 
* HOT3D: 9 subjects, 1.5K clips, 227K frames. Real-world egocentric dataset for 3D hand and object tracking, recorded with Meta Project Aria glasses (RGB + monochrome cameras) and Meta Quest 3 headsets (stereo monochrome cameras). Ground-truth 3D hand poses (UmeTrack) were captured with a marker-based motion-capture system and a custom 3D hand scanner per subject.
* H2O: 4 subjects, 184 clips, 114K frames. Real-world egocentric dataset of two hands manipulating objects, recorded indoors with synchronized multi-view RGB-D cameras including a head-mounted egocentric camera. Ground-truth 3D hand poses (with MANO fits for left/right hands) were derived via markerless multi-view triangulation.



## Inference:
**Acceleration Engine:** 
Other: None  

**Test Hardware:**
* NVIDIA Lovelace L40  
* NVIDIA Lovelace L40s


## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.

For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](egohand-subcards/bias.md), [Explainability](egohand-subcards/explainability.md), [Safety & Security](egohand-subcards/safety_security.md), and [Privacy](egohand-subcards/privacy.md) Subcards.  

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).