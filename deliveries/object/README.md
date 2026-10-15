# M3 Object deliveries

Submit a small delivery record at **`deliveries/object/<unique-delivery-id>/README.md`** using the [shared template](../TEMPLATE.md), and open a PR to `main`. This folder currently contains instructions only, not completed delivery records. Any member may contribute.

Code starts in [objects.py](../../src/v2hoi/stages/objects.py) for assets and [motion.py](../../src/v2hoi/stages/motion.py) for tracking. Shared contracts are in [contracts.py](../../src/v2hoi/contracts.py) and the [v2 foundation](../../src/v2hoi/v2/). Current registered reconstruction backends are fake; proposed native 05 requires its own typed adapter and checks.

Identify whether the delivery is an unscaled candidate, a metric asset or a tracking result. A metric asset needs explicit scale evidence, canonical axes/origin and a complete mesh identity. Tracking binds the exact mesh, camera, depth, human/prepared inputs and original frames; cover occlusions and report confidence, ambiguity and failures. Keep episode-specific permitted source evidence explicit rather than assuming that matching object names authorize pooling clips.

Review scopes include human preparation's metric asset inputs, physics' trajectory inputs and platform provenance/integration. Record participants per result. Store small checks/reviews beside the delivery note; store meshes and trajectories in a new result-specific Release or agreed team store with exact identities and hashes. Follow the [submission and receiver guide](../README.md).
