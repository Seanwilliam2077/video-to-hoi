# M4 Temporal and physics deliveries

Submit a small delivery record at **`deliveries/physics/<unique-delivery-id>/README.md`** using the [shared template](../TEMPLATE.md), and open a PR to `main`. This folder currently contains instructions only, not completed delivery records. Any member may contribute.

Code starts in [refine.py](../../src/v2hoi/stages/refine.py); shared scene/artifact contracts are in the [v2 foundation](../../src/v2hoi/v2/). Add algorithm, diagnostic, configuration and test changes in their source locations. The current registered refine backend passes through its inputs. Native 06-07 and research adapters remain separate pending integrations.

Distinguish native joint inference/post-optimization from a new research candidate. Bind the exact human/object baseline, observations, mesh, scale and full frame timeline. State permitted changes and frozen quantities; deliver coherent human/object results, comparison evidence, failure intervals and regressions. Synthetic cases can establish algorithm behavior but cannot establish improved real reconstruction quality. Keep candidate outputs separate from the frozen baseline.

Review scopes include human and object consistency and platform scene/export integration. Record participants per result. Store small checks/reviews beside the delivery note; store native arrays, meshes and large previews in a new result-specific Release or agreed team store with exact identities and hashes. Follow the [submission and receiver guide](../README.md).
