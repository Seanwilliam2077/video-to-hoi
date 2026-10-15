# M2 Human deliveries

Submit a small delivery record at **`deliveries/human/<unique-delivery-id>/README.md`** using the [shared template](../TEMPLATE.md), and open a PR to `main`. This folder currently contains instructions only, not completed delivery records. Any member may contribute.

Code starts in [human.py](../../src/v2hoi/stages/human.py); shared contracts are in [contracts.py](../../src/v2hoi/contracts.py) and the [v2 foundation](../../src/v2hoi/v2/). Add related adapters, configuration and tests in their appropriate source locations. The current registered reconstruction backend is fake; the [native split](../../docs/distributed-execution.md) is a design awaiting adapters.

Distinguish an early human/depth scale bootstrap from complete native 01-04 preparation. Bootstrap can precede the final object mesh; formal native 01-04 must name the exact metric object asset and its other consumed inputs. Include full-frame mapping, camera/depth/scale conventions, human representation and decoder identity, prepared products and unresolved errors. Do not claim a bootstrap is the final human trajectory.

Review scopes include the object stage's scale/tracking inputs, the physics stage's human/scene inputs and platform provenance/integration. Record the actual executor and reviewers per result. Small checks/review records belong beside the delivery note; large native outputs belong in a new result-specific Release or agreed team store, referenced by exact identities and hashes. Follow the [submission and receiver guide](../README.md).
