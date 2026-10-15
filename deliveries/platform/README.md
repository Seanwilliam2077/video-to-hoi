# M1 Platform deliveries

Submit a small delivery record at **`deliveries/platform/<unique-delivery-id>/README.md`** using the [shared template](../TEMPLATE.md), and open a PR to `main`. Any member may contribute.

Current delivery: [all 30 videos ready for standalone Body input](body-inputs-track1-20261016/README.md). All 16,563 source frames have target-human masks, estimated intrinsics and reviewed inputs. The complete gate, native readers and same-host relocated archive check passed; Body inference and second-host execution remain separate. The [earlier raw preparation record](raw-track1-20261015/README.md) is retained as history.

| Work | Existing code or plan location |
| --- | --- |
| Input preparation and observations | [inputs.py](../../src/v2hoi/stages/inputs.py) |
| Scene serialization | [export.py](../../src/v2hoi/stages/export.py) |
| Contracts, execution and delivery checks | [contracts.py](../../src/v2hoi/contracts.py), [run.py](../../src/v2hoi/run.py), [check.py](../../src/v2hoi/check.py), [gates.py](../../src/v2hoi/gates.py), [deliverables.py](../../src/v2hoi/deliverables.py) |
| Immutable scene/artifact foundation | [v2/](../../src/v2hoi/v2/) |
| Packaging and integration utilities | [tools/](../../tools/) |
| Planned platform scope | [platform-plan.md](../../docs/platform-plan.md) |

For observations, include original frame/timestamp mapping, source and target identity, image transforms, camera conventions and validity information. For integration/export work, identify the exact coherent scene and provide conversion/reload evidence and limitations. Typed native adapters, native transport and official export acceptance remain pending.

Ask reviewers covering the human/object consumers to check relevant conventions and scene consistency. Record participants per task; no permanent owner is required. Code remains in its source directories. Small reports may go beside the delivery record; large outputs go to a new result-specific Release or agreed team store, with exact identities and hashes. Follow the [submission and receiver guide](../README.md).
