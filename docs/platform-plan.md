# Module 1 platform scope and next deliveries

Updated 2026-10-15 for the team discussion. This is the proposed platform scope and implementation order; this update records the plan without implementing adapters or running models. The next engineering target is one complete real video whose inputs, native reconstruction, normalized scene and reloaded export can be traced and checked. Members may take tasks across all modules; no permanent module assignee is required.

## Current progress

| Area | Available evidence | Remaining boundary |
| --- | --- | --- |
| Internal delivery gates | Commit `8fabd16`: six v1 stages, fresh attempt outputs, input snapshots, typed reload checks, hashes, stale-result rejection and read-only recheck CLI | Reconstruction backends are fake; native formats and real reconstruction quality are not accepted |
| Local handoff demonstration | Five separate module folders on one host pass six internal stages; changing upstream motion blocks the consumer before its backend starts | This is independent synthetic material, not a two-host model run |
| Structural foundation | Immutable artifacts, v2 structural records and geometry helpers | Real producers, native decoding/conversion and complete scene import remain pending |
| Collaboration | Commit `ad17316`: floating roles, stable module boundaries, separate task branches/run directories and consumer review scopes | Task notes are coordination, not a distributed scheduler or lock |
| Verification | [Recorded local CPU run](evidence/delivery-gates-validation-20261015.json): 453 tests passed; [GitHub CI for `8fabd16`](https://github.com/Seanwilliam2077/video-to-hoi/actions/runs/38698342928) passed | Existing test evidence is not new evidence from this documentation update or real-model acceptance |

Historical experiments and downloaded model receipts remain useful audit inputs. They do not establish an accepted baseline in the current platform. See [component-audit.md](component-audit.md) and [delivery-gates.md](delivery-gates.md).

## Platform responsibilities

The platform makes independent model execution reproducible and connects the resulting artifacts into a coherent scene. Its boundaries remain stable as teammates change tasks or machines.

| Capability | Required output | Acceptance question |
| --- | --- | --- |
| Video input and observations | Video identity, decoded original-frame/timestamp map, target IDs, image transforms, masks and explicit validity states | Do all consumers refer to the same source frames and targets? |
| Data contracts and conversion | Versioned field meanings, units, coordinate conventions, parsers and validators | Can a consumer interpret a producer's result without implicit assumptions? |
| Model execution adapters | Stage command, effective configuration, environment requirements, actual input consumption and logs | Can a compatible worker reproduce a task without the original developer present? |
| Storage and handoff | Immutable result inventory, hashes, dependency packages and receiver verification | Does another machine have the exact complete inputs needed for the next task? |
| Dependencies and recovery | Readiness, failed/running/stale state, retained attempts and explicit rerun scope | Can execution resume from a verified boundary without reusing invalid descendants? |
| Scene integration and export | One complete scene revision, scene-bound serialization and reload/decode checks | Are human, objects, camera, scale and frame identity consistent after transfer and export? |
| Experiments and diagnostics | Frozen baseline/observation identities, separate candidate results, comparisons and review records | Can an observed improvement or regression be attributed to the declared change? |

Keep the platform's coordination and validation environment light. Heavy model runtimes stay behind explicit adapters in their own compatible environments. The internal fake path remains a useful CPU fixture; real native records must not be forced into incompatible v1 field meanings.

## Proposed task and result records

The default unit is **one complete episode, one stage, one attempt**. Independent complete episodes and alternative attempts can run in parallel. Native temporal stages must preserve the complete sequence; dividing one clip by time is not an established execution strategy.

A task declares its stage/adapter, episode and targets, exact input identities, source/model/configuration versions, runtime requirements, expected outputs and checks. A result records the actual consumed inputs, output inventory, execution environment, actual participants, logs, costs, check reports and unresolved findings. Changing participants does not change the contract or permit input substitution.

Record task activity and handoffs in a small task note or PR. Use distinct branches/worktrees and result directories for concurrent work. Preserve competing candidates and explicitly select a reviewed result rather than accepting the last file written. Automated claiming and cross-machine scheduling remain future work; the current run lock is local.

## Four integration boundaries

### Video to model inputs

Decode and preserve original frame IDs and timestamps, image dimensions, crop/resize transforms and stable target IDs. Map any sampled model frames back to that timeline. Distinguish visible, occluded and unavailable observations: an empty mask must not silently mean all three. Task descriptions and camera labels are not measured event times or camera calibration.

### Native outputs to a coherent scene

Preserve original native artifacts and use trusted, pinned parsers and decoder/converter identities. Bind native human parameters, object meshes, canonical origins, exactly-once scale, full-frame poses and the scene camera. Report normalization/mapping error separately from any fitting error.

The native preparation stages can recompute camera, depth and human estimates. Record what they actually consumed and produced. Bootstrap estimates and formal native outputs have separate lineage; compare their consistency and identify which values govern the final scene. Never assemble unrelated latest files into a shared result.

Human/depth bootstrap and an unscaled object candidate can be prepared in parallel. The metric object asset must then be available before formal native 01-04; native 05 and native 06-07 follow with their exact predecessor products. Flexible participation does not change this scientific dependency graph.

### One machine to another

Git carries source, configuration and small immutable result references; permitted large artifacts use the agreed storage. Separate the producer result, package transport manifest and receiver receipt. Check bytes, dependency completeness and stage-specific content; record local path relocation without overwriting producer evidence. Local native cache markers do not prove execution on the receiving machine.

Implement the first actual boundary around native 01-04 to 05, followed by 05 to 06-07. If machines cannot yet exchange a verified stage package, adjacent stages may temporarily execute on one compatible worker while the interfaces are completed.

### Scene to export and reload

Normalize native human results before scene publication. Export serializes that fixed scene without an independent refit or unrecorded geometry change. Reload, decode and compare the human/object geometry, frame coverage, mesh origin and scale. A preview or valid-looking array shape alone does not establish native conversion or submission compliance. Follow [R1-R6](track1-requirements.md) and the separate [G0-G4 gates](implementation-plan.md).

## Three distinct checks

| Check | Conclusion it may support | Current scope |
| --- | --- | --- |
| Delivery and dependencies | Required files are complete, readable, version-compatible and current | Implemented for the internal v1 profile; native validators pending |
| Scene and conversion consistency | Frames, coordinate conventions, scale and geometry remain coherent across representations | Synthetic helpers exist; real decoder/conversion and export evidence pending |
| Reconstruction quality | Declared video evidence supports a candidate without unacceptable regressions | Diagnosis and review protocol proposed; real baseline characterization and thresholds pending |

Delivery readiness may permit the next computation. Quality acceptance and final export acceptance require their own evidence. A model returning successfully or a low fitting loss cannot set those conclusions automatically.

## Proposed implementation order

These rounds refine P1/P2 and subsequent experiment support; they do not close existing gates or change agreed milestone dates.

| Round | Work | Completion evidence |
| --- | --- | --- |
| 1: real input to reloaded export | Verify one compatible runtime and permitted real sample; complete input mapping, native parsing, scene normalization, export and reload. Start with episode 16, then episode 12 as the existing engineering cases | Full original-frame coverage, exact input/mesh binding, conversion and geometry comparisons, source overlays, failure inventory, logs and runtime costs; execution/quality/export decisions recorded separately |
| 2: actual independent workers | Wrap independent stage commands; implement minimal pack/verify/import and local path adaptation. Transfer a complete episode between at least two actual hosts | Both machines' exact predecessor records, receiver checks, downstream execution evidence and final reload; missing/corrupt/wrong-version packages block continuation |
| 3: experiments and broader coverage | Add fixed-baseline candidate invocation, declared mutable variables, diagnostic comparisons and grouped reporting; expand input coverage and full-run planning | Reproducible baseline/candidate comparisons, visible regressions and failed attempts, explicit review/rollback, measured resource budget and complete coverage records |

Native adapter fixtures, transport prototypes and synthetic research cases may be developed in parallel. Real acceptance follows the dependency order above. Observe elapsed time, peak memory, disk use and package size from the first real run; use those measurements to budget full-set execution and reruns. Do not invent machine capability from a GPU model name alone.

## Innovation support

The platform should expose a fixed baseline and observation revision to a bounded experiment, record the permitted variable changes and compute budget, and preserve its output as a separate candidate. Review fitting and reserved observations separately, inspect full-clip regressions, and keep the baseline if evidence is insufficient. Platform automation does not establish a research contribution by itself.

M4 prototypes can use independent synthetic cases before real integration. Real comparisons require a frozen execution-valid baseline with known limitations and actual image evidence. Current local diagnostics do not supply hidden Track 1 ground truth or an official accuracy score; no clean generalization holdout has been established.

## Team discussion checklist

1. Agree on the first concrete real sample and its complete expected output inventory, provenance and environment record.
2. Identify the access and machine capability available for the next task, without assigning permanent module owners.
3. Agree on the first native handoff boundary and storage location; keep task inputs and package identities fixed.
4. Confirm the three acceptance layers and the early export/reload requirement before expanding model runs.
5. Select the first small tasks that members can start or take over, recording completion conditions and open questions. Review the existing at-risk schedule using measured runtime evidence; no new deadline is committed by this plan.

The first platform release is complete when another teammate can receive a real task, run it without its author present, verify and hand off its output, and trace the final reloaded scene back to the original video.
