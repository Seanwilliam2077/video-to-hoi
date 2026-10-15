# Independent module work and checked handoffs

**Updated 2026-10-15.** Member roles float across tasks. M1-M4 name capability and input/output boundaries, not permanent people: any member may develop, run or review work in any module. A missing permanent assignee does not block progress. The [module profile](../configs/team-modules.json) records boundaries and review scopes only; it is not a scheduler or an access-control policy.

The current executable path has six internal v1 stages with fake backends. [Delivery gates](delivery-gates.md) check those outputs and their dependencies before continuation. Passing them proves structural readiness for that internal path, not successful reconstruction, native MHR conversion, or challenge submission compliance. [Native stage boundaries](distributed-execution.md) remain a separate design awaiting adapters and schemas.

## Work that can start independently

| Module / branch area | Start now without another module's GPU run | Integration input that must eventually arrive | Required output |
| --- | --- | --- | --- |
| M1 Platform / `platform` | Contract fixtures, automatic checks, input preparation, segmentation adapter, transfer verification and export reload tooling | Exact model-stage outputs for real integration | Versioned observations; delivery reports; coherent final package and reload evidence |
| M2 Human / `human` | Human/depth adapter and standalone scale bootstrap from a permitted input revision; test schema and full-frame indexing on independent synthetic fixtures | M1 observations; metric object asset before the current native 01-04 block | Human/bootstrap evidence, then complete native 01-04 prepared products |
| M3 Object / `object` | Unscaled object candidates, origin/axis conventions, mesh and tracker interfaces; independent synthetic geometry tests | M2 bootstrap for metric scale; exact native 01-04 outputs for FoundationPose | Metric asset and native 05 trajectory, complete through occlusions |
| M4 Physics / `physics` | Synthetic event/visibility/contact cases, bounded repair implementation, failure diagnostics and ablation harness | Fixed accepted human/object baseline and observations for real improvement claims | Native 06-07 results and separately identified research candidates |

M2's early bootstrap is preparation, not the final human trajectory. It can start before M3's final mesh. M3 can prepare an unscaled candidate in parallel; depth-conditioned generation explicitly declares and waits for its depth input. The current full native 01-04 path still requires the real metric asset. Do not fabricate a mesh or completion marker to bypass that dependency.

M4 can implement and test the innovation before a real baseline exists. Synthetic success establishes algorithm and interface behavior only. A claimed improvement needs the same frozen real baseline, observations, exported candidate and review protocol on both sides of the comparison.

The same teammate may prepare a human result today and work on object tracking tomorrow. Another teammate may run a GPU-dependent converter while the first reviews its output. Record the actual executor, environment and reviewer for each task/result; no permanent module owner or member-to-machine mapping is required. Review scopes in the table below describe the knowledge needed for that result, not a fixed roster. Preserve any required independent quality review when choosing reviewers.

## Task-scoped coordination

Use a small task note or pull request to identify the module/stage, episode, exact input revision, expected outputs, current activity and any handoff. Record who actually executed or reviewed the result and link its check report. A teammate can take over using those records and a compatible environment; no permanent assignment needs to change.

Concurrent changes use separate task branches/worktrees and unique attempt directories. A Git note is coordination, not an atomic distributed lock. If two members run alternatives, preserve both candidates and select an explicitly reviewed result; never overwrite each other's output or accept whichever file arrived last. The existing runner lock remains local to one run directory. Automated task claiming and cross-machine scheduling are not implemented.

## Producer and consumer agreement

| Producer handoff | Consumer review required before integration | Review responsibility |
| --- | --- | --- |
| M1 observations and masks | M2 and M3 | Same source video, original frames/timestamps, target IDs, image dimensions and coordinate conventions |
| M2 bootstrap and scale evidence | M3; M1 checks lineage | Explicit scale uncertainty, depth/camera convention and permitted source observations |
| M3 metric asset | M2 and M4; M1 checks identity | Scale applied once, canonical transform/origin, geometry and complete asset identity |
| M2 native 01-04 | M3 and M4 | Prepared export, initialized human, aligned depth and mesh belong to the same logical run |
| M3 native 05 / internal v1 motion | M4 | Exact mesh binding, full frame coverage, proper poses, confidence and unresolved tracking failures |
| M4 native 06-07 or research candidate | M2 reviews human; M3 reviews object; M1 integrates | Consistent coupled scene, unchanged frozen inputs, demonstrated effects and regressions |
| M1 serialized scene and reload | M2 and M3 | Reloaded human/object/mesh correspond to the exact selected candidate; format and scientific review remain distinct |

These roles define the review procedure. They are not enforced branch-protection rules or automatic quality approvals. A contract change needs its own reviewed change before producer and consumers migrate. Do not let one module silently reinterpret another module's units, axes, frame IDs or array meanings.

## One immutable experiment, one handoff

1. **Choose inputs.** Pin source episodes, upstream result identities and hashes, contract/profile, code revision, model/rig identities, configuration and environment. Use only Track 1 or independently authored synthetic fixtures; reject unknown provenance and all Track 2 derivatives.
2. **Create a new experiment ID.** Keep its input snapshot fixed. Run the selected stage on a compatible available worker. Repeating or changing an experiment creates a new attempt/result identity; accepted baseline packages are never overwritten.
3. **Check the producer's own delivery.** A fresh attempt must produce all its outputs itself. Missing files cannot be silently filled from an older upstream copy. A backend returning successfully is not sufficient.
4. **Publish a complete handoff.** Open a PR to `main` with a record at `deliveries/<module>/<unique-delivery-id>/README.md`, using the [submission guide and template](../deliveries/README.md). Share small manifests, result identities, hashes, report references and review records there. Put large permitted files in new result-specific Releases or agreed team storage and retain exact package identities. Do not commit `runs/`, model weights, masks or depth arrays to ordinary Git. Add the summary and delivery link to the corresponding live module page's Progress section separately.
5. **Verify on the consumer machine.** Obtain the exact required artifacts, verify their bytes and schemas, and resolve dependencies before running. A producer's passing report alone does not establish that the receiving machine has the same files. Automated native package import/relocation is still pending.
6. **Review and continue.** Structural passing permits only the implemented internal continuation. Consumer semantic review, real reconstruction quality, scene acceptance and final submission remain separate decisions. A failed or stale dependency stops the next stage until a new valid result or an explicitly selected fixed upstream is available.

The executable v1 runner can use `--upstream` for isolated experiments once that upstream's required delivery records are valid. Separate run directories keep experiments independent; they do not create automatic scientific independence. A fake upstream is useful for contract work but cannot establish real model quality.

Prefer one episode per experiment directory during team iteration. One multi-episode invocation is one delivery batch; replacing only part invalidates the original batch receipt. Use larger batches after freezing the configuration, or rerun the entire affected batch.

Changing observations, camera, scale, mesh, human state or motion can invalidate descendants. Recompute affected outputs against the new input identities. Copying an old report, renaming a folder, or updating a manifest to say “latest” does not repair stale results. Different modules must not submit independently optimized human and object arrays as though they were one jointly verified scene.

## Current check commands

Run these from the repository root, with the environment installed and Track 1 metadata available. The example assumes `runs/team-e16-v1` already exists; checking does not run a model or fetch data.

```bash
python -m v2hoi.check --run runs/team-e16-v1 --stage human --episodes 16 --dataset-root data/v2d/track_1
python -m v2hoi.check --run runs/team-e16-v1 --before-stage motion --episodes 16 --dataset-root data/v2d/track_1
python -m v2hoi.check --run runs/team-e16-v1 --status --episodes 16 --dataset-root data/v2d/track_1
```

The first command checks a stage's delivery; the second checks whether its next consumer has valid prerequisites; the third reports current readiness. Use the [delivery-gate guide](delivery-gates.md) for report semantics, exit codes and rerun behavior. The pipeline runner applies its gates automatically around stage execution; invoking a standalone check does not schedule the next model.

For CPU contract development, use independently authored fixtures and the explicit test allowlist in `.github/workflows/tests.yml`. No module needs to wait for a GPU merely to test its producer/consumer schema, failure behavior and fixed frame coverage. Real integration still requires valid real upstream artifacts and the necessary native adapters.

## Native gates that still need implementation

| Boundary | Required acceptance work before cross-machine continuation |
| --- | --- |
| Inputs/bootstrap/metric asset | Typed observation, camera, timeline, scale and mesh identities; bootstrap-to-native consistency and review evidence |
| Native 01-04 to 05 | Complete prepared export tree, human initialization, aligned depth and mesh; original frame mapping; exact predecessor bindings; relocation and cache policy |
| Native 05 to 06-07 | Complete tracking, exact mesh/camera/depth/human binding, proper transforms and confidence/validity diagnostics; adapter-observed input closure |
| Native 06-07 to full scene | Trusted pinned bundle decoder, coupled human/object fields, all-frame coverage, MHR normalization/conversion evidence and atomic coherent scene publication |
| Scene to serialization/reload | Exact selected scene, official adapter, all required files, successful reload/decode/render and R1-R6 review |

A generic nonempty `.pth` or `.pkl` check cannot certify any of these gates. Native binary formats require trusted pinned adapters; loading arbitrary pickle data is not a validation strategy. Preserve producer files, record relocation separately, and never use copied native cache markers as proof of execution.

The first integration milestone is one complete episode transferred between at least two actual hosts with checked predecessor identities, followed by scene conversion and reload. Until that succeeds, adjacent native stages may share one capable worker while everyone continues independent module development. Keep the same reviewed interfaces so the worker allocation can change without changing scientific meaning.
