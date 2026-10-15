# TEMPLATE ONLY - replace with a unique delivery title

This file is not a submitted result. Copy it to `deliveries/<module>/<unique-delivery-id>/README.md`, replace every placeholder, and delete this notice. Use `not run`, `not applicable` or `pending` with a reason where appropriate; do not invent successful checks. This is a human review index, not a machine-readable native result schema.

## Scope and identity

| Field | Value to supply |
| --- | --- |
| Delivery ID / date | `<unique-delivery-id>` / `<YYYY-MM-DD>` |
| Module / exact stage or capability | `<platform, human, object or physics>` / `<stage>` |
| Work type | `<documentation, code, independent synthetic fixture or real Track 1 run>` |
| Actual executor / contributors | `<participants for this attempt>` |
| Code PR and exact source commit | `<PR link>` / `<full commit hash>` |
| Related or superseded delivery | `<exact record link and reason, or none>` |
| Intended consumer and review scope | `<downstream stage and required review>` |

Describe the problem, delivered change and explicit limits in a short paragraph. State whether this is a baseline candidate, research candidate or support work; none is accepted by default.

## Inputs and reproduction

- Source policy: `<Track 1 episode IDs and video/timeline hashes, or independent synthetic provenance; no Track 2 derivatives>`.
- Frame/target coverage: `<original frame IDs, timestamps, target IDs and completeness; or not applicable>`.
- Upstream dependencies: `<exact result IDs, stage/profile/schema versions and input hashes; distinguish declared inputs from actual observed consumption>`.
- Model/decoder/configuration: `<pinned identities, hashes and seeds where applicable>`.
- Environment: `<OS, runtime, packages, device requirements and environment record reference; no credentials or private host paths>`.
- Command: `<reproducible command with documented machine-local path placeholders>`.
- Resources: `<observed elapsed time, memory/disk usage and package size; or not measured>`.

For research candidates, identify the frozen baseline, permitted changes, comparison protocol and observed regressions. Preserve the baseline independently.

## Output inventory and location

| Output or required package | Exact location / immutable object identity | Bytes | SHA-256 | Purpose |
| --- | --- | --- | --- | --- |
| `<artifact or package>` | `<release tag + asset name, or team-store object URL/version>` | `<actual bytes>` | `<actual hash>` | `<produced output or dependency>` |

List all expected outputs and unresolved missing files. A code/documentation-only delivery may state that no runtime output package exists. For packaged outputs, link the internal file inventory and all required common/upstream packages; an archive hash alone does not establish complete dependencies. Keep large files outside Git. Suggested future release naming: `result-<delivery-id>`; do not modify old model releases.

## Producer checks

| Check | Validator/profile and command | Actual outcome | Evidence | Scope and limitations |
| --- | --- | --- | --- | --- |
| `<check>` | `<pinned validator/profile>` | `<pass, fail or not run>` | `<relative checks/file.json or exact external reference>` | `<what was and was not checked>` |

State separately whether the evidence concerns internal fake-stage contracts, native outputs, scene conversion, reconstruction quality or export/reload. A passing internal v1 gate does not accept a real model or official submission. Preserve exact raw receipts in the result package; label any redacted summary and retain a reference to its source.

## Receiver verification and review

Use this section or a linked `review.md`. Record actual reviewer, date, exact producer result/package identities, bytes/hash verification, available dependency closure, local content check results, any path relocation and semantic findings. Pending reviews must say `pending`. Native package import/relocation is not yet implemented; do not present manual file copying as verified import. Current internal receipts still require their valid local upstream chain.

## Separate conclusions and next action

| Conclusion | Actual state and supporting evidence |
| --- | --- |
| Execution | `<completed, failed, not run or not applicable; evidence>` |
| Structural delivery / dependencies | `<pass, fail, pending or not applicable; scope and evidence>` |
| Consumer verification | `<pass, fail, pending or not applicable; evidence>` |
| Reconstruction quality | `<accepted, rejected, pending or not applicable; independent review evidence>` |
| Export / reload acceptance | `<accepted, rejected, pending or not applicable; evidence>` |

Known failures or limitations: `<specific unresolved findings>`.

Next action: `<what is required before the intended consumer can continue>`. Submitting or merging this record alone does not accept the result or start a downstream model.
