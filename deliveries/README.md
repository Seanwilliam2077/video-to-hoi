# Module deliveries

This is the Git submission location for small module delivery records and reviews. It is a **human review index**, not a native `StageResult` schema, transport service or automatic acceptance system. The folders initially contain instructions only; no model result is registered by this setup.

| Work area | Submit the delivery record here | Typical delivery |
| --- | --- | --- |
| M1 Platform | [platform/](platform/) | Observations, contracts/checks, transport or scene/export evidence |
| M2 Human | [human/](human/) | Human/depth bootstrap or complete native 01-04 products |
| M3 Object | [object/](object/) | Metric object asset or native 05 tracking |
| M4 Physics | [physics/](physics/) | Native 06-07 results or a separately identified research candidate |

Members can work in any module. Record the actual executor and reviewer for each delivery; no permanent module assignment is required.

## Where each part goes

| Part | Destination |
| --- | --- |
| Code, configuration and tests | Their existing repository paths, through a task branch and pull request to `main` |
| Small delivery record | `deliveries/<module>/<unique-delivery-id>/README.md`, copied from [TEMPLATE.md](TEMPLATE.md), in the same or a linked pull request |
| Small check reports and review | Optional `checks/*.json` and `review.md` beside that record; link larger evidence by exact package/object identity |
| Large outputs | A new result-specific [GitHub Release](https://github.com/Seanwilliam2077/video-to-hoi/releases), or an agreed team store with exact accessible object URLs |
| Local working files | Unique local run/artifact directories; keep them out of ordinary Git |

Use a unique task branch such as `human-frame-map-20261015`. The existing `platform`, `human`, `object` and `physics` branches are work-area references, not personal assignments. A flat task name also avoids the Git ref conflict between an existing `human` branch and `human/<task>`.

The suggested naming for **future** run releases is `result-<delivery-id>`. No result release is created by this guide. Create a new tag/package for new output bytes, and never replace earlier results or add run outputs to historical model releases. Record the exact release tag, asset filename, byte count and SHA-256, or the team-store object version and equivalent hashes. A moving `latest` URL is insufficient. Use only storage permitted for that content; exclude credentials and private machine paths from public records.

## Submit and receive a delivery

1. Choose a unique delivery ID and copy the template into the appropriate module folder. State whether the work is documentation, code, an independent synthetic fixture or a real Track 1 run. Use a new result identity for each attempt or changed output.
2. Link the code revision, command/environment, expected outputs, actual upstream identities, source episodes and package locations. Preserve raw output and check evidence. Use only Track 1 or independently authored synthetic material; unknown provenance and all Track 2 derivatives are prohibited.
3. Run the applicable checks and record their actual scope and outcome. A code-only change can have no output package; explain that explicitly. Unrun checks remain unrun.
4. Open a pull request to `main` containing the small delivery record. Code may be in that pull request or another linked one. Consumers review the input/output contract and evidence for the affected stages.
5. The receiver obtains the exact required files, verifies their hashes and dependency closure, then runs applicable local content checks. Record receiving-machine verification and semantic review separately from the producer's checks.

**Submitted, merged, structurally checked, quality accepted and export accepted are distinct conclusions.** A merged note does not select a baseline or launch the next stage. Append review evidence without rewriting the producer's original claim; if results change, create a new delivery and link the superseded record. The PR preserves the review history.

## Current implementation boundary

The [internal delivery gates](../docs/delivery-gates.md) cover six v1 stages, currently backed by fake reconstruction implementations. Their passing results establish the covered format and dependency checks, not real reconstruction quality, native MHR conversion or official submission validity. Local receipts depend on the available upstream chain and current files. Copying `.gates` or a passing report to another machine does not make a run portable; do not synchronize local locks, current pointers, registry selections or caches through Git.

The [native transport design](../docs/distributed-execution.md) separately proposes producer `StageResult`, `TransportManifest` and receiver records, including future `handoffs/` and `reviews/` layouts. Typed native validation and package import/relocation remain pending. This directory can link those records when implemented; it does not replace them or their checks.

Keep large logs, videos, masks, depth, meshes, arrays, archives and weights out of this directory. The ignore rules do not block every binary extension. Keep exact raw evidence in the permitted package; label any redacted Git summary as a summary, not a byte-identical receipt. See [team handoffs](../docs/team-handoffs.md) for dependencies and consumer review scopes.
