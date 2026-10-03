# Team-page publication record and remaining work

Updated **2026-10-03**, Asia/Shanghai. The coordinating agent updated the five existing subpages through their authenticated database UI and verified persistence after reload. **35 database additions are recorded: 18 tasks, four progress notes, 12 resources and one answered Reference Q&A. Static page source publication remains incomplete.**

The reviewed core documentation was committed and pushed to `main` at [164407045fddbd6b8730c66411f34ddbf58b8033](https://github.com/Seanwilliam2077/video-to-hoi/commit/164407045fddbd6b8730c66411f34ddbf58b8033). The coordinator verified the remote branch, and the local HEAD matched when this receipt was prepared. This publication receipt was written afterwards and is not part of that core commit.

[page-updates-20261003.json](page-updates-20261003.json) separates the original proposals from `actual_publications`. The [browser evidence](evidence/page-publication-20261003.json) preserves exact added task titles, bilingual notes, the Reference answer and module count snapshots. It is a byte-identical copy of the coordinator's saved receipt, SHA-256 `9486e4431cc0273c78657cc65c1fe744665acc2734feab4f59ff395a99774f24`. A Reference screenshot is saved outside the repository at `D:/V2H/review-evidence/reference-final-rules.jpg`; no image is included here.

## What was published

| Existing page | Verified additions | Persistent counts after reload |
| --- | --- | --- |
| [Platform](https://claude.ai/artifact/A9hrz9B8qed9WWqkFWpAHg) | Tasks #11-15, one bilingual progress note, three resources | 15 tasks, 3 progress records, 28 resources |
| [Human](https://claude.ai/artifact/3jiAK6fqoNwqGtLwDKf3pk) | Tasks #9-12, one bilingual progress note, three resources | 12 tasks, 2 progress records, 21 resources |
| [Object](https://claude.ai/artifact/35K4apKHN75wiUuB8qxmq8) | Tasks #9-13, one bilingual progress note, three resources | 13 tasks, 2 progress records, 22 resources |
| [Physics](https://claude.ai/artifact/DUFpvGZmA2DyuU68afM7DV) | Tasks #8-11, one bilingual progress note, three resources | 11 tasks, 2 progress records, 23 resources |
| [Reference](https://claude.ai/artifact/YXL887jxTTp7aQByCQvC6b) | One bilingual question/answer, marked answered | 10 open questions, 7 answered; answered count was 6 before |

Each module's task numbers map in order to that module's proposed tasks in the manifest. The UI accepts a task title and label; it did not provide the proposed rich body/acceptance schema or an editor for old task text. Displayed task numbers are **not database UUIDs**. No row UUID or artifact publication revision was exposed, so those fields remain null.

Existing records were preserved, and older tasks were not silently edited or marked complete. The four dated bilingual progress notes explicitly correct conflicting older Overview/task guidance and state implementation gaps. They do not claim model inference, measured gains or an official submission. Full acceptance details remain in the linked repository documents.

Each module received these three new resource URLs:

- [Final Track 1 compliance audit](https://github.com/Seanwilliam2077/video-to-hoi/blob/main/docs/track1-compliance.md)
- [G0-G4 implementation plan](https://github.com/Seanwilliam2077/video-to-hoi/blob/main/docs/implementation-plan.md)
- [Evaluation specification](https://github.com/Seanwilliam2077/video-to-hoi/blob/main/docs/evaluation.md)

These are the **actual mutable main-branch URLs** used in the live UI. The receipt records the resolved core commit and corresponding immutable content URLs for audit; it does not pretend the live links themselves are commit-pinned. Proposed design/contracts resources were already linked in existing static sidebars and were not added as new resource rows.

The Reference page uses **Challenge, Videos, Development and Q&A**, not the proposed Tasks/Progress/Resources layout. The added question is:

> 2026-10-03: Which final Track 1 requirements supersede the September notes? / 哪些最终要求取代旧说明？

Its topic is `Final requirements`, its source field links to the main-branch compliance audit, and its bilingual answer covers the six owner requirements, first-frame ambiguity and current implementation gaps. It was marked answered and verified after reload. **No question was sent to the organizer.** The four proposed Reference tasks/resources were not published into nonexistent tables; their draft records remain `not_applicable_live_schema`, partly covered by this clarification rather than falsely marked published.

## What remains unpublished

The overall status is **`pending_static_publication`**. The authenticated session exposes database creation controls but no usable source editor or Artifact editing connector was found.

- Static Overview/Challenge text on the five subpages still contains older wording. The new notes/Q&A correct it explicitly, but the source itself was not republished.
- The team hub's local `docs/hub.html` and `docs/pipeline.svg` changes were not republished to its existing artifact.
- The Chinese status page's local `docs/status.html` changes were not republished to its existing artifact.
- The full proposed task bodies, acceptance fields and translation-dictionary edits were not applied through an unsupported live schema.

The manifest's top-level and per-page `published: false` therefore mean **the complete source/content refresh is unfinished**. They do not deny the verified database additions, each separately recorded as published in `actual_publications`. Proposed drafts keep their own narrower reconciliation outcomes; a title/label projection is not labelled an exact full-payload publication.

## Requirements and evidence boundary

The owner's **2026-10-03 user message**, with no invented external URL, is recorded as six acceptance targets:

1. One human-derived Sim(3) from the first reference frame is applied to both human and object across the clip.
2. Human submission uses native MHR parameters; SOMA or per-frame human meshes are not substitutes.
3. Object CD evaluates the posed mesh in the common world frame.
4. ACC compares predicted and reference temporal second differences.
5. Objects have poses on every original frame, including occlusion/out-of-view.
6. Reconstruction assets and estimated parameters, including camera intrinsics, use permitted Track 1 sources; Track 2 assets and derivatives are excluded, even for matching iron, bowl, table or round-table objects.

The first-reference-frame wording must be reconciled with the inspected evaluator's first-scored-frame selection through an explicit frame map. Neither wording authorizes choosing raw frame zero arbitrarily.

The current design targets G0 evaluation/native conversion/provenance, G1 real episodes 16 and 12, G2 camera/scale/observations, G3 bounded repair, and G4 complete 30-episode export. Gates remain open until their actual evidence exists. A documentation update or a newly created task does not close a gate.

The authoritative repository documents are [design.md](design.md), [implementation-plan.md](implementation-plan.md), [evaluation.md](evaluation.md), [contracts.md](contracts.md), [workflow.md](workflow.md), [track1-compliance.md](track1-compliance.md) and [official-kit evidence](evidence/official-track1-20261002.json). The official-kit link is a source reference, not permission to download a mixed-track data archive; follow the code allowlist and the persistent Track 1-only policy.

## Completing the static refresh when an editor is available

1. Read the live source and current database before any source replacement. The repository does not contain authoritative source copies of the Reference or module artifacts.
2. Use the same existing artifact URLs. Do not create replacement pages, duplicate the 35 verified additions, or republish an older database snapshot over them.
3. Reconcile the remaining static text against the reviewed documents. Preserve dated history and unrelated rows. Distinguish corrected implementation claims from actual completed work.
4. Preserve the existing EN/中文 mechanism and provide Chinese entries for every added English string. Do not claim that the current bilingual database notes already changed the source dictionary.
5. Preserve the navigation order below, with the current page represented by a `span aria-current="page"`, not a link. Follow [status.md](status.md) for the status page's separate nontechnical wording.
6. Publish the reviewed source to the existing artifact, then reload and inspect both languages, navigation, retained database rows and resource links.
7. Add the actual timestamp, revision if exposed, verification evidence and exact scope to the receipt. Leave unavailable values null. Mark static publication complete only for artifacts whose new source is confirmed.

Navigation remains:

1. [Team hub](https://claude.ai/artifact/JAeDzfVxpRezsW73C26M4S)
2. [Status](https://claude.ai/artifact/U2vgULC1h3DurtZwyQRHS1)
3. [Reference](https://claude.ai/artifact/YXL887jxTTp7aQByCQvC6b)
4. [① Platform](https://claude.ai/artifact/A9hrz9B8qed9WWqkFWpAHg)
5. [② Human](https://claude.ai/artifact/3jiAK6fqoNwqGtLwDKf3pk)
6. [③ Object](https://claude.ai/artifact/35K4apKHN75wiUuB8qxmq8)
7. [④ Physics](https://claude.ai/artifact/DUFpvGZmA2DyuU68afM7DV)

The user's authorization to update these pages persists. An unavailable source editor is the remaining capability limitation; there is no remaining login requirement, and it is not a reason to ask for the same authorization again.
