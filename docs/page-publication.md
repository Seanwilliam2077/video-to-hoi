# Team-page publication record and remaining work

Updated **2026-10-03**, Asia/Shanghai. The current Reference follow-up replaced existing answers in place. Static artifact publication remains incomplete. The single active project specification is [track1-requirements.md](track1-requirements.md), R1–R6; older page text and earlier receipts do not override it.

## Current verified follow-up

On the existing [Reference page](https://claude.ai/artifact/YXL887jxTTp7aQByCQvC6b), the coordinator overwrote the **six original Track 1 Q&A answers and the seventh summary answer** at their existing rows with the latest requirements. This was a replacement, not another appended correction or a duplicate question.

After reload, the page still showed **10 open questions and 7 answered questions**. The answers now use the first reference frame's explicit original-video identity for the one shared human/object alignment; there is no active choice between first-reference and first-scored alignment. Native MHR, world-posed object Chamfer, reference-relative second differences, all-frame object coverage and Track 1-only assets/estimates are the current requirements.

Evidence for this replacement is recorded in [page-replacements-20261003.json](evidence/page-replacements-20261003.json), with the rendered answer text in [reference-answers-20261003.txt](evidence/reference-answers-20261003.txt). No question was sent to the organizer, and no Claude AI prompt was sent to perform the update.

## Earlier publication history

The earlier **35 database additions** remain recorded: 18 tasks, four progress notes, 12 resources and one Reference question/answer. Their receipt is [page-publication-20261003.json](evidence/page-publication-20261003.json); [page-updates-20261003.json](page-updates-20261003.json) separates proposals and actual publications. Those additions are preserved as history, not the current rule authority. The Reference summary answer from that earlier pass is one of the seven answers replaced in the current follow-up.

The earlier core-document publication at [164407045fddbd6b8730c66411f34ddbf58b8033](https://github.com/Seanwilliam2077/video-to-hoi/commit/164407045fddbd6b8730c66411f34ddbf58b8033) is also historical evidence. It does not assert that the current follow-up, local source edits or diagnostic code have been committed, pushed or statically published.

Do not duplicate the earlier task/resource rows or the seven answered questions when completing the remaining work. Displayed task numbers are not database UUIDs; unavailable row IDs and publication revisions must remain null rather than invented.

## What remains unpublished

Overall status remains **`pending_static_publication`**. The coordinator checked both the direct artifact interface and the Cowork Artifact options; neither exposed a usable source editor. An authenticated database UI was available for the answer replacements, but this does not imply that complete artifact source can be edited.

- Static Overview/Challenge text on the Reference and module subpages has not been republished and may still contain superseded wording.
- The updated local team hub and diagram, `docs/hub.html` and `docs/pipeline.svg`, have not been republished to the existing hub artifact.
- The updated local Chinese status page, `docs/status.html`, has not been republished to its existing artifact.
- Proposed full task bodies, acceptance fields and translation-dictionary changes were not applied through an unsupported live schema.

The corrected Reference answers are live; the static-source refresh is not complete. Do not describe all pages as updated. A Git commit or updated local HTML file alone does not update a live artifact.

## Completing the static refresh

1. Read the live source and current database before any replacement; authoritative source copies of the Reference and module artifacts are not stored in this repository.
2. Use the same existing artifact URLs and retain the verified database rows. Follow the single current [R1–R6 specification](track1-requirements.md); preserve older rules only as explicitly superseded history.
3. Keep the navigation and EN/中文 behavior described in [AGENTS.md](../AGENTS.md). Follow [status.md](status.md) for the status page's plain-language requirements.
4. Publish through an available source-editing interface, then reload and verify both languages, navigation, retained rows and resource links.
5. Record the actual scope, timestamp, exposed revision and verification evidence. Mark static publication complete only for artifacts whose new source is confirmed.

The user's authorization to update these pages persists. The remaining blocker is the unavailable source editor, not login or permission. Current implementation status belongs in [track1-compliance.md](track1-compliance.md); newly edited answers and created tasks do not establish real reconstruction, a completed export or official scores.