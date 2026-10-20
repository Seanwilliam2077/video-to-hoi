# Video progress and evidence registry

This registry powers the public [30-video progress board](https://seanwilliam2077.github.io/video-to-hoi/)
([source](../docs/progress-board/index.html)).
It indexes per-video, per-stage evidence for iteration and review. No registered
record means **unregistered / unknown**, not that nobody has started the work.
The registry now contains 30 `body-prep-v1` Inputs submissions, one per video,
linked to the [verified Body input delivery](../deliveries/platform/body-inputs-track1-20261016/README.md).
All standalone Body inputs passed their dedicated checks. Full Inputs still
needs object/depth/scale evidence, so these records remain `submitted`, with
full-stage checks `not_run`; none represents accepted human reconstruction.
Shared platform capabilities are shown separately and never count as completed videos.

The hosted board uses GitHub Pages and needs no sign-in. The
[publication workflow](../.github/workflows/progress-pages.yml) validates the
registry and board, then publishes only the four board assets and
`tracking/board.json` when relevant changes reach `main`. It can also be run
manually from Actions. The deployed snapshot comes from that exact checkout;
use **Refresh repository** to read the latest merged snapshot immediately.
The earlier owner-private Site is no longer the team's primary entry point.
Git records remain the shared source of project progress. Browser drafts
remain local until exported, imported into the repository and merged; drafts
on the old Site or localhost are not transferred automatically to this origin.

## Data flow

```text
Existing Track 1 gallery metadata ───────────────┐
tracking/records/<unique-record-id>.json ────────┼─ tools/progress_board.py build
Six internal stage requirements and capabilities┘           │
                                                     tracking/board.json
                                                           │
                                                  progress-board interface
```

Only metadata is read from `docs/track1-videos/index.html` (`const DATA`). The
builder never downloads or opens videos, model outputs, or reference assets.
The inventory is exactly `episode_000000` through `episode_000029`. The snapshot
`source.revision` is the repository HEAD when built, not a claim that all content
was already committed at that revision. `updated_at` is the snapshot build time,
not a model run time. CI checks that its episodes, stages, capabilities and
records agree with the sources; rebuilding after every commit is unnecessary.

## Register a task or delivery

1. On the board, select a video, stage and iteration; prepare a local draft with
   the actual contributor, current state and evidence. Export the drafts as JSON.
   Alternatively, copy [record-template.json](record-template.json) outside
   `tracking/records/` and fill it in directly. Anyone can contribute to any
   module; there is no permanent owner assignment.
2. Create the module delivery record in [deliveries/](../deliveries/README.md).
   Keep large data in a unique result release or agreed team storage. Include
   exact artifact URLs, byte counts and SHA-256 hashes, plus check report links.
3. Import the exported JSON, build and validate the shared snapshot, then review
   and commit the files on a task branch and open a PR to `main`:

   ```bash
   python tools/progress_board.py import /path/to/exported-drafts.json
   python tools/progress_board.py build
   python tools/progress_board.py validate
   git diff -- tracking/
   git add tracking/records/ tracking/board.json
   git commit -m "Register video delivery progress"
   git push -u origin HEAD
   ```

4. After the PR merges, refresh the board's repository data to see the shared
   records. A browser draft and its exported file alone do not update the team
   record. Keep module delivery records linked from the progress entries.

`import` accepts one raw record or the exact browser export envelope
`{"schema_version":"hoi_progress.v1","records":[...]}`. It validates the whole
batch, including dependencies between records in that batch, before appending
`tracking/records/<id>.json` and rebuilding `tracking/board.json`. A full board
snapshot is not a record-import payload. The explicit `build` above is useful
when also editing the inventory; import already rebuilds the snapshot itself.
An identical ID and body are idempotent; a changed body with the same ID is
rejected. To advance status, correct a mistake, or rerun a stage, create a new
record ID and later `created_at`. Preserve the old record for review. Never edit
an already published record in place. Direct Git edits still require PR review:
the CLI cannot prevent someone deliberately rewriting history outside it.

The web interface can prepare a JSON record, but it does not push Git changes
or upload artifacts. Import and commit the exported JSON through this workflow
before it becomes shared project progress. `build` intentionally replaces the
generated snapshot; it never replaces an existing immutable record.

```bash
python tools/progress_board.py build
python tools/progress_board.py validate
```

Both commands support `--root /path/to/repository`; `import` does too. Validation
is offline and uses Python's standard library. Exit status is `0` for valid
declarations/snapshot and `1` for invalid or inconsistent data. Parser usage
errors exit `2`. A failed import does not silently overwrite a record.

## Record contract: `hoi_progress.v1`

### Browsing the submitted contents

The public board separates source video/timeline verification, Body input
verification, object-mask declarations, depth declarations and Body output
declarations. Preparation counts cover all 30 videos in the selected iteration;
the matrix filters do not change that denominator. These counters do not grant
full-stage acceptance.

Open a preparation card for the per-video list, or any stage in the matrix for
its files, missing requirements, checks and history. Input details include
estimated camera values, selected original QA sheets, pinned JSON receipts and
the exact path of each item inside the released archive. Remaining QA sheets
and full frame sequences are available in the archive. The record link fixes
the episode, stage and immutable record ID, including historical deliveries.
Links to browser-local drafts work for teammates only after those records have
been submitted to Git.

`delivery-catalog.json` supplements the registry with inspected content; it does
not change a record's state. `tools/build_delivery_catalog.py` verifies the
pinned all-30 delivery evidence and unchanged QA JPEGs before building it:

```bash
python tools/build_delivery_catalog.py
python tools/build_delivery_catalog.py --check
```

The check works in CI without ignored videos or model data. A catalog entry is
shown only for the exact complete record it describes. A new iteration or
modified record never inherits old readiness or review images. A missing or
invalid catalog leaves the registry usable and marks preparation unverified.

For a new delivery, first register its actual output URLs, hashes, sizes and
check reports using the normal immutable-record workflow. All six stages will
immediately expose those files through the generic delivery view. Rich previews
require a separately verified catalog entry and stage renderer; do not reuse
an older entry or fabricate previews. `build_delivery_catalog.py` continues to
support the published all-30 Body input delivery only. Human candidates use the
separate optional catalog described below.

### Published Human candidate previews

`human-catalog.json` starts empty. A matching, non-stale Human record can display
three actual release videos: the fixed-body baseline, the independent MediaPipe
hand candidate, and hand close-ups. Actual contact sheets or hand snapshots list
their original source frame IDs; they are not full-clip quality certificates.
Both languages identify the selected local candidate, including the baseline
when hand fitting abstained or the image guard rejected the alternative.

Only submitted real Track 1 Human records are supported by this preview profile.
The complete record, native archive, source/model identities, original frame
count, checks, selection and scale limitations must agree. Media and evidence
URLs are restricted to this repository's exact GitHub Release download paths,
with the same result tag and record-specific filenames. There are no local
drive links, localhost links, arbitrary remote embeds or inferred result entries.
The viewer fetches metadata and plays only the media the reader opens; it does
not run inference, decode native parameters or promote acceptance.

After packaging and publishing exact result assets, build this optional index:

```bash
python tools/progress_board.py import /path/to/dashboard-submitted-records.json
python tools/build_human_catalog.py \
  --packages /path/to/packages.json \
  --records /path/to/dashboard-submitted-records.json \
  --release-base-url https://github.com/Seanwilliam2077/video-to-hoi/releases/download/<delivery-id> \
  --publication-receipt /path/to/publication.json
python tools/build_human_catalog.py --check
```

The publication receipt must record successful anonymous verification of the
exact remote asset bytes. The builder rehashes local preview/evidence files,
checks cross-file selection and provenance, and uses the browser's pure schema
validator through Node. It appends new exact record entries, never rewrites an
existing entry with changed content. Commit the generated catalog with the
immutable progress records through the normal PR workflow. Pages publishes it
beside `board.json`; repository refresh also reloads it.

The catalog links complete baseline/refined videos and hand comparisons while
retaining `submitted`, unverified scene scale and pending quality/export review.
An absent, invalid, mismatching or stale Human entry falls back to the ordinary
registered artifact cards. It does not borrow media from a previous result or
affect Body readiness, stage dependencies or official acceptance requirements.

| Field | Meaning |
| --- | --- |
| `id` | Unique lowercase ID, up to 96 safe letters/digits/`.`/`-`/`_`; filename matches ID. |
| `episode_id`, `stage` | One registered episode and one of `inputs`, `human`, `objects`, `motion`, `refine`, `export`. |
| `iteration` | Safe lowercase experiment ID, such as `baseline` or `candidate-camera-v2`. |
| `state` | `planned`, `running`, `blocked`, `submitted`, `checked`, `accepted`, or `failed`. |
| `kind` | `real_track1`, `synthetic`, or `engineering`; only `real_track1` can be checked/accepted video progress. |
| `created_at` | Full ISO `YYYY-MM-DDTHH:mm:ss[.sss]Z` or explicit `±HH:mm` timezone, with at most three fractional digits; records sort by timestamp, then ID. |
| `contributor`, `reviewer` | Actual contributor; actual reviewer required for `accepted`. Roles float per task. |
| `summary`, `blocker`, `next_action` | Human-readable context. Blocked state requires a blocker. |
| `delivery_url`, `source_commit` | Module review-record URL and full 40-character lowercase source Git SHA; required for checked/accepted. |
| `upstream` | Exact `{stage, record_id}` bindings. Same episode required; explicit reuse from another iteration is allowed. |
| `artifacts` | One `{requirement_id, label, url, sha256, bytes}` per declared stage output; hashes are lowercase SHA-256 and bytes are positive safe integers. |
| `coverage` | `{frames: nonnegative safe integer or null, objects_complete: boolean or null}`. Unknown values stay null. |
| `checks` | `structural`, `quality`, `export`, each `{status: "not_run" / "pass" / "fail", evidence_url: "..."}`. |

Every URL must be HTTP(S), with no embedded credentials, an ASCII hostname
(use punycode for international domains), and canonical dotted IPv4 or bracketed
IPv6 when using IP addresses. Encoded hostnames are not accepted. A check marked pass or
fail needs an evidence URL. An artifact entry must always include its URL, hash
and byte count; a missing output is omitted, not represented by an empty file.
Unknown fields, stages, episodes, duplicate artifact requirements and unresolved
predecessors are rejected. Unknown values must not be replaced with guessed
coverage, fabricated metrics or placeholder checks in a submitted record.
Integer values cannot exceed `9007199254740991`, keeping Python and browser
interpretations exact.

`checked` and `accepted` require all stage outputs, full declared frame coverage
equal to inventory metadata, declared complete object coverage, structural pass
evidence, a delivery URL, a source commit and every predecessor binding. Those
predecessors must themselves be checked/accepted real Track 1 declarations.
`accepted` additionally requires a reviewer and quality pass evidence; export
acceptance also requires export pass evidence.

These are **declaration completeness checks**. The registry does not fetch
artifacts, recompute hashes, decode frames, inspect quality reports, verify a
reviewer's identity, or establish official competition compliance. The existing
[delivery gate](../docs/delivery-gates.md) checks available local internal files;
the [current Track 1 requirements](../docs/track1-requirements.md) still govern
scientific and export acceptance. A check URL or an accepted label does not
replace that review. Native adapters and official-format verification remain
separate integration work. Synthetic/internal smoke passes cannot be imported
as checked or accepted real video results.

## Dependencies and iterations

The six stages use the conservative internal dependency profile: `inputs` has
none; every later stage binds all earlier stages. This is a coordination
boundary, not a verified read set for every future native model. Artifact
requirement IDs are published in `board.json` and in `tools/progress_board.py`.

A new record becomes the current observation for its episode, iteration and
stage even if it is running or failed. Descendants bound to an older observation
become **stale**, recursively. The board derives staleness without rewriting
history. When a candidate explicitly reuses a baseline predecessor, freshness
is checked in that predecessor's own iteration. A separate experiment does not
invalidate a frozen baseline. Upstream records cannot be newer than the consumer.
Checked/accepted consumers must also bind a coherent chain: each predecessor's
own upstream IDs must equal the corresponding IDs in the consumer. Mixing a
human result from one input attempt with an object result from another is rejected,
even when both attempts separately pass and neither is stale.
If observations have equal timestamps, lexicographically greater IDs win;
prefer distinct timestamps to avoid an ambiguous human interpretation.

Do not copy local `.gates/current` pointers, locks or mutable run directories
into Git. This registry is a human coordination index, not a model execution
controller, a native package importer or a replacement for local gate receipts.
