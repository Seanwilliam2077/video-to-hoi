# Portable Human result delivery

`tools/package_human_delivery.py` packages completed episodes from the
[standalone batch runner](human-batch.md). It reads each episode's
`current-result.json` and actual successful stage receipts. It does not launch
inference, upload assets, invent a metric scale, accept quality, or export an
official scene submission.

## Create a fresh review package

Run after native checks, both full-frame reviews and the synchronized hand
comparison finish. The output directory must not already exist:

```powershell
& <python-with-numpy> tools/package_human_delivery.py `
  --batch-root <completed-batch-root> --episodes 16 11 `
  --output <fresh-delivery-directory> `
  --delivery-id human-track1-20261021-pilot01
```

Use a unique delivery ID and fresh directory for every delivery. Supplying no
release URL produces only local packages and an upload manifest with null URLs;
it creates no dashboard record. A partial episode or missing successful stage
cannot be packaged as complete.

Each per-episode ZIP contains:

- Both baseline and hand-candidate native MHR NPZs and their reports. The
  authoritative `parameters/selectedhuman.npz` is an exact byte copy of the
  selected candidate, bound by `selection.json`.
- Complete baseline/refined comparison videos, every-frame JPEG viewers,
  contact sheets and frame diagnostics where emitted by the renderer.
- The synchronized hand comparison, selected snapshots and independent
  MediaPipe observation arrays, crop images and observation provenance.
- Native checks, actual stage start/finish receipts and logs, compact runtime
  inventories, immutable input/model/code bindings and source snapshots emitted
  by the fitter.
- A portable top-level `index.html`, a file manifest and complete SHA256SUMS.

Raw per-frame SAM caches, large native mesh arrays, source video/mask assets and
model weights are omitted. Derived geometry can be regenerated with the exact
native MHR model and conventions recorded in the reports. Those omitted inputs
and model dependencies are still required to rerun inference; the review ZIP is
not a self-contained model environment or a complete HOI submission.

The packager verifies included bytes against the actual selected successful
stage inventories, validates native parameter shape and full original frame
coverage, checks source/model agreement, and refuses changed results. It verifies
all local HTML links after relocation and every ZIP member by CRC and SHA256.
Keep the returned package descriptor, not just the archive filename.

For direct playback in the dashboard, each episode also exports exact standalone
copies of the baseline full-body comparison, refined full-body comparison and
synchronized hand video. One or both actual contact sheets and an optional actual
hand snapshot are separate JPEG assets. These are copies of existing verified
renders; the packager does not create or alter their visual content. They are
additional to the complete portable ZIP.

The package descriptor's `previews` list supplies exact filenames, local paths,
byte counts, SHA256, MIME types, episode identity, source-video and MHR-model
identities, original episode length and one of these roles:

| Role | Content and coverage |
| --- | --- |
| `baseline_full_body` | Baseline comparison MP4; full original sequence |
| `refined_full_body` | Hand-candidate comparison MP4; full original sequence |
| `hands_comparison` | Same-crop source/observation/candidate MP4; full original sequence |
| `baseline_contact_sheet` | Actual baseline contact-sheet JPEG; sampled original frame IDs |
| `refined_contact_sheet` | Actual candidate contact-sheet JPEG; sampled original frame IDs |
| `hand_qa_frame` | Existing hand-comparison JPEG; its actual original frame ID |

Full-sequence preview coverage is `{"kind":"full_sequence","original_frame_count":T}`.
Still-image coverage is `{"kind":"sampled","original_frame_ids":[...]}`. Video
coverage must agree with the actual renderer's complete-frame/source identity
summary. Contact-sheet IDs come from its saved summary; the hand snapshot ID
comes from its original-frame filename. A sampled image is never presented as
complete temporal coverage. These direct assets retain the same pending quality
and scale limitations as the native candidates.

`depth-scale.json` explicitly records `unverified_monocular_scale`. It retains
the estimated intrinsics and their identity, native unit conventions and
unresolved metric/shared human-object scale. It supplies no arbitrary `1` scale,
no calibration claim and no unavailable first-reference-frame alignment.

## Prepare actual upload URLs and dashboard declarations

When an actual result-specific release/team-store download base is selected,
also supply that base and the full source commit:

```powershell
& <python-with-numpy> tools/package_human_delivery.py `
  --batch-root <completed-batch-root> --episodes 16 11 `
  --output <another-fresh-delivery-directory> `
  --delivery-id <unique-delivery-id> `
  --source-commit <full-lowercase-40-character-commit> `
  --release-base-url <actual-http-or-https-asset-download-base>
```

This creates `upload-manifest.json`, listing the exact local archive, checks,
selection, input bindings, scale record and package descriptor for each episode,
as well as every direct video/JPEG preview, including filename, byte count,
SHA256, content type and intended download URL. Preview MP4 assets use
`video/mp4`; JPEG assets use `image/jpeg`.
No upload is performed. Upload those exact assets and verify their remotely
accessible identities before importing `dashboard-submitted-records.json` into
the dashboard. Do not import a declaration whose assets are still local.

The optional dashboard envelope uses the existing `hoi_progress.v1` schema and
contains **submitted** Human records only. Each record binds the actual
`body-inputs-20261016-epXX` input declaration. The packager checks that its camera
artifact SHA256 matches the intrinsics actually consumed by this inference run.
The Human artifact is the portable ZIP; the depth-scale artifact is the separate
honest limitation record. Structural evidence links the actual native checks.
Quality and export remain `not_run`; object coverage is not claimed. These
declarations cannot bypass the dashboard's upstream dependency or acceptance
rules. Complete standalone Body inputs remain distinct from the broader Inputs
stage.

The envelope is a publication preparation artifact, not proof of upload or
acceptance. Actual executor, environment, independent review and receiver
verification belong in the corresponding `deliveries/human/` delivery note.
Keep the native parameters authoritative and retain rejected/uncertain results
in their existing evidence history.
