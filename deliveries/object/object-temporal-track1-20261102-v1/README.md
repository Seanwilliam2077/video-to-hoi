# M3 object reconstruction and full HOI handoff

Published review candidate, 2026-11-02. All 30 Track 1 episodes and 16,563 original
frames are included. The M3 owner and Codex prepared this delivery for Human,
Physics and Platform review. Reconstruction quality and official export remain
pending; publishing these files does not accept a downstream dependency.

## Download and inspect

- [Release: object results and 30 separate complete HOI videos](https://github.com/Seanwilliam2077/video-to-hoi/releases/tag/result-object-temporal-track1-20261102-v1).
- [Complete archive, including all videos](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/object-temporal-track1-20261102-v1.zip).
- [Exact internal file manifest](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/manifest.json).
- [Producer checks](checks/producer-checks.json) and [publication checks](checks/publication.json).
- [Registry declaration checks](checks/registry-checks.json) and [unit check command/result](checks/unit-tests.json).

Archive: **202,896,117 bytes**; SHA-256:
`0310d8db733783f7edf4360b4715c43cc5b0893c5f563a3c1b33446a3a29c7cc`.
All 35 release assets were verified against server digests and independently
downloaded without authentication with matching bytes and SHA-256.

Extract the archive and open `index.html` to select any full HOI preview.
Each episode contains the final fitted `object.glb`, complete `scene.npz`,
`motion.npz`, `camera.json`, raw frozen Human context, agent-bpy authoring files,
and explicitly labelled redacted inference/check/review evidence summaries.
The manifest lists each file's exact size and SHA-256 and binds original source
video identities, selected attempts, runtime source hashes and decoder identities.

## Coordinate and dependency contract

Object geometry has its fitted estimated-metre scale baked in once.
For an exported vertex, `point_scene = R[t] @ vertex + translation[t]`;
`object_scale = 1`. Preserve the exported mesh origin and axes. Do not recenter
the mesh or apply the initial fitting scale again.

The shared frame is the native OpenCV camera: x right, y down, z forward.
Camera-to-scene is identity; per-episode estimated intrinsics and source image
dimensions are recorded. Original decoded frame IDs are exactly `0..T-1`,
including occluded frames. Reconstruction uses original frame 0 as its anchor;
no competition reference alignment has been performed. Camera calibration,
metric accuracy and hidden motion are not ground-truth verified.

Human context belongs to these exact scenes. Raw `mhr_*` arrays and decoder
normalization metadata are preserved; they are not official pose136/scales68.
Do not substitute another teammate's Human result without reviewing original
frame, camera and scale compatibility. Calibrated per-frame object confidence
is **not delivered**, so the registry's Motion confidence requirement remains
missing. No internal/native execution gate or typed native importer is claimed.

## Source and reproduction

Only permitted Track 1 observations and their derived inference data are used;
no Track 2 data or derivatives are included. Actual project runtime source bytes
are archived under `runtime-sources/` and bound per selected run. The local base
commit was `e08086c2903e61b581b824f44b3b5fe54aedbb2f` on `windows-cari4d`, with
uncommitted runtime changes. There is no published generating commit or code PR.
The release tag points to existing remote commit
`fb0bac9f02d346d32e086036c80f7c3041978d56` solely as a storage anchor.

The package is self-contained for saved video review and object-state inspection.
Rerunning inference also needs the original matching video/mask/depth inputs,
licensed model dependencies and environment; those are not mirrored here.
Raw local receipts were checked before packaging; portable redacted summaries
are explicitly labelled and retain the original receipt hashes.

Packaging command, from the selected result checkout:

```powershell
python tools/package_object_handoff.py --batch <selected-batch-directory> --output <fresh-output-directory> --delivery-id object-temporal-track1-20261102-v1
```

Receiver check, from the extracted payload, using Python 3.11+ with numpy,
trimesh and av installed:

```powershell
python verify_handoff.py . --decode-videos
```

This check runs no inference or network access. Inference resource usage was
not measured by this handoff. Packaging took about 66 seconds on the producer.

## Producer checks and review state

| Conclusion | Actual state |
| --- | --- |
| Execution | All 30 selected episode computations completed; original batch unchanged by packaging. |
| Portable structure | Pass: 473 file identities, mesh/pose/camera pairing, proper full-frame transforms, Human context, archive CRC and per-file bytes. |
| HOI preview coverage | Pass: fully decoded all 30 previews, matching original frame counts and frame rates. |
| Unit checks | Explicit local CI allowlist: 334 passed; independent synthetic fixtures only. |
| Consumer verification | Pending; no receiver has accepted relocation, compatibility or dependency closure. |
| Reconstruction quality | Pending; ring motion, desk orientation, block landing and hand contact require review. |
| Official export/reload | Not run. No competition package or score is claimed. |

## Progress board registration

[60 submitted declarations](dashboard-submitted-records.json) cover Objects and
Motion for all 30 episodes. Matching immutable files are under
`tracking/records/object-temporal-20261102-v1-ep*-*.json`. Structural evidence
refers only to this portable delivery. No accepted upstream records from another
Human pipeline are borrowed, no generating commit is invented, and Motion
confidence remains missing.

These records and the Object preview catalog are proposed on `codex/m3-object-preview`. The public board
reads the registry on remote `main`; publishing a Release does not update it.
To integrate on an authorized checkout containing the board tools:

```powershell
python tools/progress_board.py import deliveries/object/object-temporal-track1-20261102-v1/dashboard-submitted-records.json
python tools/progress_board.py build
python tools/progress_board.py validate
```

The owner authorized publishing this task branch and opening a PR on 2026-11-02.
Merging `main` still requires separate owner confirmation. Pages publishes the
records and full HOI previews after the reviewed changes reach `main`.
Next: receivers review exact scene compatibility and observed motion before
promoting any declaration or using these results as a downstream prerequisite.

## Board preview integration

[Integration checks](checks/board-preview-checks.json): 918 Python checks from the
37-file CI allowlist and 51 Node checks passed. The local board exposes Objects
and Motion for all 30 videos; actual release video playback, English/Chinese,
phone layout, light/dark themes and the status countdown were checked. This is
viewer/registry validation; the earlier 334-test result belongs to the packaging
checkout, and neither test run accepts reconstruction quality.

## Full HOI previews

| Episode | Object | Original frames | Full HOI video |
| --- | --- | ---: | --- |
| 00 | hula_hoop | 790 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000000_hula_hoop.mp4) |
| 01 | hula_hoop | 668 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000001_hula_hoop.mp4) |
| 02 | hula_hoop | 866 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000002_hula_hoop.mp4) |
| 03 | big_red_bowl | 592 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000003_big_red_bowl.mp4) |
| 04 | big_red_bowl | 747 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000004_big_red_bowl.mp4) |
| 05 | big_red_bowl | 668 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000005_big_red_bowl.mp4) |
| 06 | iron | 816 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000006_iron.mp4) |
| 07 | iron | 777 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000007_iron.mp4) |
| 08 | iron | 634 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000008_iron.mp4) |
| 09 | white_desk | 415 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000009_white_desk.mp4) |
| 10 | white_desk | 577 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000010_white_desk.mp4) |
| 11 | white_desk | 877 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000011_white_desk.mp4) |
| 12 | black_pan | 405 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000012_black_pan.mp4) |
| 13 | black_pan | 425 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000013_black_pan.mp4) |
| 14 | black_pan | 442 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000014_black_pan.mp4) |
| 15 | foam_grass_block | 501 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000015_foam_grass_block.mp4) |
| 16 | foam_grass_block | 360 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000016_foam_grass_block.mp4) |
| 17 | foam_grass_block | 419 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000017_foam_grass_block.mp4) |
| 18 | paint_roller | 535 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000018_paint_roller.mp4) |
| 19 | paint_roller | 443 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000019_paint_roller.mp4) |
| 20 | paint_roller | 549 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000020_paint_roller.mp4) |
| 21 | pink_foam_roll | 563 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000021_pink_foam_roll.mp4) |
| 22 | pink_foam_roll | 533 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000022_pink_foam_roll.mp4) |
| 23 | pink_foam_roll | 552 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000023_pink_foam_roll.mp4) |
| 24 | short_wood_stool | 420 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000024_short_wood_stool.mp4) |
| 25 | short_wood_stool | 365 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000025_short_wood_stool.mp4) |
| 26 | short_wood_stool | 399 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000026_short_wood_stool.mp4) |
| 27 | white_laptop_cart | 440 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000027_white_laptop_cart.mp4) |
| 28 | white_laptop_cart | 366 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000028_white_laptop_cart.mp4) |
| 29 | white_laptop_cart | 419 | [Open/download](https://github.com/Seanwilliam2077/video-to-hoi/releases/download/result-object-temporal-track1-20261102-v1/episode_000029_white_laptop_cart.mp4) |
