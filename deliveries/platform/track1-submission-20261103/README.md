# Track 1 Kaggle submissions, 2026-11-03

## Scope and identity

| Field | Value |
| --- | --- |
| Delivery ID / date | `track1-submission-20261103` / 2026-11-03 |
| Module / capability | Platform / official export and submission, with a Temporal baseline |
| Work type | Real Track 1 submissions scored by the organizers |
| Actual executor / contributors | Claude Code on the owner's Windows workstation; the owner authorized each upload |
| Code branch and commits | `platform-track1-submission-assembly`: `1cf306b`, `b1faf50`, `8370ac9`, `5dcc1b6` |
| Upstream results | [sam-mediapipe-20261021](../../human/sam-mediapipe-20261021/README.md) selected baseline; [object-temporal-track1-20261102-v1](../../object/object-temporal-track1-20261102-v1/README.md) |
| Intended consumer and review scope | Human, Object and Physics reviewers: pairing, smoothing widths and the final-submission choice |

This is the team's first official Track 1 entry. [tools/assemble_track1_submission.py](../../../tools/assemble_track1_submission.py) assembled the published Human and Object results without changing their reconstructions, then the kit packer built the Parquet. The second round adds Gaussian temporal smoothing as a stopgap for the pass-through refine stage. Its widths were chosen from public leaderboard feedback, not from independent video evidence. Nothing here accepts reconstruction quality.

## Inputs and reproduction

- Source policy: the 30 Track 1 videos and results derived from them. The kit's Track 2 members were neither extracted nor read.
- Coverage: all 30 episodes, every original frame (16,563 frames). The packer keeps the 740,780 rows of the official sample.
- Kit: `v2d_submission_kit.zip`, downloaded 2026-11-03 (members dated 2026-11-01), SHA-256 `7e1fdd00b50189191b4f1bf03158607f4f1a7bb5bfb0bf867b14eb06270a77f0`.
- Human: `parameters/selectedhuman.npz` from each published Human archive, read by `fetch-human` through HTTP byte ranges with the zip CRC checked.
- Object: the published handoff archive, SHA-256 `0310d8db733783f7edf4360b4715c43cc5b0893c5f563a3c1b33446a3a29c7cc`. Its 17 runtime source files are now committed beside that delivery record; base commit `e08086c` is still unpublished.
- Environment: Windows 11, CPython 3.12 venv with the kit requirements. Assembly and packing take under a minute on CPU.
- Final command, where `<sha>` is `5dcc1b6fa3657c20521c31daa05a1110d8db058c`:

```bash
python tools/assemble_track1_submission.py assemble --human data/submission-inputs/human --objects <object-handoff> --out <episodes> --temporal-sigma 6 --object-sigma 16
python tools/assemble_track1_submission.py pack --kit <v2d_submission_kit> --episodes <episodes> --commit https://github.com/Seanwilliam2077/video-to-hoi/commit/<sha> --out track1.parquet
python <v2d_submission_kit>/submit/v2d_submit.py track1.parquet --track track_1 -m "<message>" --wait
```

## Submissions and public scores

All values are lower-is-better public leaderboard scores. Ranks come from the downloaded public leaderboards at the end of the session, with 57-58 teams.

| Commit | Human / Object smoothing (frames) | CD-H | CD-O | ACC-H | ACC-O | PEN | Parquet SHA-256 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1cf306b` | none | 15.738 | 27.222 | 2.120 | 0.708 | **0.0580** | `602a92f6…007279` |
| `b1faf50` | none (same values; commit adds Object sources) | 15.738 | 27.222 | 2.120 | 0.708 | **0.0580** | `be4dbe7d…55e356` |
| `8370ac9` | 2 / 2 | 15.132 | 27.035 | 0.185 | 0.205 | 0.0595 | `7d1392a1…f6c5ac` |
| `8370ac9` | 4 / 4 | **15.040** | 26.722 | 0.141 | 0.120 | 0.0612 | `17c7ed10…ee3401` |
| `8370ac9` | 8 / 8 | 15.372 | 26.426 | 0.139 | 0.082 | 0.0633 | `c14f9d51…b37422` |
| `8370ac9` | 12 / 12 | 16.558 | 26.705 | 0.141 | 0.075 | 0.0660 | `1ec8db79…bdcec0` |
| `5dcc1b6` | **6 / 16 (proposed final)** | 15.087 | **26.080** | **0.1386** | 0.0735 | 0.0723 | `32f7c612…2ec300` |
| `5dcc1b6` | 6 / 24 | 15.087 | 26.283 | **0.1386** | **0.0734** | 0.0774 | `319c3ad7…1b9b55` |

Best public rank per metric: CD-H 37/58, CD-O 38/57, ACC-H 26/57, ACC-O 25/57, PEN 55/57. The first round ranked 48, 40, 57, 51 and 55. The proposed final submission IDs are 57951767 (CD-H), 57951770 (CD-O), 57951775 (ACC-H), 57951780 (ACC-O) and 57951786 (PEN).

## Producer checks

| Check | Result |
| --- | --- |
| Assembly | 30/30 episodes; timelines, handoff hashes, rotations and baked object scale checked |
| Shared-scene screen | 0 failures; per-episode median hip residual 0.8-3.8 cm, worst p95 21 cm (episode 11); same MHR model hash on both sides |
| Kit packing and reload | Every candidate passed: exact sample row set, one commit URL, exact parameter round trip, unchanged mesh extents |
| Kit mesh warning | Episode 15's object is a box-like primitive surface |
| Unit checks | `tests/test_assemble_track1_submission.py`: 9 passed. The full CI allowlist passed except the two Node-dependent object-catalog tests, because Node.js is not installed on this host |

Final-candidate reports: [assembly](checks/assembly-5dcc1b6-h6-o16.json) and [Parquet verification](checks/verify-5dcc1b6-h6-o16.json). The Parquet files and Kaggle receipts stay local under the ignored `data/submission-inputs/`.

## Limits and remaining work

- The smoothing widths were tuned on public scores, so they may overfit the public split. Spikes are spread over neighbouring frames rather than rejected, and edge frames flatten. PEN worsens as smoothing grows.
- CD-O and PEN remain far behind the public medians (23.5 and 0.0002). Object orientation, size and hand placement need real reconstruction work; separating the hands from the object only to lower PEN stays prohibited.
- Kaggle auto-selection keeps each leaderboard's best public submission unless final submissions are chosen on each competition page. Select the `5dcc1b6` 6 / 16 entry, or keep the best entry per metric. In either case, keep every artifact behind the selected submissions for the container.
- The Object base commit `e08086c` must be pushed by its owner, so that the submitted commit's Object code is complete.
