# Track 1 post-competition assessment, 2026-11-08

The Track 1 leaderboards closed on 2026-11-04 at 17:00 EST. This assessment records the final results, what the submitted entry was built from, the work that is not on `main`, and a prioritized improvement list. It covers `main` at `afd6e84`, the remote branches as of 2026-11-08, and the five final leaderboards downloaded with the Kaggle CLI on 2026-11-08. It does not accept any reconstruction result; [track1-requirements.md](track1-requirements.md) still governs acceptance.

## Final results

Our team is **SeanSiu2077**. Each leaderboard keeps the team's best submission for that metric. All values are lower-is-better.

| Metric | Ours | Rank | Median | Best | Our best entry on this board |
| --- | ---: | ---: | ---: | ---: | --- |
| CD-H | 15.040 | 39 / 60 | 14.850 | 11.370 | `8370ac9`, Human/Object smoothing 4/4 frames |
| CD-O | 26.080 | 37 / 59 | 24.211 | 15.454 | `5dcc1b6`, 6/16 |
| ACC-H | 0.1386 | 25 / 59 | 0.1398 | 0.1319 | `5dcc1b6`, 6/16 and 6/24 |
| ACC-O | 0.0734 | 27 / 59 | 0.0740 | 0.0669 | `5dcc1b6`, 6/24 |
| PEN | 0.0580 | 57 / 59 | 0.00008 | 0 | `1cf306b` / `b1faf50`, no smoothing |

- The acceleration metrics are at the median. They moved there only after Gaussian smoothing; the first unsmoothed entry ranked 57 and 51.
- The geometry metrics are slightly worse than the median.
- PEN is the largest gap, about 700 times the median.
- No single submission holds all five best values. The coherent `5dcc1b6` 6/16 entry scores 15.087, 26.080, 0.1386, 0.0735 and 0.0723.
- For the latest submissions listed on the CD-H, CD-O, ACC-H and ACC-O boards, the private score equals the public score. The PEN submission list could not be read because the CLI login had expired.
- The downloaded leaderboards are in the ignored `work/leaderboards-final-20261108/`. Every submission, its Parquet hash and its first-round rank are in the [submission record](../deliveries/platform/track1-submission-20261103/README.md).

## How the submitted entry was built

The entry did not run through the registered six-stage pipeline. It was assembled from two published module deliveries:

1. **Human:** [sam-mediapipe-20261021](../deliveries/human/sam-mediapipe-20261021/README.md). SAM 3D Body with fixed per-clip native MHR identity, all 30 videos and 16,563 frames. Every MediaPipe hand candidate failed its transition screen, so the body-only baseline was selected.
2. **Object:** [object-temporal-track1-20261102-v1](../deliveries/object/object-temporal-track1-20261102-v1/README.md). Metric meshes and per-frame poses in the OpenCV camera frame, with the scale baked in.
3. **Assembly:** [tools/assemble_track1_submission.py](../tools/assemble_track1_submission.py). It reads the selected Human parameters through HTTP byte ranges, checks the Object handoff hashes, and screens Human/Object scene consistency. It applies optional Gaussian smoothing, runs the official packer, and verifies the Parquet. See [development.md](development.md#official-track-1-submission-files).

The registered backends in `src/v2hoi/stages/` are still fake, and `refine` passes its inputs through. The temporal smoothing lives in the assembler as a stopgap. Its widths were chosen from public leaderboard feedback, so they are development evidence, not an independent validation result.

## What can be checked without hidden references

PEN uses only our own prediction. In the kit's `metric_code/track_1/PEN.py`:

1. It decodes our MHR hand vertices and moves them into our object's local frame.
2. It measures how deep each vertex sits inside our object mesh, using winding numbers.
3. It averages that depth in centimetres over all hand vertices, then over frames, then over episodes.

The reference enters only through the scale of the first-frame human Sim(3), which is close to 1 when our human scale is right. A local PEN check is therefore a close proxy for the official value. One gap remains: the participant kit omits the hand-vertex role indices, so a local version must define the hand vertices from the MHR model.

CD-H, CD-O, ACC-H and ACC-O need the hidden reference trajectories. Locally, only proxies are available: image reprojection, Human/Object consistency screens, and prediction-only motion diagnostics. None of these replaces the reference-relative definitions in R3 and R4.

## Work that is not on `main`

| Branch | State | Content |
| --- | --- | --- |
| `human` | 12 commits after merged PR #9 (2026-10-23 to 2026-11-01), no PR, about 680 changed files | Research candidates only, none accepted or merged:<br>- [Hand continuity pilots](https://github.com/Seanwilliam2077/video-to-hoi/tree/human/deliveries/human/hand-continuity-pilots-20261023): bounded hand transitions.<br>- [HandFlow attempt](https://github.com/Seanwilliam2077/video-to-hoi/tree/human/deliveries/human/handflow-local-attempt-20261023): blocked by the missing MANO models.<br>- [EgoHand pilots](https://github.com/Seanwilliam2077/video-to-hoi/tree/human/deliveries/human/egohand-refined-pilots-20261031): partial improvement, no overall win over MediaPipe.<br>- [EP16 contact A/B](https://github.com/Seanwilliam2077/video-to-hoi/tree/human/deliveries/human/ep16-contact-ab-20261031): failed its motion and object-skin guards.<br>- [EP16 occluded-hand depth A/B](https://github.com/Seanwilliam2077/video-to-hoi/tree/human/deliveries/human/ep16-occluded-depth-ab-20261101): modest conditional skin-depth gain with residual deep penetration. |
| `physics-dynamic-contact-topology-v4` | 2 commits, no PR, about 630 changed files | EP16 frames 90-209, dynamic contact topology pilot v4. Mean sampled penetration fell from 26.3 to 19.5 mm, but maximum penetration and object acceleration got worse. The report concludes that the pilot does not reach the target. |
| Object base commit `e08086c` | Not on GitHub | The Object runtime changes are committed under the Object delivery, but the base checkout they patch is still only on the producer's machine. |

These branches are already merged into `main` and have no commits of their own: `codex/m3-object-preview`, `windows-cari4d`, `codex/zip-assets-preparation` and `platform-track1-submission-assembly`. The `object`, `physics` and `platform` work-area references have no commits of their own and were last changed on 2026-09-27. Deleting any of them needs the owner's approval.

Both research branches carry large evidence trees. Before merging, move large outputs to a result-specific Release as [deliveries/README.md](../deliveries/README.md) requires, and keep only the records and small checks in Git.

## Improvement priorities

Order follows the size of the gap and how directly it can be checked.

1. **Local PEN check (Platform, Physics).** Port the kit's penetration measurement to a tested tool that runs on the assembled episode files. Report per-episode and per-frame values to find where our hands sit inside our objects. Use independent synthetic fixtures for the tests, and record the hand-vertex definition as a known difference from the kit.
2. **Contact-aware joint refinement (Physics, Human).** Fix the worst PEN intervals using video-supported contact and hand placement. Start from the EP16 pilots on `human` and the topology pilot. Do not move hands out of the object only to lower PEN: under P4.2 in [implementation-plan.md](implementation-plan.md), low penetration alone cannot promote a candidate. Smoothing raised PEN from 0.0580 to 0.0723, so compare every candidate against both the unsmoothed and the smoothed baselines.
3. **Object geometry (Object).** These issues are already flagged in review:
   - ring motion
   - desk orientation
   - block landing
   - EP15's box-like primitive surface

   The Human/Object scene screen also shows scale disagreement: hip residual medians of 0.8-3.8 cm, and a worst p95 of 21 cm on EP11. CD-O is scored after the human-derived alignment, so human scale errors also move the object.
4. **Human shape and hands (Human).** The submission used the fixed-body baseline because 544 of 587 hand transitions failed the screen. The specialist hand models (HaMeR, HandFlow) need the MANO models, which are not yet available to the owner.
5. **Temporal refinement in the pipeline (Physics, Platform).** Move the smoothing from the assembler into `refine`, make it respect contact changes, and stop tuning widths on the leaderboard.
6. **Pipeline integration (Platform).** Connect the real Human and Object backends to the stage pipeline, so that a submission is produced by one reproducible run rather than by assembling separate deliveries. Then update [track1-compliance.md](track1-compliance.md) against R1-R6.

## Housekeeping

- Push `e08086c` so that the submitted commit's Object code is complete.
- Decide whether to merge, archive or close the `human` and `physics-dynamic-contact-topology-v4` work, and delete merged branches only with the owner's approval.
- Back up `data/submission-inputs/` (Parquet files, receipts, verification reports) and `work/leaderboards-final-20261108/`. Both are ignored by Git.
- Check on the competition pages whether late submissions are accepted. Late submissions do not change the final ranking. The local Kaggle CLI login has expired; run `kaggle auth login` before using it again.
- Install Node.js locally to run the two Node-dependent object catalog tests from the CI list.
- The online status page needs republishing. Its old address now redirects to the Team hub, and the hub's status section was last updated on 2026-09-27.
- `deliveries/physics/` has no delivery records yet. The physics pilot exists only on its branch.
