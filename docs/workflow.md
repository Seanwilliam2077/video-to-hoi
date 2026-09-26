# Team workflow

Four people build the pipeline in parallel. `main` always runs end to end: every stage has a fake backend, so the pipeline produces a valid, scoreable submission from the start. Each person replaces one stage's fake backend with a real one and measures the change with the shared scorer. The files the stages exchange are specified in [contracts.md](contracts.md).

## Roles

The split follows the five leaderboard metrics, so each person can see their own effect on the score.

| Role | Owner | Stages | Metrics it moves | Starting without waiting for others |
|---|---|---|---|---|
| Platform | Sean | inputs, export; scorer, contracts, CI, compute, Kaggle submissions, CARI4D baseline | all (keeps them honest) | — |
| Human | TBD | human | CD-H, ACC-H | Run the human stage and export on Tier 1 with `--score`. The first-frame Sim(3) absorbs the world frame and the global scale, so camera intrinsics are not needed yet |
| Objects | TBD | objects | CD-O (shape and scale) | Generate meshes from Tier 1 frames; Tier 1 has reference meshes for all 29 of its objects to compare against |
| Motion | TBD | motion | CD-O (pose), ACC-O, PEN | Track with `--backend objects=reference` (the Tier 1 mesh) and score with `--align first-object`, which keeps human errors out of the object metrics |

## Data

| Set | Episodes | Use |
|---|---|---|
| `dev-mini` | Tier 1: 7 iron, 9 big red bowl, 10 white desk, 19 cane, 21 tall bar stool | every PR: a hand-held object, a symmetric one, large furniture, a thin one, and furniture dragged across the floor |
| `dev-full` | Tier 1: all 30 | weekly integration |
| Track 1 | all 30 | what we submit; no public ground truth |

Tier 1 is for scoring only. Reference meshes (`--backend objects=reference`) may be used in development runs to unblock the motion stage. They are never used in Track 1 runs, where the backend refuses, or in submissions, which the scorer flags as invalid. See design section 6.

## Branches and pull requests

- `main` stays green and runnable. Everything reaches it through a pull request.
- Each person works on one branch of their own; the platform branch is `sean_pipeline_test`. Merge `main` into your branch at least once a week.
- **Contracts first.** A change to a contract (a field's meaning or shape) goes in a small PR of its own, bumps `CONTRACT_VERSION`, and is approved by the owners of the stages that read it. Implementation PRs follow.
- The scorer (`score.py`, `metrics.py`, `dataset.py`, `body.py`), `contracts.py`, and CI belong to the platform role. A change to a metric bumps `SCORER_VERSION`; scores from different versions are not compared.
- The repository is in English: code, docs, commit messages, and PR descriptions.

## Merge gate

A PR that changes a stage:

1. passes CI (`pytest`; the fake pipeline runs end to end into the scorer);
2. is scored on `dev-mini`, against the current baseline for the stages it does not touch:

   ```bash
   python -m v2hoi.run --run-id <branch>-<n> --dataset tier1 --episodes 7 9 10 19 21 \
       --upstream runs/<baseline> --stages <your stages> export --backend <stage>=<name> --score
   ```

3. commits `runs/<run_id>/summary.json` as `benchmarks/<YYYY-MM-DD>_<branch>.json`, one file per PR so parallel PRs do not conflict;
4. improves the metrics its role owns, does not make any other metric worse by more than 2% without an explanation in the PR, and does not make the internal score worse.

The **internal score** puts the five leaderboard metrics on one scale: each is divided by its Tier 2 value, each axis takes the mean of its metrics, and the two axes weigh equally, as the challenge page says. Tier 2 scores 1 and lower is better. It exists for our merge and submission decisions only; the official way of combining the five numbers is not published.

A motion PR scored with `--align first-object` is compared with the baseline under the same alignment.

## Weekly integration

- **Tuesday:** merge the PRs that pass the gate, run `dev-full` and Track 1 with the newest backends, and pin the result as the next baseline (`runs/baseline-vN`, never overwritten). Everyone moves their `--upstream` to it.
- **Kaggle:** up to 5 submissions a week, unlimited in the last 3 days before the 2026-11-04 freeze. Only the platform role submits, after `python -m v2hoi.score --strict` passes on `dev-full`. All four people are in one Kaggle team, with the same members in all five Track 1 competitions: the challenge page joins the competitions by member usernames, and different members would split the team into several rows.

## Runs and storage

- `runs/` is not committed. Code travels through git; run artifacts live on the rented GPU machine (or a private Hugging Face dataset) and are referred to by run id.
- Name runs `<role>-<topic>-<n>`, for example `human-sam3d-2`. Baselines are `baseline-vN`.

## Open items

| Item | Role |
|---|---|
| Standalone mesh metric: shape and scale error of a generated mesh against the reference mesh, independent of any pose | objects, reviewed by platform |
| Real backends for every stage, starting with the CARI4D baseline end to end | everyone; baseline by platform |
| Per-stage CODEOWNERS and required reviews on `main`, once teammates are collaborators | platform |
| Official submission format: a second export backend once `eval_reconstruction.py` is published | platform |
| Resume keyed on input hashes (design section 3) | platform |
