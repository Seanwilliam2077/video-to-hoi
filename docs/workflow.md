# Team workflow

Four people build the pipeline in parallel, one module each. `main` always runs end to end: every stage has a fake backend, so the pipeline produces a valid, scoreable submission from the start. Each person replaces their stages' fake backends with real ones and measures the change with the shared scorer. The files the stages exchange are specified in [contracts.md](contracts.md).

## Modules

The pipeline is coupled in three places, and the module boundaries follow them:

1. **The scale anchor.** The human sets the metric scale of the depth, and the depth sets the scale of the object meshes. The human module therefore also delivers a `DepthScale`.
2. **Mesh and tracking.** Choosing among candidate meshes needs the tracking, and tracking needs a mesh. Split between two people, each would wait on the other, so one module owns both.
3. **Contact refinement changes the human and the object together.** It belongs to neither of them, so it is a module of its own that runs after both.

The result is one module per group of leaderboard metrics:

| Module | Owner | Stages | Metrics | Develops on, from day one | Main risk |
|---|---|---|---|---|---|
| 1 Platform & perception | Sean | inputs, export | all, as gatekeeper | the CARI4D baseline run as is | on the critical path for the first two weeks |
| 2 Human | TBD | human | CD-H | Tier 1 videos, scored directly: the first-frame Sim(3) absorbs the world frame and the global scale, so no camera is needed | the official Sim(3) is fitted on the first frame, so a wrong first frame shifts the whole clip, objects included |
| 3 Object | TBD | objects, motion | CD-O | meshes: compared with the Tier 1 reference meshes; tracking: `--backend objects=reference`, scored with `--align first-object` | the largest workload: the hula hoop, the white desk and cart, furniture pushed with a foot |
| 4 Temporal & physics | TBD | refine | ACC-H, ACC-O, PEN (the physical axis, half the weight) | Tier 2 as input and Tier 1 as the answer: Tier 2 is the organizer's noise sampled from Track 1 error distributions, so no upstream is needed | over-smoothing costs Chamfer; Tier 2 has SOMA-X only, so the human side must later be ported to MHR |

What each module does:

- **1 Platform & perception:** the rented GPU machine; masks (GroundingDINO → SAM2), intrinsics merged per physical camera, and depth (MoGe2); the CARI4D baseline end to end, which becomes the first upstream snapshot; export, the scorer, CI; Kaggle submissions.
- **2 Human:** SAM 3D Body → MHR; one identity per clip; MHR → SOMA-X with the toolkit's `export_soma.py` for local scoring; the `DepthScale` that makes the human the metric anchor; the first frame above all.
- **3 Object:** meshes for the 10 Track 1 objects (candidates from SAM 3D Objects and Hunyuan3D-2, selection, one scale per object from the scaled depth, symmetry, the hula hoop as a torus); FoundationPose registration and tracking, a pose and a confidence on every frame.
- **4 Temporal & physics:** smoothing with the fingers on their own, static segments locked, low-confidence frames re-filled, and human–object contact (CARI4D's joint refinement and contact-guided optimization), without costing Chamfer.

Balancing the load: module 3 is the heaviest, and CARI4D already includes FoundationPose tracking, so module 3 starts with the meshes. Module 1 is busiest in weeks 1–2 and lighter afterwards, when it can take one of module 3's hard cases. Each module sets up the environment for its own models on the GPU machine: module 2 for SAM 3D Body, module 3 for SAM 3D Objects and FoundationPose.

## Data

| Set | Episodes | Use |
|---|---|---|
| `dev-mini` | Tier 1: 7 iron, 9 big red bowl, 10 white desk, 19 cane, 21 tall bar stool | every PR: a hand-held object, a symmetric one, large furniture, a thin one, and furniture dragged across the floor |
| `dev-full` | Tier 1: all 30 | weekly integration |
| Tier 2 | same 30 episodes | input for module 4 (`--backend human=tier2 motion=tier2 objects=reference`) |
| Track 1 | all 30 | what we submit; no public ground truth |

Tier 1 and Tier 2 serve development and scoring only. The development backends that read them refuse Track 1, and the scorer marks runs that use reference meshes as invalid submissions. Tuning a method's settings on Tier 2 (for example, smoothing weights) is how we read the organizer's rule, which bans Track 2 meshes, poses, and camera parameters from Track 1 reconstructions. Confirm this with the organizer. See design section 6.

## Branches and pull requests

- `main` stays green and runnable. Everything reaches it through a pull request.
- Each module works on its own branch and merges `main` into it at least once a week.
- **Contracts first.** A change to a contract (a field's meaning or shape) goes in a small PR of its own, bumps `CONTRACT_VERSION`, and is approved by the owners of the stages that read it. Implementation PRs follow.
- The scorer (`score.py`, `metrics.py`, `dataset.py`, `body.py`), `contracts.py`, and CI belong to module 1. A change to a metric bumps `SCORER_VERSION`; scores from different versions are not compared.
- The repository is in English: code, docs, commit messages, and PR descriptions.

## Merge gate

A PR that changes a stage:

1. passes CI (`pytest`; the fake pipeline runs end to end into the scorer);
2. is scored on `dev-mini`, against the current baseline for the stages it does not touch:

   ```bash
   python -m v2hoi.run --run-id <branch>-<n> --dataset tier1 --episodes 7 9 10 19 21 \
       --upstream runs/<baseline> --stages <your stages> refine export --backend <stage>=<name> --score
   ```

3. commits `runs/<run_id>/summary.json` as `benchmarks/<YYYY-MM-DD>_<branch>.json`, one file per PR so parallel PRs do not conflict;
4. improves the metrics its module owns, does not make any other metric worse by more than 2% without an explanation in the PR, and does not make the internal score worse.

The **internal score** puts the five leaderboard metrics on one scale. Each metric is divided by its Tier 2 value, each axis takes the mean of its metrics, and the two axes weigh equally, as the challenge page says. Tier 2 scores 1 and lower is better. It exists for our merge and submission decisions only; the official way of combining the five numbers is not published.

A module 3 PR scored with `--align first-object`, or a module 4 PR scored on Tier 2 input, is compared with a baseline scored the same way.

## Weekly integration

- **Tuesday:** merge the PRs that pass the gate, run `dev-full` and Track 1 with the newest backends, and pin the result as the next baseline (`runs/baseline-vN`, never overwritten). Everyone moves their `--upstream` to it.
- **Kaggle:** up to 5 submissions a week, unlimited in the last 3 days before the 2026-11-04 freeze. Only module 1 submits, after `python -m v2hoi.score --strict` passes on `dev-full`. All four people are in one Kaggle team, with the same members in all five Track 1 competitions. The challenge page joins the competitions by member usernames, and different members would split the team into several rows.

## Runs and storage

- `runs/` is not committed. Code travels through git; run artifacts live on the rented GPU machine (or a private Hugging Face dataset) and are referred to by run id.
- Name runs `<module>-<topic>-<n>`, for example `human-sam3d-2`. Baselines are `baseline-vN`.

## Open items

| Item | Module |
|---|---|
| Standalone mesh metric: shape and scale error of a generated mesh against the reference mesh, independent of any pose | 3, reviewed by 1 |
| Real backends for every stage, starting with the CARI4D baseline end to end | everyone; baseline by 1 |
| MHR versions of module 4's human refinement (Tier 2 has SOMA-X only) | 4 |
| Per-module CODEOWNERS and required reviews on `main`, once teammates are collaborators | 1 |
| Official submission format: a second export backend once `eval_reconstruction.py` is published | 1 |
| Resume keyed on input hashes (design section 3) | 1 |
