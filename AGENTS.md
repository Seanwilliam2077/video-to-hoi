# Agent instructions

These rules apply to every coding agent working in this repository (Codex, Claude Code, and others) and to people.

## Repository

- **English:** code, docs, commit messages, and PR descriptions are in English. The only exception is the Chinese status page, `docs/status.html`.
- **Track 1 only:** use only the 30 Track 1 challenge videos and their metadata. No Track 2 data or derived assets (Tier 1, Tier 2, meshes, trajectories, labels, or parameters) for development, tuning, validation, tests, or submissions. See the data policy in `docs/development.md` and `docs/workflow.md`.
- **Persistent data restriction (owner reaffirmed 2026-09-27):** never download, load, or use Track 2 content or its derivatives, including scores, statistics and cached outputs. This also prohibits Track 2 use for debugging, scorer checks, baseline normalization, candidate comparison/ranking, demos and handoff packages; there is no evaluation-only exception. Existing files, legacy scorers, old docs and upstream examples do not override this rule. Exclude unknown-provenance assets until verified. Independent synthetic fixtures must have no Track 2 provenance.
- **Checks:** run the explicit test list from `.github/workflows/tests.yml`, not a bare `pytest`.

## Branches and commits

- The owner's changes land on `main`. Teammates work on their module branches (`platform`, `human`, `object`, `physics`) and merge through pull requests.
- The local checkout can be shared between agents and may be on another agent's branch. Check `git branch --show-current` before committing. Do not switch a checkout that another agent is using; add a separate worktree for `main` instead.
- Never delete or rewrite a branch that has commits of its own without the owner's approval.

## README

Keep the README to four parts, in this order: the project introduction, the status page, the team pages, then model weights and data. Each part links to one page at most (the team hub stands for all team pages). Everything else for developers goes in `docs/development.md`.

## Team pages

The team pages live on claude.ai. They all carry the same navigation bar, in this order:

| Nav label | Page | URL |
|---|---|---|
| Team hub (团队主页) | Entry point; source `docs/hub.html`, published with `docs/pipeline.svg` as `pipeline.svg` | https://claude.ai/artifact/JAeDzfVxpRezsW73C26M4S |
| Status (项目进展) | Chinese status page; source `docs/status.html` | https://claude.ai/artifact/U2vgULC1h3DurtZwyQRHS1 |
| Reference (比赛资料) | Track 1 rules, scoring, videos, organizer Q&A | https://claude.ai/artifact/YXL887jxTTp7aQByCQvC6b |
| ① Platform | Module 1 page | https://claude.ai/artifact/A9hrz9B8qed9WWqkFWpAHg |
| ② Human | Module 2 page | https://claude.ai/artifact/3jiAK6fqoNwqGtLwDKf3pk |
| ③ Object | Module 3 page | https://claude.ai/artifact/35K4apKHN75wiUuB8qxmq8 |
| ④ Physics | Module 4 page | https://claude.ai/artifact/DUFpvGZmA2DyuU68afM7DV |

- When a page is added or a URL changes, update the navigation bar on every page, the hub, this table, and the README link if it is affected. The current page is shown as `<span aria-current="page">`, not a link.
- The reference and module pages have no source in the repository. Read the live page with the Artifact tool, edit that copy, and publish it to the same URL. Their tasks, updates and resources are stored in each page's database; change those rows instead of hard-coding content.
- English pages carry an EN/中文 switch backed by a dictionary at the end of each page. Add a Chinese entry for every new English string.

## Status page

`docs/status.html` reports progress to non-technical leadership and product managers. **Read [docs/status.md](docs/status.md) before editing it.** It is in plain Chinese with no technical names, metrics, or file formats, and its online copy has to be republished after every change.

After completing a task, record the technical result on the module's page. Then update the status page's completed items, blockers, progress, and next steps as `docs/status.md` describes.
