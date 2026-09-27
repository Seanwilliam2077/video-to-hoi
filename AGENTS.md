# Agent instructions

These rules apply to every coding agent working in this repository (Codex, Claude Code, and others) and to people.

## Repository

- **English:** code, docs, commit messages, and PR descriptions are in English. The only exception is the Chinese status page, `docs/status.html`.
- **Track 1 only:** use only the 30 Track 1 challenge videos and their metadata. No Track 2 data or derived assets (Tier 1, Tier 2, meshes, trajectories, labels, or parameters) for development, tuning, validation, tests, or submissions. See the README's data policy and `docs/workflow.md`.
- **Checks:** run the explicit test list from `.github/workflows/tests.yml`, not a bare `pytest`.

## Branches and commits

- The owner's changes land on `main`. Teammates work on their module branches (`platform`, `human`, `object`, `physics`) and merge through pull requests.
- The local checkout can be shared between agents and may be on another agent's branch. Check `git branch --show-current` before committing. Do not switch a checkout that another agent is using; add a separate worktree for `main` instead.
- Never delete or rewrite a branch that has commits of its own without the owner's approval.

## Status page

`docs/status.html` reports progress to non-technical leadership and product managers. **Read [docs/status.md](docs/status.md) before editing it.** It is in plain Chinese with no technical names, metrics, or file formats, and its online copy has to be republished after every change.

After completing a task, record the technical result on the module's page. Then update the status page's completed items, blockers, progress, and next steps as `docs/status.md` describes.
