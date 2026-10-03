# Status page guide

`docs/status.html` is the project's status page. The project owner presents it to **non-technical leadership and product managers** and uses it in team meetings. Read this guide before changing the page, whether you are a person or an agent.

- Online copy: https://claude.ai/artifact/U2vgULC1h3DurtZwyQRHS1, linked from the first line of the README.
- Source of truth: `docs/status.html` on `main`. Editing the file does not update the online copy; see [Publishing](#publishing).

## Language and audience

- Write the page in **Simplified Chinese**. It is the one exception to the repository's English-only rule.
- The reader has no technical background and wants to know four things: what we are building, how far along we are, what is blocking us, and what we need from them.
- Write plain language. Say what a fact means for the project, not how it works technically.
- Keep it short. A leader should get the picture from the headline, the one-line summary, and the progress bar alone.

## Words that must not appear

The page names no models, tools, platforms, file formats, or technical metrics. Technical detail belongs in the module pages, `docs/platform.md`, and `deployment/PREPARATION_STATUS.md`.

| Kind | Examples to leave out |
|---|---|
| Model and method names | CARI4D, SAM 3D Body, SAM 3D Objects, SAM2, GroundingDINO, MoGe, FoundationPose, Hunyuan3D, MHR, SOMA-X |
| Platforms and services | Hugging Face, Google Drive, iOA, GitHub, PR numbers, Kaggle |
| Infrastructure | GPU architectures and technical specs other than memory size, CUDA, Docker, container, SSH, ZIP, Linux, Python versions, token, API, HTTP error codes such as 403 |
| Code and data terms | 权重 (weights), 源码 (source code), 依赖 (dependencies), 接口 (interfaces), stage names, file names and paths, commands, parquet, mesh |
| Metrics and dataset terms | CD-H, CD-O, ACC-H, ACC-O, PEN, Chamfer, Sim(3), Track 1 / Track 2, Tier 1 / Tier 2, episode numbers |

Words that are fine: 服务器, 显卡, 显卡内存, AI 模型, 模型文件, 使用权限, 三维模型, 打包上传, 自动检查, 比赛给的 30 段视频, 前两段视频（泡沫块、平底锅）. Servers may be called by the names the team uses for them (L20、A10、RTX PRO 5000 服务器), with the card memory in GB as the only spec; explain what that means in plain words, as the server table on the page does.

Check before you commit. The command must print nothing:

```bash
grep -nE "CARI4D|SAM ?3D|SAM2|GroundingDINO|MoGe|FoundationPose|Hunyuan|MHR|SOMA|Hugging ?Face|Google Drive|iOA|GitHub|PR #|Kaggle|GPU|CUDA|Docker|SSH|ZIP|Linux|Python|token|API|403|权重|源码|依赖|接口|parquet|CD-H|CD-O|ACC-|PEN|Chamfer|Sim\(3\)|Track ?[12]|Tier ?[12]" docs/status.html
```

## Rewriting technical facts

Keep the fact and change the wording. These rewrites are already on the page:

| Technical wording | On the page |
|---|---|
| 12 model source archives and 7 public weight groups (5.48 GB) downloaded and verified | 能公开下载的 AI 模型文件都已下载并核对（约 5.5 GB） |
| 28 Linux / Python 3.10 host dependencies (119 MB) | 服务器上要装的基础软件（约 120 MB） |
| ZIP bundling and receiver-side verification; no SSH needed | "打包上传、到达后自动核对"的工具，不需要远程登录服务器也能把文件交过去 |
| File checks and interface tests pass; GPU environment and baselines unverified | 自动检查全部通过，但只说明文件完整、各环节衔接正确；真正用视频跑出结果还没验证 |
| No SSH access; upload a ZIP and use the server console | 没法远程登录这台服务器，只能把文件打包上传，再在服务器自带的操作界面里操作 |
| Hugging Face login works, but the gated CARI4D and SAM 3D weights return 403 | 账号能正常登录模型网站，但三个模型要单独申请使用权限，光能登录还不够 |
| iOA blocks Codex Google Drive access; the user confirms manual downloads are allowed | 公司只限制自动助手访问，项目负责人可以自行下载；两组必需模型文件仍待取得 |
| The five leaderboard metrics | 准不准、动作像不像、真不真实 |

Round sizes and counts, or leave them out, when the exact number means nothing to the reader.

## Facts and honesty

- **Keep preparation and results apart.** Downloading, packaging, and passing automated checks is preparation. Only a real run on the challenge videos is a result. Never present preparation as a result.
- **No invented scores.** The challenge videos have no public answers, so we cannot compute official scores ourselves. Do not show scores or percentages that suggest otherwise.
- **Every blocker says what it needs.** Each item under "卡在哪里" ends with a "需要：" line naming who has to do what.
- **Date every update.** Update "数据截至" and the footer's "本次更新" line each time.
- **No unagreed dates.** Change a date in "时间安排" only after the team has agreed to it, and mark slipping items "有风险".
- **Data rule.** The project uses only the 30 challenge videos. On the page, say "只用比赛给的 30 段视频", not the dataset's technical name.

## Page structure

Keep these sections in this order:

0. Navigation bar (`<nav class="sitenav">`): the links shared by all team pages, labelled 团队主页 · 项目进展 · 比赛资料 · ① 平台 · ② 人体 · ③ 物体 · ④ 物理, with 项目进展 marked as the current page. Keep it at the very top and keep its links the same as on the other pages (see `AGENTS.md`, Team pages).
1. Header: a headline sentence, and the **live countdown** to the deadline (2026-11-04 17:00 US Eastern = 2026-11-05 06:00 Beijing). It shows days, hours, minutes, and seconds and updates every second. Do not replace it with a fixed number.
2. 一句话: one paragraph that says where we are and the main blockers.
3. 我们在做什么: the task in everyday terms, with the input → system → output sketch and the three scoring questions.
4. 团队分工: straight after the introduction, because meetings move from the project to who does what. It holds the hand-off diagram; one card per role (owns, delivers, needs, first two weeks, scoring share, workload, suitable background, link to the module page); what each role delivers for the first results; the working rules; and open questions about roles.
5. 进度: the six-step progress bar.
6. 已经完成的
7. 卡在哪里: each item gives what is wrong and what is needed.
8. 时间安排: milestones with a status for each.
9. 需要支持的事: the decisions and help we ask of leadership.
10. Footer: "数据截至" date, "本次更新" summary, and a link to the repository for technical detail.

When a role gets an owner, replace "待认领" with the person's name on that role's card.

## When to update

After finishing a task, record the technical result on the module page. Then update this page: "已经完成的", "卡在哪里", "进度", "时间安排", and the next steps, in plain language. Update it before a meeting with leadership as well.

## Publishing

1. Edit `docs/status.html` on `main`. The shared checkout may be on another agent's branch: check `git branch --show-current` first, and use a separate worktree for `main` if needed.
2. Run the word check above. Open the page at phone width and in both light and dark themes. Confirm that the countdown ticks.
3. Commit to `main` in English, like the rest of the repository.
4. Republish the online copy to the **same** URL. In Claude Code, publish `docs/status.html` with the Artifact tool and `url` set to the address above; a publish without `url` creates a new page. An agent that cannot publish to claude.ai tells the owner that the online copy is out of date.
