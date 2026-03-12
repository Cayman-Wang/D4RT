# D4RT Repository Guidelines

## Repo Role
- This directory is the D4RT authority repo plus its repo-specific research workspace.
- Workspace-level current planning lives at root `research/`; D4RT-specific historical and continuation planning lives at local `research/` within this repo.

## Session Bootstrap Protocol (Mandatory)
- Scope: every new chat started with work primarily inside the `D4RT/` tree.
- Before proposing or executing work for this repo, always load:
  1) `research/plans/ACTIVE_PLAN.md` relative to the `D4RT/` repo root
  2) all files listed under `must_read` in that `ACTIVE_PLAN.md`
- The first assistant response in a new D4RT-scoped chat must include exactly these three intent lines:
  - `Goal: ...`
  - `Current Milestone: ...`
  - `Next Action: ...`
- Treat the D4RT-local `must_read` list as the canonical repo-specific context entry list.
- If any required plan file is missing or inconsistent, stop and report the issue first.

## Research Workspace Note
- Local `research/` under this repo is the canonical D4RT history and continuation workspace.
- Use root `research/` only when the task is primarily about Route A, cross-model planning, or future Gaussian-first exploration.
