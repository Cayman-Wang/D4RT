# D4RT Research Workspace Layout

`D4RT/research/` is the repo-specific research workspace for the D4RT baseline, historical milestones, and any D4RT-only continuation work.

## Current Entry Points
- Repo-specific active plan: `D4RT/research/plans/ACTIVE_PLAN.md`
- Repo-specific main plan: `D4RT/research/plans/d4rt_static_dynamic_separation/master_plan_zh.md`
- Workspace-level current Route A plan lives at root `research/`

## Role in the Dual-Workspace Setup
- Use root `research/` for the current system-level Route A plan and future cross-model decisions.
- Use `D4RT/research/` when the task is specifically about D4RT history, D4RT implementation continuation, or repo-scoped audits and retrospectives.
- This workspace is the canonical home for the old D4RT-centered research documents copied from the root workspace.

## Directory Layout
- `research/plans/`: D4RT-specific active plans, milestone plans, and bootstrap prompts.
- `research/guides/`: D4RT-specific guides and runbooks.
- `research/reviews/`: repo-scoped audits and fix prompts.
- `research/handoffs/`: D4RT session handoff notes.
- `research/retrospectives/`: D4RT milestone retrospectives.
- `research/artifacts/`: optional kept artifacts; temporary files go in `research/artifacts/tmp/`.

## Retention Rule
- Runtime outputs should be written to `D4RT/outputs/` by default.
- Do not store temporary replay or training artifacts under reviews or handoffs.
- If a temporary artifact is needed, place it under `D4RT/research/artifacts/tmp/` and clean it after use.
