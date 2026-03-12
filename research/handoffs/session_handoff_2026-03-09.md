# Session Handoff

Date: 2026-03-09
Workspace: `/home/grasp/Desktop/wym-project/4d-gaussgym`
Primary repo: `D4RT`
Branch: `feat/分离点云`

## 1. User Goal Across This Session
- Continue validating D4RT-based static/dynamic point separation after M2.
- Move from sparse single-frame inspection to sequence-level inspection and formal non-smoke export.
- Decide whether current point-cloud quality is sufficient for mesh/M3, or whether more training/export scale is needed first.

## 2. Major Work Completed

### 2.1 Sequence-Level Visualization / Export Added
- Added `scripts/visualize_separation_sequence.py`.
- Added `d4rt/tests/test_visualize_separation_sequence.py`.
- The script supports:
  - static accumulation in world coordinates
  - dynamic accumulation modes: `latest`, `window`, `all`
  - voxel downsampling
  - export of static/dynamic/combined PLY
  - per-instance dynamic PLY export
  - summary JSON export
- `README_zh.md` was updated earlier to include this workflow.

### 2.2 Timeline Viewer Added
- Added `scripts/visualize_separation_timeline.py`.
- Added `d4rt/tests/test_visualize_separation_timeline.py`.
- Purpose: inspect `replay_full/frames/frame_*.npz` with a timeline slider instead of only viewing one frame or one accumulated PLY.
- Supports:
  - slider scrubbing over frame indices
  - `Prev` / `Next` buttons
  - keyboard left/right step-through
  - `static_mode`: `all`, `upto`, `current`, `none`
  - `dynamic_mode`: `frame`, `window`

### 2.3 Formal Export / Replay Completed for a Non-Smoke Run
- Preserved checkpoint:
  - `outputs/gpu_tune_bench4/best_nonquick_val/lightning_logs/version_0/checkpoints/epoch=0-step=132.ckpt`
- Generated formal stream output:
  - `outputs/d4rt_formal_sequence/nonquick_val64_gpu/separation_stream.npz`
- Completed replay output:
  - `outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/summary.json`
  - `outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/frames/frame_*.npz`
- Exported sequence-level PLYs:
  - `static_scene_accumulated.ply`
  - `dynamic_window_last4.ply`
  - `combined_scene_last4.ply`
  - `dynamic_all_frames.ply`
  - `combined_scene_all_dynamic.ply`
  - `dynamic_instances/instance_*.ply`

## 3. Validated Runtime / Output Evidence

### 3.1 Formal Run Statistics
Source: `outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/summary.json`

- processed frames: `512`
- total static points: `22091`
- total dynamic points: `8446`
- total uncertain points: `10935`
- mean active tracks: `0.998046875`

### 3.2 Sequence Export Statistics
Source: `outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/sequence_summary.json`

- static downsampled points: `10173`
- dynamic downsampled points with `dynamic_mode=window,last4`: `94`

Source: `outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/sequence_summary_all.json`

- static downsampled points: `10173`
- dynamic downsampled points with `dynamic_mode=all`: `6005`

### 3.3 Tests Run in This Session
- `PYTHONPATH=. /home/grasp/miniconda3/envs/d4rt/bin/python -m unittest d4rt.tests.test_visualize_separation_sequence` -> pass
- `PYTHONPATH=. /home/grasp/miniconda3/envs/d4rt/bin/python -m unittest d4rt.tests.test_visualize_separation_timeline` -> pass
- `PYTHONPATH=. /home/grasp/miniconda3/envs/d4rt/bin/python scripts/visualize_separation_timeline.py --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/frames --static_mode all --dynamic_mode window --dynamic_window 4 --backend none` -> pass

## 4. User-Facing Conclusions Reached
- Single-frame viewer is inherently sparse and is not enough to judge scene completeness.
- Sequence-level accumulated static PLY is the correct way to inspect “完整场景”.
- Timeline scrubbing over replay frames is the correct way to inspect “动态点云随时间变化”.
- Current result is good enough to validate M2-style separation / replay behavior.
- Current result is still too sparse to judge full mesh quality confidently; for M3 mesh output, stronger training or at least larger formal export settings are still recommended.

## 5. Environment Findings / Constraints
- In the Codex execution environment, GPU was not usable:
  - `torch.cuda.is_available()` was effectively unusable here
  - `nvidia-smi` / NVML initialization was unreliable
- The user’s local machine can run GPU export.
- In Codex runtime, `export_separation_stream.py` needed `--num_workers 0`; otherwise multiprocessing semaphore creation caused `PermissionError`.
- The user’s earlier “terminal flashed closed” replay issue was explained by a missing `separation_stream.npz` combined with `set -euo pipefail`.

## 6. Current Repo State / Files of Interest
- New/updated scripts:
  - `scripts/visualize_separation_sequence.py`
  - `scripts/visualize_separation_timeline.py`
- New tests:
  - `d4rt/tests/test_visualize_separation_sequence.py`
  - `d4rt/tests/test_visualize_separation_timeline.py`
- New judgement guide to read next:
  - `D4RT/pointcloud_result_guide_zh.md`

## 7. Recommended Next Step
1. If the goal is better inspection only:
   - use `visualize_separation_timeline.py` with `--static_mode all --dynamic_mode window --dynamic_window 4`
2. If the goal is M3 mesh output quality:
   - do not start mesh tuning on this sparse input first
   - rerun formal export with stronger checkpoint / training budget / larger point density
   - then evaluate mesh on the denser sequence output

## 8. Quick Resume Prompt for New Chat
Use this prompt in a new conversation:

```text
Continue from the 2026-03-09 D4RT separation visualization session.
Read these files first:
1) research/plans/ACTIVE_PLAN.md
2) research/plans/d4rt_static_dynamic_separation/master_plan_zh.md
3) research/plans/d4rt_static_dynamic_separation/m3_bootstrap_plan_zh.md
4) research/retrospectives/d4rt_static_dynamic_separation/2026-03-06_m2_retrospective_zh.md
5) research/handoffs/session_handoff_2026-03-05.md
6) research/handoffs/session_handoff_2026-03-09.md
7) D4RT/pointcloud_result_guide_zh.md

Workspace: /home/grasp/Desktop/wym-project/4d-gaussgym
Repo: /home/grasp/Desktop/wym-project/4d-gaussgym/D4RT
Branch: feat/分离点云

Current validated output:
- outputs/d4rt_formal_sequence/nonquick_val64_gpu/separation_stream.npz
- outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/summary.json
- outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/static_scene_accumulated.ply
- outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/dynamic_all_frames.ply
- outputs/d4rt_formal_sequence/nonquick_val64_gpu/replay_full/combined_scene_all_dynamic.ply

Need help with either:
- denser formal export / further training, or
- M3 mesh builder based on the current replay outputs.
```
