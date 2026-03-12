# Session Handoff

Date: 2026-03-10
Synced: 2026-03-11
Workspace: `/home/grasp/Desktop/wym-project/4d-gaussgym`
Primary repo: `D4RT`
Branch: `feat/分离点云`

## 1. Current Goal
- Keep the D4RT static/dynamic separation roadmap, but treat `M3a Mesh Smoke` only as a health-check baseline.
- Sync static-mesh research framing to `S0 / S1a / S1b / S2a / S2b`:
  - `S0` = replay -> mesh smoke baseline
  - `S1a` = `replay static points -> volumetric/static mesh`
  - `S1b` = replay-based surface refinement after gate
  - `S2a` = `raw RGBD -> dense static mesh` low-cost oracle
  - `S2b` = high-cost upper-bound / paper-baseline line
- Current practical focus is `S1a + S2a`, not `M3b Mesh Quality`.

## 2. Locked Decisions
- `/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT` is the only authority repo.
- `/mnt/windows_data2/wym-project/paper/D4RT` is historical output reference only.
- `PointOdysseyDataset + PointOdysseyDataModule` is the only production path for train/test/export.
- `D4RTDataset + D4RTDataModule` remains in-tree as legacy/baseline compatibility only.
- Query GT semantics stay in `t_cam` during training; world-frame conversion happens only at `scripts/export_separation_stream.py`.
- `mesh_builder` must consume replay outputs (`summary.json + frames/frame_*.npz`) instead of dataloader batches.
- Static mesh research is now split into `S0 / S1a / S1b / S2a / S2b`.
- `S2a` is diagnostic only and does not replace the production mesh contract.
- `S2b` is upper-bound / paper-baseline only and does not enter the current milestone.
- `S1b` and `M3b Mesh Quality` remain gated by `D4RT/pointcloud_result_guide_zh.md` “情况 C”.
- External-project roles are fixed for now:
  - `Voxblox` = `S1a` core engineering reference
  - `GO-Surf` = `S2b` offline high-fidelity upper bound
  - `BundleFusion` = `S2b` classical control / paper baseline
  - `BundleSDF` = object-level RGBD reconstruction supplement only
  - `4DTAM / GauSTAR / DynaSurfGS / dynsurf` = future-route pool only

## 3. Documents Updated
- `research/plans/ACTIVE_PLAN.md`
- `research/plans/d4rt_static_dynamic_separation/master_plan_zh.md`
- `research/plans/d4rt_static_dynamic_separation/m3_bootstrap_plan_zh.md`
- `research/guides/d4rt_static_mesh_research_reorder_zh.md`
- `research/guides/reference_projects_mesh_static_dynamic_zh.md`
- `research/guides/reference_projects_mesh_static_dynamic_index.csv`
- `research/guides/d4rt_repro_static_mesh_guide_zh.md`
- `research/reviews/m2_audit_report.md`
- `research/retrospectives/d4rt_static_dynamic_separation/2026-03-10_m2_5_alignment_and_authority_replay_zh.md`

## 4. Authority Replay Evidence
Authority output paths now available:
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/separation_stream.npz`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/summary.json`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames/frame_*.npz`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/mesh_smoke/mesh_summary.json`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/mesh_smoke/meshes/**/*.ply`

Recommended rebuild commands:

```bash
cd /home/grasp/Desktop/wym-project/4d-gaussgym/D4RT

PYTHONPATH=. /home/grasp/miniconda3/envs/d4rt/bin/python -m pytest \
  d4rt/tests/test_motion_score.py \
  d4rt/tests/test_instance_tracker.py \
  d4rt/tests/test_io_contract.py \
  d4rt/tests/test_replay_cli_dry_mode.py \
  d4rt/tests/test_test_d4rt_arg_guard.py -q

python scripts/export_separation_stream.py \
  --test_data_path /home/grasp/Desktop/wym-project/4d-gaussgym/datasets/PointOdyssey \
  --test_dset val \
  --ckpt logs/recheck_trainonly/lightning_logs/version_0/checkpoints/epoch=0-step=230.ckpt \
  --output_npz outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/separation_stream.npz \
  --img_size 128 \
  --S 4 \
  --N 96 \
  --num_queries 96 \
  --strides 4 \
  --clip_step 16 \
  --max_clips 64 \
  --batch_size 1 \
  --num_workers 2 \
  --device auto

python scripts/run_separation_replay.py \
  --input_npz outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/separation_stream.npz \
  --output_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full

python scripts/build_separation_meshes.py \
  --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames \
  --output_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/mesh_smoke
```

## 5. Current Assessment
Current observed stats:
- `processed_frames = 256`
- `total_static_points = 3734`
- `total_dynamic_points = 1549`
- `total_uncertain_points = 2141`
- `mean_active_tracks = 0.0`
- `mesh_smoke.exported_static_meshes = 128`
- `mesh_smoke.exported_dynamic_meshes = 255`

Interpretation:
- `mesh_smoke` proves the `S0 / M3a` chain is alive.
- Authority replay is still below “情况 C”, so this does **not** justify entering `S1b` or `M3b Mesh Quality`.
- The right next comparison is `S1a` vs `S2a`, using the same mesher parameters.
- If `S2a` is clearly better than `S1a`, focus on replay point quality / dynamic pollution rather than mesher tuning.

## 6. Next Executable Step
1. Keep `mesh_smoke` as the current `S0 / M3a` baseline.
2. Pick one PointOdyssey sequence and one fixed time window.
3. Produce three outputs with the same mesher parameters:
   - `S0`: current smoke mesh
   - `S1a`: replay static mesh baseline
   - `S2a`: raw RGBD static mesh low-cost oracle
4. Compare four dimensions only:
   - large-structure continuity
   - hole count
   - ghosting / trailing
   - local surface smoothness
5. Do not enter `S1b` or `M3b Mesh Quality` until authority replay reaches “情况 C”.
