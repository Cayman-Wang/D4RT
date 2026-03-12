# ACTIVE_PLAN

goal: 基于 D4RT 完成静/动态点云分离，并接入 GaussGym 的动态场景交互链路
current_milestone: M3a Mesh Smoke（authority replay 已重建，M3b 门禁未通过）
must_read:
  - research/plans/d4rt_static_dynamic_separation/master_plan_zh.md
  - research/plans/d4rt_static_dynamic_separation/m3_bootstrap_plan_zh.md
  - research/retrospectives/d4rt_static_dynamic_separation/2026-03-10_m2_5_alignment_and_authority_replay_zh.md
  - research/retrospectives/d4rt_static_dynamic_separation/2026-03-06_m2_retrospective_zh.md
  - research/handoffs/session_handoff_2026-03-10.md
  - research/reviews/m2_audit_report.md
  - pointcloud_result_guide_zh.md
  - research/guides/d4rt_static_mesh_research_reorder_zh.md
locked_decisions:
  - 采用轻改 D4RT，不做全框架替换
  - 实例级静/动态分离，在线目标 2–5Hz
  - 本工作区是 D4RT repo-specific 冻结与延续入口；系统级当前主线已迁移到根目录 `research/`
  - `/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT` 是唯一权威代码仓库
  - `PointOdysseyDataset + PointOdysseyDataModule` 是训练/测试/导出的唯一生产路径
  - `D4RTDataset + D4RTDataModule` 仅保留为 legacy/baseline 兼容路径
  - 训练期 query GT 保持 `t_cam` 相机坐标语义；world-frame 仅在 `scripts/export_separation_stream.py` 边界统一转换
  - `mesh_builder` 只消费 replay 产物 `summary.json + frames/frame_*.npz`
  - 分阶段上线：先做 M3a mesh smoke，再按点云结果门禁决定是否进入 M3b/M4
  - 静态 mesh 研究按 `S0 / S1a / S1b / S2a / S2b` 双基线推进；`S2a` 仅作诊断，`S2b` 仅作上界/论文对照
next_action: 保持 M3a mesh smoke 作为当前基线，优先提升训练或正式导出规格；并用 `S2a` low-cost oracle 对照 `S1a` 做判责，但只有 authority replay 达到 pointcloud guide 的“情况 C”后才进入 `S1b/M3b`
latest_retrospective: research/retrospectives/d4rt_static_dynamic_separation/2026-03-10_m2_5_alignment_and_authority_replay_zh.md
out_of_scope:
  - 本阶段不实现 M4/M5 的完整 GaussGym 交互碰撞闭环
  - 本阶段不做训练链路到 world-frame 的全量语义重构
  - `/mnt/windows_data2/wym-project/paper/D4RT` 仅作历史输出参考，不作为正式里程碑判定依据
  - 不保留 debug 运行产物，按需重生成
last_updated: 2026-03-12
