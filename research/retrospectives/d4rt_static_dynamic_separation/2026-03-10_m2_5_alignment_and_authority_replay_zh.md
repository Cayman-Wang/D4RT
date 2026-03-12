# M2.5 复盘：权威仓库对齐、formal replay 证据重建与静态 Mesh 研究重排同步

- 日期：2026-03-10
- 同步更新：2026-03-11
- 阶段：M2.5（仓库 / 证据 / 数据契约冻结）
- 对应主计划：`research/plans/d4rt_static_dynamic_separation/master_plan_zh.md`

## 1. 问题
在 M2 功能已经可用后，推进 M3 前暴露出两层问题：

### 1.1 第一层：工程对齐问题
1. `research` 文档绑定的权威仓库是 `/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT`，但历史讨论和运行产物有一部分位于 `paper/D4RT`，导致“代码状态”和“证据状态”分叉。
2. `d4rt/data/dataset.py` 同时存在 `D4RTDataset` 与 `PointOdysseyDataset` 两条采样路径，容易把 legacy 路径误当成生产路径。
3. 原 `ACTIVE_PLAN` 直接把 `mesh_builder.py` 设为下一步，但权威仓库内尚未重建 formal replay 证据，也没有把进入 M3 的门禁写死。

### 1.2 第二层：研究节奏问题
即使 authority replay 已经重建，仍然存在研究组织上的混淆：
1. 容易把 `M3a Mesh Smoke` 当成“高质量 mesh 已经可以推进”的信号。
2. 容易把 `replay static points -> static mesh` 和 `raw RGBD -> dense static mesh` 混成同一条链路。
3. 容易在 replay 仍处于 `pointcloud_result_guide_zh.md` 的“情况 A/B”时，过早投入高成本 mesher 或动态表面路线。

## 2. 决策
采用“先对齐，再分层推进静态 mesh”的重排策略：
1. 将当时里程碑从 `M3 准备阶段` 调整为 `M2.5 对齐阶段`，先冻结 authority repo、formal replay 证据、dataset/query/坐标契约。
2. 明确 `/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT` 是唯一权威代码仓库；`paper/D4RT` 仅保留为历史输出参考。
3. 冻结 `PointOdysseyDataset + PointOdysseyDataModule` 为训练、测试、导出的唯一生产路径；`D4RTDataset + D4RTDataModule` 标记为 legacy。
4. 冻结 query GT 语义：训练期保持 `t_cam` 相机坐标，world-frame 统一只在 `scripts/export_separation_stream.py` 的导出边界执行。
5. 保留 `M3a Mesh Smoke / M3b Mesh Quality` 的工程里程碑划分，但把静态 mesh 研究进一步固定为：
   - `S0`：当前 smoke mesh，只做链路健康检查
   - `S1a`：`replay static points -> volumetric/static mesh`
   - `S1b`：replay-based surface refinement
   - `S2a`：`raw RGBD -> dense static mesh` low-cost oracle
   - `S2b`：高成本上界线 / 论文对照
6. 固定门禁：
   - authority replay 若为“情况 A/B”，允许保留 `S0 / M3a` 和少量 `S2a` 诊断，但不进入 `S1b / M3b`
   - 只有达到“情况 C”，才允许把重点从输入质量转向高质量 mesh
7. 固定新拉取项目的角色：
   - `Voxblox` 升级为 `S1a` 核心工程参考
   - `GO-Surf` 固定为 `S2b` 离线高保真上界
   - `BundleFusion` 固定为 `S2b` 经典控制组 / 论文基线
   - `BundleSDF` 降为对象级 RGBD 重建补充参考
   - `4DTAM / GauSTAR / DynaSurfGS / dynsurf` 保持 future route（未来路线池）

## 3. 落地内容
1. 更新 `research/plans/ACTIVE_PLAN.md`：
   - 维持当前里程碑为 `M3a Mesh Smoke`
   - 将静态 mesh 研究写死为 `S0 / S1a / S1b / S2a / S2b`
   - 明确 `S2a` 仅作诊断，`S2b` 仅作上界/论文对照
2. 重写主计划 `research/plans/d4rt_static_dynamic_separation/master_plan_zh.md`：
   - 保留 authority replay 与 world-frame 契约
   - 新增 `S1a / S1b / S2a / S2b` 的位置与判责逻辑
3. 收紧 `research/plans/d4rt_static_dynamic_separation/m3_bootstrap_plan_zh.md`：
   - 明确它只覆盖 `S0 / M3a`
   - 明确 smoke 成功不等于 mesh 质量成功
4. 新增重排指南 `research/guides/d4rt_static_mesh_research_reorder_zh.md`：
   - 固定分层、优先级、统一产物接口、执行顺序
5. 更新参考对照：
   - `research/guides/reference_projects_mesh_static_dynamic_zh.md`
   - `research/guides/reference_projects_mesh_static_dynamic_index.csv`
6. 更新 `research/guides/d4rt_repro_static_mesh_guide_zh.md`：
   - 把当前默认执行顺序同步为 `S0 -> S1a + S2a -> S1b / S2b`

## 4. 证据与当前状态
权威仓库内的 formal 证据现已具备：
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/separation_stream.npz`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/summary.json`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames/frame_*.npz`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/mesh_smoke/mesh_summary.json`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/mesh_smoke/meshes/**/*.ply`

当前关键统计：
- `processed_frames = 256`
- `total_static_points = 3734`
- `total_dynamic_points = 1549`
- `total_uncertain_points = 2141`
- `mean_active_tracks = 0.0`
- `mesh_smoke.exported_static_meshes = 128`
- `mesh_smoke.exported_dynamic_meshes = 255`

当前判断：
- `stream -> replay -> mesh_smoke` 已证明 `S0 / M3a` 链路可运行。
- authority replay 仍未达到“情况 C”，因此不能把 `mesh_smoke` 的成功解读为高质量 mesh 阶段可立即启动。
- 当前最合理的研究重点应从“直接追 `M3b`”转为“并行比较 `S1a + S2a`，先做判责”。

## 5. 经验教训
1. **必须区分工程 smoke 与质量研究**：`M3a` 是链路健康检查，不是质量里程碑。
2. **必须区分生产线与对照线**：`replay static points` 和 `raw RGBD` 回答的问题不同，不能混用。
3. **必须先判责，再调 mesher**：如果 `S2a` 已明显优于 `S1a`，继续堆 mesher 多半无效。
4. **参考项目的价值主要在角色，不在是否整仓迁移**：当前更需要借“工程边界、判责方式、模块思想”，而不是把每个项目都变成短期复现目标。

## 6. 下一步
1. 保留 `S0 / M3a` 作为当前 baseline。
2. 固定 1 个 PointOdyssey 序列和同一时间窗口。
3. 产出三份结果：
   - `S0`：当前 smoke mesh
   - `S1a`：replay static mesh 基础融合版
   - `S2a`：raw RGBD static mesh low-cost oracle
4. 第一轮必须使用同一套 mesher 参数，只看四个维度：
   - 大结构连续性
   - 孔洞数量
   - 重影 / 拖尾
   - 局部表面平滑度
5. 只有在 authority replay 达到“情况 C”后，才允许进入 `S1b` 或 `M3b Mesh Quality`。
