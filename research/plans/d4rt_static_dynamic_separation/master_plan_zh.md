# 基于 D4RT 的动静点云分离与 GaussGym 动态交互接入方案（静态 Mesh 重排同步版）

## 一、简要总结
- 总目标不变：在尽量少改动 D4RT 的前提下，实现**实例级**静/动态点云分离，并逐步接入 GaussGym 的动态场景交互链路。
- 当前优先级仍不是直接追高质量 mesh，而是保持 `M3a Mesh Smoke` 作为链路健康检查，同时把静态 mesh 研究固定为 `S0 / S1a / S1b / S2a / S2b` 双基线分层推进。
- 执行原则保持不变：轻改 D4RT、实例级分离、在线目标 2–5Hz、动态碰撞采用**分阶段上线**（先视觉，再关键实例碰撞）。

## 二、当前基线与问题收敛
- M1/M2 核心问题已收敛：`train.py` 已优先消费 dataset 提供的 query-level GT，`scripts/test_d4rt.py` 已加入 `num_queries <= N` 校验，`run_separation_replay.py` 已补齐 `--dry_run` 路径。
- `d4rt/separation/` 现有 `motion_score.py`、`instance_tracker.py`、`io_contract.py` 已形成可回放的 M2 基线，不需要回退重做。
- 当前真正的阻塞不是分离逻辑本身，而是三项工程对齐问题：
  - `research` 绑定的权威代码仓库是 `/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT`，而不是 `paper/D4RT`。
  - formal 输出证据此前主要存在于 `paper/D4RT/outputs/...`，未在权威仓库内重建。
  - `d4rt/data/dataset.py` 同时保留 `D4RTDataset` 和 `PointOdysseyDataset` 两套语义，必须明确哪条是生产路径。
- 结合新拉取项目后的新结论已经固定：
  - `Voxblox` 升级为 `S1a` 核心工程参考。
  - `GO-Surf` 保留为 `S2b` 离线高保真上界，不进入当前主线。
  - `BundleFusion` 保留为 `S2b` 经典控制组 / 论文基线，不作为第一波复现对象。
  - `BundleSDF` 降为对象级 RGBD 重建补充参考，不与静态场景 dense mesh 主对照线并列。
  - `4DTAM / GauSTAR / DynaSurfGS / dynsurf` 继续冻结在 future route（未来路线池）。

## 三、数据路径与语义冻结（M2.5 冻结项）
### 3.1 仓库角色
- `/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT` 是唯一权威代码仓库，里程碑判断、handoff、retrospective、formal 输出均以此为准。
- `/mnt/windows_data2/wym-project/paper/D4RT` 只保留为历史输出参考，不再作为正式基线来源。

### 3.2 Dataset / DataModule 主路径
- `PointOdysseyDataset + PointOdysseyDataModule` 是训练、测试、导出的唯一生产路径。
- `D4RTDataset + D4RTDataModule` 仅保留为 legacy/baseline 兼容路径；后续 plan/review/handoff 默认不再把它当作里程碑验收前提。

### 3.3 Query GT 与坐标语义
- 训练期 query-level GT 默认字段固定为：`gt_3d / gt_motion / gt_2d_tgt / gt_visibility_tgt`。
- 训练期 `gt_3d` 继续保持当前 `t_cam` 相机坐标语义；当前阶段不做训练链路到 world-frame 的全量重构。
- world-frame 统一只在 `scripts/export_separation_stream.py` 的导出边界执行，用于生成 replay / mesh / 可视化共享的 world-frame 契约。

### 3.4 Mesh 输入边界
- 正式生产链路的 mesh 输入边界保持不变：`mesh_builder` 只消费 `scripts/run_separation_replay.py` 产出的 `summary.json + frames/frame_*.npz`。
- `mesh_builder` 不直接消费 dataloader batch，不与训练期 `t_cam` 语义耦合。
- `raw RGBD -> dense static mesh` 只作为研究诊断线，不替代生产契约。

## 四、实现方案（保留主线，重排顺序）
### 4.1 分离主算法（实例级）
- 输入：每帧 D4RT 查询点预测（3D、motion、visibility、confidence、normal）+ 相机位姿。
- 动态分数默认：`S = 0.45*轨迹离散 + 0.35*去自运动后的流残差 + 0.20*局部占据不稳定度`。
- 点级判定阈值：`S>=0.55` 动态，`S<=0.35` 静态，中间区间延迟决策；仅对 `confidence>=0.6 && visibility>=0.5` 的点做硬判定。
- 实例聚类：DBSCAN（`eps=0.25m, min_samples=30`）+ 时序 Hungarian 关联（中心距离+点云 IoU）。

### 4.2 M2.5 对齐阶段
- 在权威仓库内重建一次 formal replay 证据，完成 `separation_stream.npz -> replay_full/summary.json -> frames/frame_*.npz` 闭环。
- 使用 `D4RT/pointcloud_result_guide_zh.md` 做门禁判定：
  - 若为“情况 A/B”，继续训练或提高正式导出规格，不进入 `S1b` 或 `M3b Mesh Quality`。
  - 即使仍处于“情况 A/B”，也允许保留 `S0 / M3a` 作为链路健康检查，并允许少量运行 `S2a` 作为诊断对照。
  - 只有达到“情况 C”，才允许把重点转向 `S1b` 和 `M3b Mesh Quality`。

### 4.3 `S0 / M3a`：Mesh Smoke 最小闭环
- 目标仅是 replay 到 mesh 的最小可运行闭环。
- static 分支：跨帧融合并周期性导出 mesh。
- dynamic 分支：按实例滑窗更新并导出 mesh。
- 本阶段只验收“能生成、可读、实例连续更新”，不追求高质量表面。

### 4.4 `S1a`：`replay static points -> volumetric/static mesh`
- 这是当前静态 mesh 生产线的第一优先层。
- 目标是把 replay 的 `static_points_world` 稳健融合成基础静态 mesh，先解决输入点云密度、动态污染、融合稳定性，再谈高质量补面。
- 固定参考：`估计mesh/vdbfusion`、`/home/grasp/reference-projects/mesh_static_dynamic/voxblox`、`静动态点云分离/ERASOR`、`估计mesh/CAD-Mesher`。

### 4.5 `S1b`：`replay static points -> surface refinement`
- 只在 `S1a` 已经证明输入足够好后再做。
- 目标是补孔、修面、提升表面连续性和细节。
- 固定参考：`估计mesh/points2surf`、`估计mesh/NeuralPull`、`估计mesh/point2mesh`、`/home/grasp/reference-projects/mesh_static_dynamic/ppsurf`。
- 前提：authority replay 已达到“情况 C”，且 `S1a` 已能生成非退化的基础静态 mesh。

### 4.6 `S2a / S2b`：raw RGBD 对照线
- `S2a`：直接使用 `D4RT/rgbd_scene_guide_zh.md` 的 raw RGBD 回投与同一套本地 mesher 做 low-cost oracle，用于最低成本判责。
- `S2b`：使用 `GO-Surf` 与 `BundleFusion` 作为高成本上界线 / 论文对照；只服务于上界判断，不进入当前主线里程碑。
- `BundleSDF` 只保留为对象级 RGBD 重建补充参考，不再作为静态场景 dense mesh 主对照线。

### 4.7 固定判责逻辑
- `S2a` 好、`S1a` 差：问题在 replay 点云密度、静动态污染或分离保真，不在 mesher。
- `S1a` 好、`S1b` 才明显改善：主要瓶颈在表面拟合与补面，不在 replay 本身。
- `S1a` 和 `S2a` 都差：优先回到数据覆盖、深度质量、位姿链路，或 `pointcloud_result_guide_zh.md` 的 A/B 类问题。
- `S2a` 已经差，但 `S2b` 明显更好：问题主要在本地 RGBD 建面流程，而不是数据本体。

### 4.8 GaussGym 接入（分阶段）
- 阶段 A：动态实例仅视觉可见/可跟踪，不进物理碰撞。
- 阶段 B：关键动态实例开启碰撞体，更新频率默认 `1Hz`，视觉仍维持 `2–5Hz`。

## 五、公共接口与类型契约
### 5.1 D4RT 分离模块
- `D4RT/d4rt/separation/motion_score.py`：动态分数计算。
- `D4RT/d4rt/separation/instance_tracker.py`：实例聚类与跨帧跟踪。
- `D4RT/d4rt/separation/io_contract.py`：统一 world-frame `SeparationFrame` 契约。
- `D4RT/d4rt/separation/mesh_builder.py`：当前主要服务 `S0 / M3a` 与 `S1a`，消费 replay 产物而不是 dataloader batch。

### 5.2 D4RT 导出脚本
- `D4RT/scripts/export_separation_stream.py` 是 world-frame 边界。
- `SeparationFrame` 字段维持：
  - `timestamp`
  - `static_points_world`
  - `dynamic_points_world`
  - `dynamic_instance_ids`
  - `dynamic_scores`
  - `confidence`
  - `visibility`
  - `static_mesh_path` (optional)
  - `dynamic_meshes[{instance_id, mesh_path, pose}]`

### 5.3 训练/评测口径
- 训练、测试、导出主路径统一为 `PointOdysseyDataModule`。
- query-level GT 默认读取 `gt_3d / gt_motion / gt_2d_tgt / gt_visibility_tgt`。
- `D4RTDataset` 仅作兼容，不再作为里程碑验收语义。

### 5.4 研究比较产物契约
- `S0 / S1a / S1b / S2a / S2b` 的研究对照统一导出：
  - `points_world.ply`
  - `mesh_raw.ply`
  - `mesh_clean.ply`
  - `metrics.json`
  - `compare_notes.md`
- 第一轮 `S1a` 与 `S2a` 对照必须使用同一套 mesher 参数，避免把输入问题误判为 mesher 问题。

## 六、测试、命令与验收标准
### 6.1 固定回归命令
- `PYTHONPATH=. /home/grasp/miniconda3/envs/d4rt/bin/python -m pytest d4rt/tests/test_motion_score.py d4rt/tests/test_instance_tracker.py d4rt/tests/test_io_contract.py d4rt/tests/test_replay_cli_dry_mode.py d4rt/tests/test_test_d4rt_arg_guard.py -q`
- formal replay smoke：在权威仓库中执行 `scripts/export_separation_stream.py` + `scripts/run_separation_replay.py`，验证 `stream -> replay` 闭环。
- timeline gate：用 `scripts/visualize_separation_timeline.py` 按 `pointcloud_result_guide_zh.md` 判断是否允许进入 `S1b / M3b`。

### 6.2 当前阶段推荐比较
- 固定 1 个 PointOdyssey 序列和同一时间窗口，先只跑三份结果：
  - `S0`：当前 smoke mesh
  - `S1a`：replay static mesh 基础融合版
  - `S2a`：raw RGBD static mesh low-cost oracle
- 第一轮只看 4 个维度：大结构连续性、孔洞数量、重影/拖尾、局部表面平滑度。

### 6.3 验收口径
- `M2.5`：权威仓库内现有 M2 回归测试通过，formal replay 证据存在，`ACTIVE_PLAN`、本主计划、最新 handoff / retrospective 描述一致。
- `S0 / M3a`：`mesh_builder.py` 能消费 replay 输出并生成非空 static mesh 与 dynamic mesh；动态实例至少可连续两帧保持同一 instance id 的 mesh 更新；不破坏现有 replay / visualize 契约。
- `S1a`：在同一序列和同一窗口下，能够给出可与 `S2a` 横向比较的基础静态 mesh 结果。
- `S1b / M3b`：只有在 replay 达到“情况 C”后，才把重点转向静态连续性、动态稳定性与高质量 mesh。

## 七、实施阶段与里程碑
- M1：修正 D4RT GT/测试链路，打通稳定推理导出。已完成。
- M2：完成点级分离 + 实例跟踪（离线回放验证）。已完成。
- M2.5：冻结 authority repo、formal replay 证据、dataset/query/坐标契约。已完成并持续沿用。
- `S0 / M3a`：完成 replay -> mesh smoke 闭环。当前保留为链路健康检查。
- `S1a + S2a`：完成生产线基础建面与 low-cost oracle 对照。当前研究重点。
- `S1b / M3b`：在通过 pointcloud guide 门禁后推进表面精修与 mesh 质量。
- M4：接入 gauss_gym 视觉动态对象。
- M5：接入关键实例碰撞体并完成交互回归测试。

## 八、明确假设与默认值
- 默认输入为 RGB-D + 已知相机位姿；若位姿漂移大，先做位姿稳健化再分离。
- 默认不引入大型新框架（NeRF/SLAM 全替换）；仅轻量增补 D4RT 周边模块。
- 默认以几何+时序为主，不依赖语义类别；后续可选接入语义先验提升 instance 稳定性。
- 默认按“先保留 `S0 / M3a`，并行推进 `S1a + S2a`，达到门禁后再进入 `S1b / M3b`”的顺序执行。
