# M2 复盘：点级动静分离、实例跟踪与离线回放闭环

- 日期：2026-03-06
- 里程碑：M2（已完成）
- 对应主计划：[research/plans/d4rt_static_dynamic_separation/master_plan_zh.md](../../plans/d4rt_static_dynamic_separation/master_plan_zh.md)

## 1. 问题
M2 需要解决“如何把 D4RT 的查询点预测稳定分成静态/动态，并在时间上保持实例 ID 一致”的工程问题，同时要有可快速验收的回放链路。

## 2. 思路
采用“点级评分 + 阈值判定 + 中间区间滞回 + 实例聚类跟踪 + 离线回放验收”的组合：
1. 点级评分融合三项：轨迹离散、运动残差、占据不稳定度。
2. 仅对高质量点做硬判定；中间分数沿用上一帧状态，抑制抖动。
3. 动态点做 DBSCAN 聚类，再做 Hungarian 关联；无 `scipy` 时回退贪心匹配。
4. 用 `run_separation_replay.py` 做 dry-run/full-run 两阶段验证。

## 3. 实现
1. 动态评分权重与质量门限落地：见 [motion_score.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/motion_score.py:21)。
2. 三项评分融合计算：见 [motion_score.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/motion_score.py:240)。
3. 中间区间滞回（保持历史静/动态标签）：见 [motion_score.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/motion_score.py:262)。
4. DBSCAN 聚类实现（NumPy 版本）：见 [instance_tracker.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/instance_tracker.py:33)。
5. Hungarian/贪心回退匹配：见 [instance_tracker.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/instance_tracker.py:89)。
6. 回放脚本支持 dry-run 与 summary 输出：见 [run_separation_replay.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/scripts/run_separation_replay.py:54)。

## 4. 验证
1. 点级评分与阈值行为：见 [test_motion_score.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/tests/test_motion_score.py:13)。
2. 滞回状态连续性：见 [test_motion_score_hysteresis_stateful.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/tests/test_motion_score_hysteresis_stateful.py:12)。
3. 实例跟踪跨帧 ID 稳定：见 [test_instance_tracker.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/tests/test_instance_tracker.py:23)。
4. dry-run/full-run 行为正确：见 [test_replay_cli_dry_mode.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/tests/test_replay_cli_dry_mode.py:32)。
5. 会话内 smoke 结果已确认：`separation_stream_smoke.npz`、`replay_smoke_dry/summary.json`、`replay_smoke_full/frames/frame_*.npz` 都可生成，说明离线分离回放链路闭环可运行。

## 5. 局限
1. 当前是离线回放分离，不是在线 mesh 增量更新。
2. 未实现 static/dynamic 双路 mesh builder（M3 任务）。
3. 未接入 GaussGym 动态碰撞体（M4/M5 任务）。

## 6. 下一步
1. 进入 M3：在 M2 分离输出上增加 static/dynamic mesh 双路构建与导出频率控制。
2. 对接 GaussGym 时先做视觉动态对象，再做关键实例碰撞更新。

## 7. 借鉴关系与实现边界
### 7.1 借鉴层级与来源边界
本阶段对“静动态点云分离”目录的借鉴是“思路级借鉴 + 本地重写实现”，不是把外部仓库源码直接拷入 D4RT。借鉴来源与方向定义见 [master_plan_zh.md](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/research/plans/d4rt_static_dynamic_separation/master_plan_zh.md:8)。

### 7.2 具体借鉴了哪些工作（来源位置 + 借鉴点）
1. `EmerNeRF`  
来源位置：`静动态点云分离/EmerNeRF/radiance_fields/radiance_field.py`（静/动态双分支与密度融合，见 [radiance_field.py](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/EmerNeRF/radiance_fields/radiance_field.py:422)）。  
借鉴点：把“静态分量 + 动态分量”拆开建模，再做融合输出的思想，用于指导 D4RT 的静/动态判别评分设计。
2. `CAD-Mesher`  
来源位置 A：`静动态点云分离/CAD-Mesher/CAD_Mesher/src/cad_mesher_node.cpp`（滑窗子图与动态点剔除，见 [cad_mesher_node.cpp](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/CAD-Mesher/CAD_Mesher/src/cad_mesher_node.cpp:1020)）。  
来源位置 B：`静动态点云分离/CAD-Mesher/CAD_Mesher/src/map.cpp`（稳定度 odds 更新，见 [map.cpp](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/CAD-Mesher/CAD_Mesher/src/map.cpp:1532)）。  
借鉴点：动态剔除不是单帧阈值，而是滑窗 + 历史稳定度统计。
3. `ERASOR`  
来源位置：`静动态点云分离/ERASOR/src/offline_map_updater/src/erasor.cpp`（scan ratio test 与地面恢复，见 [erasor.cpp](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/ERASOR/src/offline_map_updater/src/erasor.cpp:366)）。  
借鉴点：对“占据变化”做比率判定，并在需要时恢复地面，降低动态污染静态图。
4. `DeSiRe-GS`  
来源位置：`静动态点云分离/DeSiRe-GS/separate.py`（离线分离脚本，见 [separate.py](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/DeSiRe-GS/separate.py:30)）。  
借鉴点：工程上先做“离线分离回放验证”再进入在线系统，避免直接在训练主链路里调试。
5. `DeGauss`  
来源位置 A：`静动态点云分离/DeGauss/README.md`（3DGS 背景 + 4DGS 前景解耦，见 [README.md](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/DeGauss/README.md:16)）。  
来源位置 B：`静动态点云分离/DeGauss/train.py`（coarse→fine 两阶段与前后景并行优化，见 [train.py](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/DeGauss/train.py:146)）。  
来源位置 C：`静动态点云分离/DeGauss/gaussian_renderer/__init__.py`（foreground/background 渲染通道，见 [__init__.py](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/DeGauss/gaussian_renderer/__init__.py:20)）。  
借鉴点：采用“先粗后细、分支解耦”的流程化思想，而非直接迁移其 3DGS/4DGS 实现。

### 7.3 这些借鉴在 D4RT 中具体落在了哪里
| 借鉴来源 | 借鉴到的思路 | D4RT 落点 | 当前状态 |
| --- | --- | --- | --- |
| EmerNeRF 双分支分解 | 静/动态分量分开度量再融合决策 | [motion_score.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/motion_score.py:240) 的多项评分融合 | M2 已落地 |
| CAD-Mesher 滑窗与稳定度 | 不做单帧硬分割，使用历史信息抑制抖动 | [motion_score.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/motion_score.py:262) 的滞回状态保持 | M2 已落地 |
| ERASOR 占据变化判定 | 将占据变化作为动态证据之一 | [motion_score.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/motion_score.py:187) 到 [motion_score.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/motion_score.py:227) 的 occupancy instability 计算 | M2 已落地（简化版） |
| 点云聚类+关联思路 | 动态点先聚类再跨帧关联稳定 ID | [instance_tracker.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/instance_tracker.py:33)、[instance_tracker.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/separation/instance_tracker.py:89) | M2 已落地 |
| DeSiRe-GS 分阶段分离工程化 | 先离线回放验收，再进入在线模块 | [run_separation_replay.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/scripts/run_separation_replay.py:54) 的 dry-run/full-run | M2 已落地 |
| CAD-Mesher 增量网格/ERASOR地面恢复 | 在线 mesh 更新、地面恢复增强 | `d4rt/separation/mesh_builder.py`（规划中） | M3+ 待落地 |

### 7.4 为何“静动态点云分离”目录里其他项目未在 M2 直接采用
1. `DS-SLAM` / `DynaSLAM` / `VDO_SLAM`：偏 SLAM 主系统替换，和“轻改 D4RT”策略冲突。  
2. `staticfusion`：目标是动态场景中的静态重建，不覆盖当前 M2 的点级分离 + 实例跟踪主链。  
3. `4D-OR` / `d2nerf`：任务域或表示层与当前 D4RT 点查询分离链路耦合较弱。  
4. 结论：这些项目用于方法对照和后续可扩展参考，不作为 M2 直接实现来源。

### 7.5 独立实现证据
2026-03-06 在 `D4RT/` 下执行关键字检索：  
`rg -n "EmerNeRF|CAD-Mesher|CAD_Mesher|ERASOR|DeSiRe|DeGauss|静动态点云分离" D4RT`  
返回空结果，说明当前实现没有这些外部项目的直接 import 或路径依赖，属于“参考方法后本地重写”。
