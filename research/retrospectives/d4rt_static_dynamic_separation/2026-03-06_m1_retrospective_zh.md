# M1 复盘：D4RT GT/测试链路收敛与分离前置打通

- 日期：2026-03-06
- 里程碑：M1（已完成）
- 对应主计划：[research/plans/d4rt_static_dynamic_separation/master_plan_zh.md](../../plans/d4rt_static_dynamic_separation/master_plan_zh.md)

## 1. 问题
M1 的核心问题是“分离功能还不能开始前，训练/测试数据语义和测试入口要先收敛”，否则后续静/动态分离结果不可信。主要风险有两类：
1. GT 提取语义不稳定：训练侧可能混用 query 级 GT 与 depth 回推 GT。
2. 测试入口缺少参数保护：`num_queries > N` 时会在后续流程触发隐蔽错误。

## 2. 思路
采用“先收敛输入语义，再做功能扩展”的策略：
1. 训练与测试统一优先读取 dataset 提供的 query 级 GT。
2. 仅在 query 级 GT 缺失时回退到 legacy depth/intrinsics 推导。
3. 在 CLI 入口前置参数校验，尽早失败，避免进入模型后才报错。

## 3. 实现
1. 训练侧 GT 抽取重构：新增 query 优先、depth 回退的双路径逻辑，并统一 mask 语义。见 [train.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/train.py:103)。
2. 测试侧 GT 抽取与训练对齐：复用同样优先级和有效性筛选规则。见 [test.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/test.py:101)。
3. 测试脚本加入采样参数防线：`--num_queries` 必须小于等于 `--N`。见 [test_d4rt.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/scripts/test_d4rt.py:18)。
4. 数据集采样逻辑明确 query 级输出语义（含 `gt_3d/gt_motion/gt_visibility`）。见 [dataset.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/data/dataset.py:364)。

## 4. 验证
1. CLI 参数守卫测试可复现：见 [test_test_d4rt_arg_guard.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/tests/test_test_d4rt_arg_guard.py:8)。
2. smoke 训练可跑通并产出 checkpoint（1 epoch 跑通日志已在会话中确认），为 M2 的分离回放提供可用输入。

## 5. 局限
1. M1 只完成“语义和链路收敛”，不包含静/动态分离本体算法。
2. 未涉及实例级跟踪与 mesh 双路构建，仍需 M2/M3 完成。

## 6. 下一步
1. 进入 M2，落地点级动态评分、阈值分离、滞回机制与实例跟踪。
2. 提供离线回放脚本（含 dry-run）用于快速验收分离效果与数据契约。

## 7. 借鉴关系与实现边界
### 7.1 借鉴层级与来源边界
M1 对“静动态点云分离”目录中的工作，主要借鉴的是“实施顺序与工程约束”，不是直接引入分离算法源码。  
原则是：先把 D4RT 的 GT 语义和测试入口收敛，再进入 M2 的动静分离算法实现。

### 7.2 具体借鉴了哪些工作（来源位置 + 借鉴点）
1. `EmerNeRF`  
来源位置：`静动态点云分离/EmerNeRF/radiance_fields/radiance_field.py`（静/动态分支建模，见 [radiance_field.py](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/EmerNeRF/radiance_fields/radiance_field.py:422)）。  
借鉴点：静/动态分解前必须先保证监督语义一致，否则分支学习和后续分离都会漂移。
2. `CAD-Mesher`  
来源位置：`静动态点云分离/CAD-Mesher/CAD_Mesher/src/cad_mesher_node.cpp`（滑窗更新流程，见 [cad_mesher_node.cpp](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/CAD-Mesher/CAD_Mesher/src/cad_mesher_node.cpp:1020)）。  
借鉴点：时序链路依赖稳定输入语义，必须先把上游坐标/可见性/采样约束打牢。
3. `ERASOR`  
来源位置：`静动态点云分离/ERASOR/src/offline_map_updater/src/erasor.cpp`（scan ratio 与地面恢复流程，见 [erasor.cpp](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/ERASOR/src/offline_map_updater/src/erasor.cpp:366)）。  
借鉴点：做动态判别前要先定义高质量输入门槛（有效深度、可见性、有限值），否则误判会被时序累积放大。
4. `DeSiRe-GS`  
来源位置：`静动态点云分离/DeSiRe-GS/separate.py`（离线分离脚本，见 [separate.py](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/DeSiRe-GS/separate.py:30)）。  
借鉴点：先建立可独立运行的脚本化流程和参数边界，再推进主功能模块。
5. `DeGauss`  
来源位置：`静动态点云分离/DeGauss/train.py`（coarse→fine 分阶段训练，见 [train.py](/home/grasp/Desktop/wym-project/4d-gaussgym/静动态点云分离/DeGauss/train.py:146)）。  
借鉴点：采用“分阶段落地”策略，M1 专注基础链路收敛，避免在同一阶段叠加过多变化。

### 7.3 这些借鉴在 M1 中具体落在了哪里（来源映射表）
| 借鉴来源 | 借鉴到的思路 | M1 的 D4RT 落点 | 当前状态 |
| --- | --- | --- | --- |
| EmerNeRF 双分支前提 | 先统一监督语义再做分解 | 训练侧 query GT 优先 + depth 回退逻辑，[train.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/train.py:103) | M1 已落地 |
| EmerNeRF/ERASOR 的质量门槛意识 | 动态判断前先做输入质量筛选 | `finite + depth + visibility` 有效 mask，[train.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/train.py:127) 与 [test.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/test.py:124) | M1 已落地 |
| CAD-Mesher 时序稳定需求 | 时序模块依赖一致输入定义 | 数据集 query 输出语义对齐，[dataset.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/d4rt/data/dataset.py:364) | M1 已落地 |
| DeSiRe-GS 脚本化分离流程 | 先规范 CLI 边界，再扩展功能 | 测试入口参数守卫 `num_queries <= N`，[test_d4rt.py](/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT/scripts/test_d4rt.py:18) | M1 已落地 |
| DeGauss 分阶段工程策略 | 先链路基线、后算法增强 | M1 与 M2 分阶段拆分（M1 只做链路收敛） | M1 已执行 |

### 7.4 M1 未直接落地的借鉴项（阶段边界）
1. `CAD-Mesher` 的在线增量网格更新机制：留到 M3。  
2. `ERASOR` 的完整地面恢复/极坐标分桶细节：留到 M3+。  
3. `DeGauss`/`EmerNeRF` 的完整双分支模型训练范式：当前不替换 D4RT 主体，仅保留为思路参考。

### 7.5 独立实现证据
M1 修改集中在 D4RT 的训练/测试与数据接口收敛（`train.py`、`test.py`、`test_d4rt.py`、`dataset.py`），没有引入外部仓库代码路径或依赖。
