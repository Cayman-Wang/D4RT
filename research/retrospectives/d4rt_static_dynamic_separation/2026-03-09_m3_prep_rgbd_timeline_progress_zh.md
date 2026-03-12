# M3 准备阶段进展：RGBD 时间轴动态查看与导出闭环

- 日期：2026-03-09
- 阶段：M3 准备阶段（可视化与数据检查能力补齐）
- 对应主计划：[research/plans/d4rt_static_dynamic_separation/master_plan_zh.md](../../plans/d4rt_static_dynamic_separation/master_plan_zh.md)

## 1. 问题
在 M2 分离回放可用后，项目仍存在一个工程缺口：  
1. 现有 `visualize_dataset_rgbd_sequence.py` 偏“序列融合静态查看”，不适合按时间轴观察稠密点云动态变化。  
2. 用户需要可交互地验证“同一场景在不同时间的稠密 RGBD 变化”，并导出逐帧结果用于离线分析。  
3. 需要一个不依赖 GUI 的导出路径，便于远程机器/批处理环境运行。

## 2. 思路
采用“保留融合基线 + 新增独立时间轴脚本”的策略：  
1. 保留 `visualize_dataset_rgbd_sequence.py` 的定位，不改原逻辑。  
2. 新增 `visualize_dataset_rgbd_timeline.py` 实现逐帧稠密 RGBD 重建与时间轴回放。  
3. 动态层采用“当前帧或最近 K 帧滑窗叠加”，静态层固定为全局累计点云，先满足可视检查，再进入后续真实分离/mesh 阶段。

## 3. 实现
1. 新增脚本 [visualize_dataset_rgbd_timeline.py](/mnt/windows_data2/wym-project/4d-gaussgym/D4RT/scripts/visualize_dataset_rgbd_timeline.py)：  
   - 读取 `rgbs/`、`depths/`、`anno.npz`，回投 world 稠密彩色点云。  
   - `dynamic_mode=frame|window`，并支持 `dynamic_window`。  
   - Matplotlib slider 回放（含 Prev/Next、键盘左右键）。  
   - `backend=none` 下可只做统计与导出。  
   - 支持 `--export_frames_dir`（逐帧 PLY）和 `--export_summary_json`（摘要）。
2. 新增测试 [test_visualize_dataset_rgbd_timeline.py](/mnt/windows_data2/wym-project/4d-gaussgym/D4RT/d4rt/tests/test_visualize_dataset_rgbd_timeline.py)：  
   - 帧选择和窗口边界行为。  
   - `dynamic_window` 超当前帧数时的截断行为。  
   - `backend=none` 导出 PLY/JSON 与空动态帧路径。
3. 文档更新：  
   - [README_zh.md](/mnt/windows_data2/wym-project/4d-gaussgym/D4RT/README_zh.md) 增加 timeline 命令入口。  
   - [rgbd_scene_guide_zh.md](/mnt/windows_data2/wym-project/4d-gaussgym/D4RT/rgbd_scene_guide_zh.md) 增加 4D 时间轴查看与无 GUI 导出章节。

## 4. 验证
1. 新增测试通过：  
   `PYTHONPATH=. conda run -n d4rt pytest d4rt/tests/test_visualize_dataset_rgbd_timeline.py -q` -> `3 passed`。  
2. 全量 D4RT 测试通过：  
   `PYTHONPATH=. conda run -n d4rt pytest d4rt/tests -q` -> `27 passed`。  
3. 真实数据 smoke 通过（`backend=none`）：  
   可正常输出选帧统计、初始动态窗口信息和 summary JSON。

## 5. 局限
1. 动态层目前是“时间窗口表达”，不是“真实静/动态分离结果”。  
2. 尚未接入 M3 的 static/dynamic mesh builder。  
3. Matplotlib 方案适合检查与演示，不是高帧率实时渲染路径。

## 6. 下一步
1. 按 M3 计划实现 `d4rt/separation/mesh_builder.py`，打通 static/dynamic mesh 最小闭环。  
2. 在 separation 链路补齐真实 RGB 贯通（stream -> replay -> visualize）。  
3. 基于非 quick 配置进行更大规模导出与回放验收（含 1536 档位）。

## 7. 借鉴关系与实现边界
### 7.1 借鉴层级与来源边界
本次属于可视化与数据导出增强，借鉴的是项目内已有时间轴交互习惯（`visualize_separation_timeline.py`），不引入外部项目代码。

### 7.2 具体借鉴了哪些工作（来源位置 + 借鉴点）
1. 项目内 `visualize_separation_timeline.py`  
借鉴点：Matplotlib slider + frame index 交互方式，保证交互习惯一致。
2. 项目内 `visualize_dataset_rgbd_sequence.py`  
借鉴点：RGBD 回投与体素下采样逻辑，保证数据语义一致。

### 7.3 与 M2/M3 主链路的边界
1. 不修改 M2 分离算法与 replay 契约。  
2. 不替代 `visualize_dataset_rgbd_sequence.py`。  
3. 不提前实现 mesh 构建与 GaussGym 动态碰撞接入。
