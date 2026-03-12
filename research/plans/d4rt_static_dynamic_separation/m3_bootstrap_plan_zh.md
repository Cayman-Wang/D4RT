# `S0 / M3a` 启动子计划（Mesh Smoke 最小闭环）

## 前置条件
- `M2.5` 已完成：权威仓库、formal replay 证据、dataset/query/坐标契约已经冻结。
- 当前 authority replay 即使还未达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”，也允许保留 `S0 / M3a` 作为链路健康检查。
- 若 replay 仍被 guide 判为“情况 A/B”，本子计划仍可执行，但结果只用于 smoke 验证，不得外推为高质量 mesh 结论。

## 目标
- 在 replay 产物基础上，打通 static mesh 与 dynamic mesh 的最小可运行链路。
- 本阶段只验证“能生成、可读、实例连续更新”，不追求高质量 mesh。
- 本子计划只覆盖 `S0 / M3a`，不覆盖 `S1a / S1b / S2a / S2b` 的研究比较阶段。

## 输入
- `scripts/run_separation_replay.py` 输出的 `summary.json` 与 `frames/frame_*.npz`。
- `SeparationFrame` 中静态点、动态点、实例 ID。
- 输入语义固定为 world-frame replay 产物；不直接消费 dataloader batch。

## 交付物
- `d4rt/separation/mesh_builder.py` 最小实现：
  - static 分支：跨帧融合并周期性导出 mesh。
  - dynamic 分支：按实例滑窗更新并导出 mesh。
- 一个可执行 smoke 命令，验证 mesh 文件可稳定生成。
- 研究比较接口保持兼容，后续 `S1a / S2a` 可复用同名输出文件：
  - `points_world.ply`
  - `mesh_raw.ply`
  - `mesh_clean.ply`
  - `metrics.json`
  - `compare_notes.md`

## 非目标
- 本阶段不解决 mesh 质量、表面连续性、动态外形精修。
- 本阶段不进入 `S1b`、`S2b` 或动态表面高成本路线。
- 若 formal replay 仍被 guide 判为“情况 A/B”，应优先返回训练或导出规格提升，不在本阶段调 mesh 质量。

## 当前实现口径
- 默认 `include_untracked_dynamic=True`，允许把 replay 中 `instance_id=-1` 的动态点作为 smoke 阶段的 fallback 实例流，先验证 mesh 链路不空跑。
- 这只服务于 `S0 / M3a` 的最小闭环；进入 `S1b / M3b` 前仍需要更稳定的 tracker 输出与更高密度输入。
- `S2a` 可以并行作为 low-cost oracle 运行，但它不属于本子计划的交付范围。

## 验收
- static 与 dynamic 两类 mesh 均非空且可读。
- 动态实例至少可连续两帧保持同一 instance id 的 mesh 更新。
- 不引入对现有 M2 CLI 的破坏性改动。
- 不改变 `PointOdysseyDataset` 训练期 `t_cam` 语义，也不绕过 replay 契约直接耦合训练 batch。
- 即使 authority replay 仍为“情况 A/B”，也必须把结论表述为 smoke 成功，而不是 mesh 质量成功。

## 当前默认参数（可调整）
- static 更新频率：0.5Hz
- dynamic 更新频率：2Hz
- dynamic 滑窗：8 帧
