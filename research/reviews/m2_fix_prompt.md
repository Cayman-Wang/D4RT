# 给其他 AI 的修复提示词（归档版：M2 已收口）

这份提示词保留为历史归档，不再作为当前主线执行提示。

原因：`research/reviews/m2_audit_report.md` 已同步为“`M2` 已通过、当前阻塞转到 `M2.5 / M3` 门禁”的口径，因此下面这组“修复 M2 阻塞”的任务不再是当前优先工作。

如果需要把任务交给其他 AI，当前更适合的目标应是：
- 固定 `S0 / S1a / S1b / S2a / S2b` 的研究分层
- 在同一序列、同一时间窗、同一套 mesher 参数下对比 `S1a` 与 `S2a`
- 用 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”作为进入 `S1b / M3b Mesh Quality` 的门禁
- 不把 `GO-Surf / BundleFusion / BundleSDF / 4DTAM / GauSTAR / DynaSurfGS / dynsurf` 越级拉入当前主线

下面内容仅用于回看 2026-03-05 那一轮 `M2` 修复目标。

## 仓库
`/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT`

## 修复目标（必须全部完成）
1. 修复 `motion_score` 的滞回逻辑：
   - 当 `S >= dynamic_threshold` -> 动态
   - 当 `S <= static_threshold` -> 静态
   - 当 `static_threshold < S < dynamic_threshold` -> 保持上一状态（基于 `point_id`）
   - 若无上一状态，才置 `UNCERTAIN`

2. 修复 `scripts/test_d4rt.py` 参数可运行性风险：
   - 启动时校验 `num_queries <= N`
   - 若不满足，抛出清晰错误信息（不要等深层报错）
   - 可选：把默认值改为可运行组合

3. 对齐 M2 计划中的 replay CLI：
   - 在 `scripts/run_separation_replay.py` 增加 `--dry_run`
   - 增加 `--save_json`（默认可开或可选）
   - `--dry_run` 下只做输入校验/时序检查/统计，不落盘帧数据

4. 新增测试：
   - `test_motion_score_hysteresis_stateful.py`（必须）
   - `test_test_d4rt_arg_guard.py`（必须）
   - `test_replay_cli_dry_mode.py`（必须）

## 验收标准（你必须自测并输出结果）
1. 语法检查通过：
   - `conda run -n d4rt python -m compileall d4rt/separation scripts/run_separation_replay.py scripts/test_d4rt.py`

2. 测试通过：
   - `conda run -n d4rt pytest d4rt/tests -q`
   - 若 pytest 不可用，回退：`conda run -n d4rt python -m unittest discover -s d4rt/tests -p 'test_*.py' -v`

3. CLI 检查：
   - `conda run -n d4rt python scripts/run_separation_replay.py --help`
   - `conda run -n d4rt python scripts/run_separation_replay.py --dry_run ...`（给出一条可复现命令）

4. 非退化输出：
   - 给出一次 smoke 运行结果，证明动态点比例不是全 0/全 1

## 修改边界
- 只改 M2 相关文件，不要改 gauss_gym、M3/M4/M5。
- 不要引入重型新依赖。
- 保持 M1 的 train/test GT 语义和 loss 入参兼容。

## 输出格式（必须）
1. 先列出 `Changed Files`
2. 再列出 `Why`（每个文件一句）
3. 再列出 `Validation Commands + Results`
4. 最后给出 `Residual Risks`

如果你发现某项无法完成，必须说明阻塞原因并提供可行替代。