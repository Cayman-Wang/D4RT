# M2 审查报告（已按 M2.5 / 静态 Mesh 重排同步）

审查时间：2026-03-05  
同步更新：2026-03-11  
审查范围：`/home/grasp/Desktop/wym-project/4d-gaussgym/D4RT`

## 0. 文档定位
这份文档保留 **M2 严格审计** 的角色，但表述已经同步到当前权威口径，避免继续把它误读成“当前主阻塞仍在 M2 代码缺口”。

当前应这样理解它：
- 它负责回答：**M2 的动静点分离 + replay 契约本身是否已经达到可交付基线**。
- 它不再负责回答：**mesh 质量是否已经可以推进到高质量阶段**。
- 当前 mesh 研究节奏已改由 `research/guides/d4rt_static_mesh_research_reorder_zh.md` 管理，并固定为 `S0 / S1a / S1b / S2a / S2b`。

换句话说：
- **M2 审计通过** 只代表分离与 replay 边界已经成立；
- **不代表** authority replay 已达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”；
- **也不代表** 已经可以直接进入 `M3b Mesh Quality`。

---

## 1. 当前结论（同步后的简要版）
- 原 2026-03-05 审计中的阻塞项和高优先问题，现已完成闭环或降为非阻塞说明项。
- M2 现在应被视为：**可交付的静/动态点分离 + 实例跟踪 + replay 输出基线**。
- 当前真正的主阻塞已经转移到 **M2.5 / M3 门禁**：
  - authority repo 与 formal 证据是否一致；
  - authority replay 是否达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”；
  - `S1a` 与 `S2a` 对照后，问题到底在 replay 输入质量，还是在 mesher。
- 当前 mesh 研究固定为：
  - `S0`：smoke mesh / 链路健康检查
  - `S1a`：`replay static points -> volumetric/static mesh`
  - `S1b`：replay-based surface refinement
  - `S2a`：`raw RGBD -> dense static mesh` low-cost oracle
  - `S2b`：高成本上界 / 论文对照线

---

## 2. 原审计问题的当前状态

### 2.1 [已关闭 / 原严重] 中间区间未继承上一状态
- 原问题：`D4RT/d4rt/separation/motion_score.py` 仅输出 `UNCERTAIN`，没有实现“滞回中间区保持上一状态”，会导致时序抖动。
- 当前状态：**已关闭**。
- 当前证据：
  - `D4RT/d4rt/separation/motion_score.py:262` 已对 `middle_mask` 应用 hysteresis；
  - `D4RT/d4rt/separation/motion_score.py:264` 会读取 `prev_label`；
  - `D4RT/d4rt/separation/motion_score.py:275` 会回写稳定后的上一状态。
- 结论：M2 对“中间区滞回”这一关键能力的要求已满足。

### 2.2 [已关闭 / 原高] `scripts/test_d4rt.py` 默认参数可直接触发采样失败
- 原问题：`num_queries=2048` 与 `N=32` 一类组合会触发 `Not enough trajectory points`，导致默认入口不稳定。
- 当前状态：**已关闭**。
- 当前证据：
  - `D4RT/scripts/test_d4rt.py:18` 新增 `validate_sampling_args`；
  - `D4RT/scripts/test_d4rt.py:20` 明确要求 `num_queries <= N`；
  - `D4RT/scripts/test_d4rt.py:92` 在 datamodule 构建前执行参数校验。
- 结论：M1/M2 的常用测试入口已具备参数保护，不再是高优先阻塞。

### 2.3 [已关闭 / 原中] `run_separation_replay.py` 缺少 `--dry_run/--save_json`
- 原问题：计划中提到的 dry-run 与可选 JSON 输出开关，在早期脚本中未显式暴露。
- 当前状态：**已关闭**。
- 当前证据：
  - `D4RT/scripts/run_separation_replay.py:54` 已提供 `--dry_run`；
  - `D4RT/scripts/run_separation_replay.py:59` 已提供 `--save_json`；
  - `D4RT/scripts/run_separation_replay.py:285` 已按 `dry_run/save_json` 组合决定是否写 `summary.json`。
- 结论：replay CLI 已与计划约定对齐，可用于 formal 输出与快速校验两种模式。

### 2.4 [已收口 / 原低] `io_contract.py` 中 mesh 字段边界语义不清
- 原问题：`SeparationFrame` 在 M2 阶段即包含 `static_mesh_path/dynamic_meshes`，容易让人误读为 M2 已包含 mesh 质量验收。
- 当前状态：**已收口**。
- 当前证据：
  - `D4RT/d4rt/separation/io_contract.py:74` 已明确说明：`static_mesh_path` 与 `dynamic_meshes` 在 M2 只是可选占位；
  - 当前 `ACTIVE_PLAN` 已进一步冻结：`mesh_builder` 只消费 replay 产物 `summary.json + frames/frame_*.npz`，不把 mesh 字段当作 M2 主合同。
- 结论：该项不再构成语义歧义阻塞，但仍需在后续 mesh 文档中持续保持这一边界。

---

## 3. A-F 验收结果（按当前口径重判）
1. A. 模块与接口：通过  
   证据：`d4rt/separation/io_contract.py`、`motion_score.py`、`instance_tracker.py`、`__init__.py` 均存在；`SeparationFrame` 已覆盖 `timestamp/static_points_world/dynamic_points_world/dynamic_instance_ids/dynamic_scores/confidence/visibility` 等核心字段。

2. B. 分离逻辑有效性：通过  
   证据：质量门控、双阈值分区、hysteresis 中间区继承上一状态已落地；`motion_score.py` 中已存在 `middle_mask -> prev_label -> labels` 的闭环。

3. C. 跟踪逻辑有效性：通过  
   证据：`dbscan_cluster`、Hungarian/贪心回退、`missed_frames` 生命周期管理仍在；当前未发现需要回退的结构性问题。

4. D. 脚本可运行性与 replay 契约：通过  
   证据：`run_separation_replay.py` 已具备 `--dry_run`、`--save_json`、`summary.json + frames/frame_*.npz` 输出路径；当前 formal replay 已被纳入权威仓库 handoff 与 retrospective。

5. E. 回归风险（M1 不破坏）：通过（仍需守住生产路径）  
   证据：`scripts/test_d4rt.py` 已增加参数保护；当前口径已冻结 `PointOdysseyDataset + PointOdysseyDataModule` 为训练/测试/导出的唯一生产路径，`D4RTDataset + D4RTDataModule` 仅作 legacy/baseline 兼容。

6. F. 测试与验证：通过（但不等于高质量点云通过）  
   证据：`D4RT/d4rt/tests/test_motion_score.py`、`test_instance_tracker.py`、`test_io_contract.py` 已覆盖 M2 核心单元；同时 authority replay 证据已经在当前仓库重建。

---

## 4. Pass/Fail 总结（同步后）
**Pass（达到 M2 可交付基线）**

但这里的 `Pass` 需要严格限定含义：
- 表示 **M2 的 separation + tracking + replay contract** 已经成立；
- 不表示 replay 点云质量已经达到 mesh 质量阶段的门禁；
- 不表示当前可以跳过 `S1a / S2a` 的诊断与判责，直接投入 `S1b` 或动态 mesh 高成本路线。

因此，当前项目状态应该表述为：
- **M2 已通过**；
- **M2.5 对齐已完成**；
- **当前里程碑仍是 `M3a Mesh Smoke`，`M3b` 门禁尚未通过**。

---

## 5. 与当前静态 Mesh 研究重排的关系
这份审计报告现在只负责说明：**可以放心把 replay 输出作为后续 mesh 研究的正式输入边界**。

具体边界固定为：
- 正式生产 mesh 输入：`replay_full/summary.json + replay_full/frames/frame_*.npz`
- 不允许：mesh builder 直接消费 dataloader batch
- 不允许：把训练期 `t_cam` 语义直接泄漏到 downstream mesh 链路

在此基础上，静态 mesh 研究按以下顺序推进：
- `S0`：当前 smoke mesh，只做链路健康检查
- `S1a`：`replay static points -> volumetric/static mesh`，当前正式生产线第一优先
- `S2a`：`raw RGBD -> dense static mesh` low-cost oracle，只作诊断
- `S1b`：只有 authority replay 达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”后才进入
- `S2b`：只作高成本上界与论文对照

也就是说：
- 这份报告通过之后，允许继续做 `S0 / S1a / S2a`；
- 但**不自动授权**进入 `S1b / M3b Mesh Quality`。

---

## 6. 当前最重要的非 M2 风险
当前真正应关注的，不再是 2026-03-05 那一批 M2 代码缺口，而是下面这些问题：
- authority replay 是否已达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”；
- `S2a` 好而 `S1a` 差时，是否说明 replay 静态点密度不足或仍有动态污染；
- `S1a` 好但仍需 `S1b` 才明显改善时，是否说明主要瓶颈已转为表面拟合与补面；
- `S2a` 本身也差时，是否应该先回到 raw RGBD 回投、深度质量、位姿链路与时间窗口选择；
- 是否仍误把 `GO-Surf / BundleFusion / BundleSDF / 4DTAM / GauSTAR / dynsurf` 这类高成本外部项目拉入当前主线。

当前固定口径是：
- `Voxblox + vdbfusion` 属于 `S1a` 核心工程参考；
- `ERASOR + CAD-Mesher` 属于 `S1a` 前置净化/抗动态污染参考；
- `GO-Surf` 属于 `S2b` 离线高保真上界；
- `BundleFusion` 属于 `S2b` 经典控制组；
- `BundleSDF` 只作对象级补充参考；
- `4DTAM / GauSTAR / DynaSurfGS / dynsurf` 继续冻结在 future route（未来路线池）。

---

## 7. 已执行命令与关键结果（保留审计证据 + 当前解释）
1. `git -C D4RT status --short && git -C D4RT diff --name-only`  
   结果：最初审计时已确认 `d4rt/separation/*`、`scripts/run_separation_replay.py`、`d4rt/tests/*` 存在新增修改。

2. `conda run -n d4rt python -m compileall D4RT/d4rt/separation D4RT/scripts/run_separation_replay.py`  
   结果：编译通过。

3. `conda run -n d4rt python D4RT/scripts/run_separation_replay.py --help`  
   结果：CLI 可用；当前同步口径下应理解为 replay 合同入口已建立。

4. `conda run -n d4rt pytest D4RT/d4rt/tests -q`  
   结果：当时环境缺 pytest（后续环境问题已单独解决，不再作为 M2 逻辑阻塞）。

5. `conda run -n d4rt python -m unittest discover -s D4RT/d4rt/tests -p 'test_*.py' -v`  
   结果：6/6 通过。

6. 端到端 smoke：  
   `/home/grasp/miniconda3/envs/d4rt/bin/python D4RT/scripts/run_separation_replay.py --input_npz /tmp/m2_replay_input.npz --output_dir /tmp/m2_replay_out2 --max_frames 4 --cluster_min_samples 2 --cluster_eps 0.3`  
   结果：运行成功，`summary.json` 与 `frames/*.npz` 生成，动态/静态点数量非退化。

7. 当前权威 formal 证据（已在后续文档中冻结）：
   - `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/separation_stream.npz`
   - `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/summary.json`
   - `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames/frame_*.npz`
   - `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/mesh_smoke/mesh_summary.json`

---

## 8. 未验证项与残余风险（同步后）
- 未在真实长序列上系统量化 ID switch rate、动态召回、静态污染率。
- `scipy` 不可用时仍会回退到贪心匹配，复杂场景下跟踪精度可能下降。
- 当前 `Pass` 只证明 M2 边界成立，不证明 authority replay 已达到高质量 mesh 门禁。
- `S2a` 的实现与参数若和 `S1a` 不可比，仍可能把输入问题误判为 mesher 问题。
- 外部高成本项目即使已拉取，也不应越级进入当前主线。

---

## 9. 后续建议（按当前口径）
1. 不再把这份报告当作当前主阻塞来源；当前主阻塞应转到 authority replay 质量与 `S1a / S2a` 判责。
2. 在 `S0 / S1a / S2a` 上维持统一产物接口：
   - `points_world.ply`
   - `mesh_raw.ply`
   - `mesh_clean.ply`
   - `metrics.json`
   - `compare_notes.md`
3. 第一轮对照始终固定同一序列、同一时间窗、同一套 mesher 参数，避免错误归因。
4. 只有当 replay 达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”后，才允许把重点从输入质量问题转向 `S1b / M3b Mesh Quality`。
5. 动态 mesh 路线当前继续只保留粗粒度占位，不把 `4DTAM / GauSTAR / DynaSurfGS / dynsurf` 拉进本轮里程碑。
