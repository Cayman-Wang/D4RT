# D4RT 静态优先 Mesh 研究重排指南（并入新拉取项目后的固定版）

## 1. 目标
这份指南用于固定当前阶段的静态 mesh 研究节奏，避免以下三种混淆：
- 把 `M3a Mesh Smoke` 当成高质量 mesh 路线
- 把 `replay static points -> static mesh` 和 `raw RGBD -> dense static mesh` 混成同一条链路
- 在 authority replay 还没达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”前，就过早投入高成本 mesher 或动态表面路线

当前阶段的核心原则不变：
- `M3a Mesh Smoke` 只作为链路健康检查
- 正式生产线仍是 `replay static points -> static mesh`
- `raw RGBD -> dense static mesh` 只作为诊断和上界对照，不替代生产契约

---

## 2. 固定分层：`S0 / S1a / S1b / S2a / S2b`

### 2.1 `S0`：当前 smoke mesh
用途：
- 只验证 replay 到 mesh 的链路可运行
- 不承担质量 KPI
- 继续服务当前 `M3a Mesh Smoke`

输入：
- `replay_full/summary.json + replay_full/frames/frame_*.npz`

验收：
- 文件能生成
- mesh 非空
- 基本可读

### 2.2 `S1a`：`replay static points -> volumetric/static mesh`
用途：
- 这是当前静态 mesh 生产线的第一优先层
- 目标是把 replay 的 `static_points_world` 稳健融合成基础静态 mesh
- 先解决输入点云密度、动态污染、融合稳定性，再谈高质量补面

固定参考：
- `估计mesh/vdbfusion`
- `/home/grasp/reference-projects/mesh_static_dynamic/voxblox`
- `静动态点云分离/ERASOR`
- `估计mesh/CAD-Mesher`

当前定位：
- `vdbfusion + Voxblox` 是核心工程参考
- `ERASOR + CAD-Mesher` 负责“先净化输入，再建面”的边界设计参考

### 2.3 `S1b`：`replay static points -> surface refinement`
用途：
- 只在 `S1a` 已经证明输入足够好后再做
- 目标是补孔、修面、提升表面连续性和细节
- 不允许在 replay 明显稀疏或污染明显时直接跳到这一步

固定参考：
- `估计mesh/points2surf`
- `估计mesh/NeuralPull`
- `估计mesh/point2mesh`
- `/home/grasp/reference-projects/mesh_static_dynamic/ppsurf`

门禁：
- authority replay 至少达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”
- `S1a` 已经能生成结构连续、非退化的静态基础 mesh

### 2.4 `S2a`：`raw RGBD -> dense static mesh` low-cost oracle
用途：
- 用当前工作区已有工具做低成本诊断线
- 判断 raw RGBD 直接回投后，静态 mesh 的本地上限大概在哪
- 与 `S1a` 使用同一套 mesher 参数，避免把输入问题误判成 mesher 问题

固定参考：
- `D4RT/rgbd_scene_guide_zh.md`
- `D4RT/scripts/visualize_dataset_rgbd_sequence.py`
- `估计mesh/staticfusion`
- `估计mesh/co-fusion`
- `估计mesh/maskfusion`
- `估计mesh/mid-fusion`
- `估计mesh/emfusion`
- `估计mesh/panoptic_mapping`

当前定位：
- `S2a` 不是生产链路
- `S2a` 也不是外部大系统复现任务
- `S2a` 是最低成本、最快出结论的判责工具

### 2.5 `S2b`：高成本上界线
用途：
- 只用于论文 related work、几何上界估计和长期路线判断
- 不进入当前主线里程碑，也不直接驱动生产 mesher 调参

固定参考：
- `/home/grasp/reference-projects/mesh_static_dynamic/go-surf`
- `/home/grasp/reference-projects/mesh_static_dynamic/BundleFusion`

当前定位：
- `GO-Surf`：离线高保真 RGBD 表面重建上界
- `BundleFusion`：经典控制组 / 论文基线
- `BundleSDF`：对象级 RGBD 重建补充，不与静态场景 dense mesh 主基线并列

---

## 3. 项目分工重排

### 3.1 升级为当前核心参考
- `vdbfusion + Voxblox`：升级为 `S1a` 第一优先
- `ERASOR + CAD-Mesher`：升级为 `S1` 前置净化 / 抗动态污染参考
- `staticfusion + D4RT raw RGBD 导出脚本`：升级为 `S2a` 第一优先

### 3.2 降级为条件触发参考
- `points2surf / NeuralPull / point2mesh / PPSurf`：只在 `S1a` 已经成立后进入 `S1b`
- `4dNDF`：从主参考降为 `S1` 失败后再看的时序隐式几何储备
- `BundleSDF`：从控制线主参考降为对象级补充参考

### 3.3 保持冻结在未来路线池
- `4DTAM`
- `GauSTAR`
- `DynaSurfGS`
- `dynsurf`
- `DeGauss`
- `DeSiRe-GS`
- `DG-Mesh`
- `MaGS`
- `D-NeRF`
- `d2nerf`
- `EmerNeRF`

这些项目当前只服务两件事：
- 论文 related work
- 后续动态表面 / Gaussian / 4D 表达路线储备

---

## 4. 固定判责逻辑
- `S2a` 好、`S1a` 差：问题在 replay 点云密度、静动态污染或分离保真，不在 mesher。
- `S1a` 好、`S1b` 才明显改善：主要瓶颈在表面拟合与补面，不在 replay 本身。
- `S1a` 和 `S2a` 都差：优先回到数据覆盖、深度质量、位姿链路，或 `D4RT/pointcloud_result_guide_zh.md:229` 的 A/B 类问题。
- `S2a` 已经差，但 `S2b` 明显更好：问题主要在本地 RGBD 建面流程，而不是数据本体。
- 只有当 replay 达到“情况 C”后，才允许把重点从输入质量转移到 `S1b` 或 `M3b Mesh Quality`。

---

## 5. 固定统一产物接口
所有研究对照都统一导出以下结果，便于论文复盘和横向比较：
- `points_world.ply`
- `mesh_raw.ply`
- `mesh_clean.ply`
- `metrics.json`
- `compare_notes.md`

建议目录模板：

```text
outputs/static_mesh_research/<sequence>/<stage>/
  points_world.ply
  mesh_raw.ply
  mesh_clean.ply
  metrics.json
  compare_notes.md
```

字段约定：
- `metrics.json` 至少包含：点数、面数、体素大小、时间窗口、输入来源、主要 mesher 参数
- `compare_notes.md` 至少记录：观察到的大结构连续性、孔洞数量、重影/拖尾、局部表面平滑度

---

## 6. 固定执行顺序
### 第一轮：只跑三份结果
1. `S0`：当前 smoke mesh
2. `S1a`：replay static mesh 基础融合版
3. `S2a`：raw RGBD static mesh low-cost oracle

要求：
- 固定同一 PointOdyssey 序列
- 固定同一时间窗口
- 第一轮必须使用同一套 mesher 参数

### 第二轮：只在第一轮结论明确后再扩展
- 若 `S2a` 明显优于 `S1a`：先回到 replay 质量和动静分离质量
- 若 `S1a` 已经接近 `S2a`：再考虑 `S1b`
- 若想做论文上界比较：再少量启用 `S2b`

### 第三轮：仍不进入动态 mesh 质量线
动态分支本阶段只检查：
- 实例 ID 是否连续
- mesh 是否非空
- 不设质量 KPI

---

## 7. 固定优先级
- `P0`：`vdbfusion`、`Voxblox`、`ERASOR`、`staticfusion`、现有 raw RGBD 可视化 / 导出脚本
- `P1`：`GO-Surf`、`BundleFusion`、`CAD-Mesher`、`PPSurf`
- `P2`：`4dNDF`、`BundleSDF`、`points2surf`、`NeuralPull`、`point2mesh`
- `P3`：`4DTAM`、`GauSTAR`、`DynaSurfGS`、`dynsurf`、`DeGauss`、`DG-Mesh`、`MaGS` 等未来路线

---

## 8. 与当前里程碑的关系
这份重排**不改变**当前里程碑顺序：
- 仍保留 `M3a Mesh Smoke`
- 仍以 authority replay 是否达到“情况 C”决定是否进入 `M3b`
- 新增的只是研究组织方式与判责方式，不是训练/推理 API 变更

因此，当前最合理的动作是：
- 一边继续提高 authority replay 的正式输出质量
- 一边用 `S2a` 对照 `S1a`，尽早判断问题更可能出在输入还是 mesher
