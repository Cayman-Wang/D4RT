# 静态 Mesh / 动静点云分离参考项目对照表（双基线 + 分层固定版）

## 1. 目的
- 固定当前长期使用的参考项目对照表，避免每次回顾方案或写论文时重新翻仓库。
- 把参考工作明确映射到固定分层：`S0 / S1a / S1b / S2a / S2b`。
- 让“项目分工、优先级、论文定位、落地路径”在一个文档里一次说清楚。

关联文件：
- 研究重排：`research/guides/d4rt_static_mesh_research_reorder_zh.md`
- 拉取脚本：`research/tools/pull_reference_projects.sh`
- 机器索引：`research/guides/reference_projects_mesh_static_dynamic_index.csv`

---

## 2. 当前固定分层

### 2.1 `S0`：smoke mesh
- 作用：只做链路健康检查
- 输入：`replay_full/summary.json + replay_full/frames/frame_*.npz`
- 当前地位：保留，不追质量

### 2.2 `S1a`：replay 基础建面
- 作用：`replay static points -> volumetric/static mesh`
- 重点：稳健融合、抗动态污染、基础 mesh
- 当前地位：正式生产线第一优先

### 2.3 `S1b`：replay 表面精修
- 作用：在 `S1a` 基础上做补面、修面、表面细化
- 当前地位：门禁后再进入，不抢当前主线

### 2.4 `S2a`：raw RGBD low-cost oracle
- 作用：最便宜的诊断对照线
- 重点：用同一套本地 mesher 判断输入问题还是 mesher 问题
- 当前地位：可提前做，但不进入生产链路

### 2.5 `S2b`：高成本上界线
- 作用：论文相关工作、几何上界和长期路线判断
- 当前地位：只做上界，不进当前里程碑

---

## 3. `S1a` 核心参考：基础融合与抗动态污染

| 项目 | 来源 | 作用定位 | 该借什么 | 当前优先级 |
| --- | --- | --- | --- | --- |
| `估计mesh/vdbfusion` | 本地 | `S1a` 核心 | 静态点到 TSDF 融合、增量更新、mesh 导出 | `P0` |
| `Voxblox` | 外部 | `S1a` 核心 | 成熟的 TSDF / ESDF 抽象、体素融合、mesh 抽取接口 | `P0` |
| `静动态点云分离/ERASOR` | 本地 | `S1a` 前置净化 | 先剔动态再建图的抗污染思想 | `P0` |
| `估计mesh/CAD-Mesher` | 本地 | `S1a` 边界设计 | 动态环境下分离和建面的工程接口 | `P1` |
| `估计mesh/4dNDF` | 本地 | `S1a` 失败后储备 | 时序一致性与隐式几何思路 | `P2` |

结论：
- `vdbfusion + Voxblox` 是当前最值得直接借工程结构的组合。
- `ERASOR + CAD-Mesher` 负责告诉你“怎样先把静态保真做好，再谈 meshing”。
- `4dNDF` 不再作为第一波主参考，而是后备路线。

---

## 4. `S1b` 条件触发参考：表面拟合与补面

| 项目 | 来源 | 作用定位 | 该借什么 | 当前优先级 |
| --- | --- | --- | --- | --- |
| `估计mesh/points2surf` | 本地 | `S1b` 核心 | 点云到隐式表面的离线补面思路 | `P2` |
| `估计mesh/NeuralPull` | 本地 | `S1b` 核心 | 从点云学习 SDF / 拉回表面的拟合思路 | `P2` |
| `估计mesh/point2mesh` | 本地 | `S1b` 核心 | 点云到 mesh 的优化式精修思路 | `P2` |
| `PPSurf` | 外部 | `S1b` 补充 | 高质量点云到表面的外部参考 | `P1` |

门禁：
- authority replay 达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”
- `S1a` 已经证明输入不是主要瓶颈

---

## 5. `S2a` 当前可落地对照线：raw RGBD low-cost oracle

| 项目 / 资产 | 来源 | 作用定位 | 该借什么 | 当前优先级 |
| --- | --- | --- | --- | --- |
| `D4RT/rgbd_scene_guide_zh.md` | 本地 | `S2a` 核心 | 最低成本 raw RGBD 回投与时序查看 | `P0` |
| `估计mesh/staticfusion` | 本地 | `S2a` 核心 | 动态场景中的静态背景融合思路 | `P0` |
| `估计mesh/co-fusion` | 本地 | `S2a` 扩展 | object-level fusion | `P1` |
| `估计mesh/maskfusion` | 本地 | `S2a` 扩展 | 语义 / 实例辅助动态重建 | `P1` |
| `估计mesh/mid-fusion` | 本地 | `S2a` 扩展 | 对象级 mapping 与重识别 | `P1` |
| `估计mesh/emfusion` | 本地 | `S2a` 扩展 | 动态对象跟踪与概率式融合 | `P1` |
| `估计mesh/panoptic_mapping` | 本地 | `S2a` 扩展 | panoptic scene understanding + mapping | `P1` |

结论：
- `S2a` 最重要的不是“做出最强效果”，而是快速给出判责证据。
- 第一轮建议只用现有 raw RGBD 回投脚本和同一套本地 mesher，不要一开始就跳进外部大系统复现。

---

## 6. `S2b` 高成本上界线：论文对照与长期参考

| 项目 | 来源 | 作用定位 | 该借什么 | 当前优先级 |
| --- | --- | --- | --- | --- |
| `GO-Surf` | 外部 | `S2b` 上界 | 离线高保真 RGBD 表面重建上界 | `P1` |
| `BundleFusion` | 外部 | `S2b` 控制组 | 经典控制组、dense RGBD 系统组织与论文基线 | `P1` |
| `BundleSDF` | 外部 | 对象级补充 | RGBD 对象跟踪与重建结合方式 | `P2` |

结论：
- `GO-Surf` 不进入当前在线或生产主线。
- `BundleFusion` 更适合“先读、先对照、再决定是否复现”。
- `BundleSDF` 不再与静态场景 dense mesh 主对照线并列。

---

## 7. 未来路线池：暂不进入当前里程碑

| 项目 | 来源 | 当前定位 | 论文写法定位 | 当前优先级 |
| --- | --- | --- | --- | --- |
| `4DTAM` | 外部 | future route（未来路线池） | 4D tracking and mapping | `P3` |
| `GauSTAR` | 外部 | future route（未来路线池） | Gaussian surface tracking / reconstruction | `P3` |
| `DynaSurfGS` | 外部 | future route（未来路线池） | 动态 Gaussian surface | `P3` |
| `dynsurf` | 外部 | future route（未来路线池） | 动态表面重建 | `P3` |
| `静动态点云分离/DeGauss` | 本地 | future route（未来路线池） | Gaussian 动静分解 | `P3` |
| `静动态点云分离/DeSiRe-GS` | 本地 | future route（未来路线池） | 动态 Gaussian 重建 | `P3` |
| `估计mesh/D-NeRF` | 本地 | future route（未来路线池） | 动态 NeRF 经典工作 | `P3` |
| `估计mesh/DG-Mesh` | 本地 | future route（未来路线池） | mesh-gaussian 混合表达 | `P3` |
| `估计mesh/MaGS` | 本地 | future route（未来路线池） | 动态对象重建与模拟 | `P3` |
| `静动态点云分离/d2nerf` | 本地 | future route（未来路线池） | 单目动静解耦 | `P3` |
| `静动态点云分离/EmerNeRF` | 本地 | future route（未来路线池） | 复杂动态时空表示 | `P3` |

---

## 8. 固定判责逻辑
- `S2a` 好、`S1a` 差：先查 replay 点云密度、静动态污染、分离保真。
- `S1a` 好、`S1b` 才明显改善：说明主要瓶颈是表面拟合与补面。
- `S1a` 和 `S2a` 都差：优先检查数据覆盖、深度质量、位姿链路，或 `D4RT/pointcloud_result_guide_zh.md` 的 A/B 类问题。
- `S2a` 差、`S2b` 好：问题主要在本地 RGBD 建面流程，不在数据本体。
- 只有 replay 达到“情况 C”后，才把重点从输入质量转移到 `S1b` 或 `M3b Mesh Quality`。

---

## 9. 固定输出接口
所有研究对照都统一导出：
- `points_world.ply`
- `mesh_raw.ply`
- `mesh_clean.ply`
- `metrics.json`
- `compare_notes.md`

这组文件名必须在 `S0 / S1a / S1b / S2a / S2b` 保持一致，便于后续论文与复盘检索。

---

## 10. 固定阅读 / 复现顺序
### 第一波：直接服务当前主线
1. `估计mesh/vdbfusion`
2. `Voxblox`
3. `静动态点云分离/ERASOR`
4. `D4RT/rgbd_scene_guide_zh.md`
5. `估计mesh/staticfusion`

### 第二波：条件触发 / 上界对照
1. `GO-Surf`
2. `BundleFusion`
3. `估计mesh/CAD-Mesher`
4. `PPSurf`

### 第三波：补面或后备路线
1. `估计mesh/points2surf`
2. `估计mesh/NeuralPull`
3. `估计mesh/point2mesh`
4. `估计mesh/4dNDF`
5. `BundleSDF`

### 第四波：未来路线池
- `4DTAM / GauSTAR / DynaSurfGS / dynsurf / DeGauss / DG-Mesh / MaGS` 等
