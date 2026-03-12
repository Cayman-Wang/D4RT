# D4RT 静/动态点云分离入门指南（原理优先，不改代码版）

> 适用人群：第一次做 4D 重建后处理、希望先把“静态点云与动态点云分开”的同学。  
> 本文只做一件事：**教你把 D4RT 推理输出分成 `static / dynamic / uncertain` 三类点云**。  
> Mesh 重建是下一步，不在本文展开。

---

## 1. 先建立全局认知：你现在处在什么阶段

完整链路是：

`D4RT训练 -> 推理得到4D点 -> 静/动态分离 -> 仅静态点建mesh`

你当前要完成的是第 3 步（静/动态分离）。

为什么必须先分离再建 mesh？
- Mesh 假设场景几何相对稳定；
- 动态物体（人、车、门、宠物）会让网格出现重影、拉丝、破洞；
- 先把“可能动的点”剔出去，静态 mesh 会更完整、更干净。

---

## 2. 必备前置知识（小白友好版）

### 2.1 `query` 是什么

D4RT 不是直接对整帧每个像素都输出 3D 点，而是对一批查询点（query）做预测。  
每个 query 由空间位置 + 时间信息组成，大致是：
- 空间：`(u, v)`
- 时间：`t_src, t_tgt, t_cam`

你可以把它理解成：
- “我在 `t_src` 看到的某个位置，想问它在 `t_tgt` 时刻、以 `t_cam` 相机参考系表示时，3D 在哪里？”

### 2.2 `t_src / t_tgt / t_cam` 分别代表什么

- `t_src`：query 来源帧（从哪一帧取局部外观）
- `t_tgt`：要预测该点几何状态的目标时间
- `t_cam`：3D 坐标最终写在哪个相机坐标系下

> 训练数据采样逻辑可参考 `D4RT/d4rt/data/dataset.py`（query 采样和时序索引生成）。

### 2.3 模型输出里和分离最相关的四个量

D4RT 模型输出包含（关键位置：`D4RT/d4rt/models/d4rt_model.py:145`）：
- `coords_3d`：预测 3D 点
- `visibility`：可见性概率
- `motion`：运动向量
- `confidence`：预测置信度

这四个就是分离的核心信号。

### 2.4 相机坐标系 vs 世界坐标系

- 相机坐标系（camera frame）：坐标随相机视角变化；
- 世界坐标系（world frame）：全局统一坐标，不随相机变化。

**静/动态分离必须尽量在世界系做**，因为“静态”本质是“在全局空间里几乎不动”。

### 2.5 为什么本文坚持三分类而不是二分类

我们输出三类：
- `static`（静态）
- `dynamic`（动态）
- `uncertain`（不确定）

原因：真实数据有噪声，硬分成静态/动态会把大量边界样本误分。把不确定样本单独放出来，后续 mesh 阶段更稳。

---

## 3. 为什么不能只看 motion（原理重点）

你会很自然地想：`motion` 小就是静态，大就是动态。  
但工程里只看 motion 往往不稳，主要有 3 个原因：

1. motion 监督通常偏弱  
   在当前训练实现里，`gt_motion` 是可选项，很多情况下并没有强监督（参考 `D4RT/d4rt/train.py:200`）。

2. motion 会混入噪声  
   深度误差、外参误差、纹理少区域都会让 motion 抖动。

3. 慢速动态物体容易漏检  
   比如缓慢移动的人，motion 可能不大，但时序几何其实并不稳定。

所以更稳的方案是**多信号联合**：
1) `visibility`（可见性可信度）  
2) `confidence`（预测可靠性）  
3) 世界系时序一致性（主判据）  
4) `motion`（辅助判据）

---

## 4. 分离算法：从 0 到 1 的完整流程

下面是推荐默认规则（可直接作为第一版基线）。

## Step A：质量预过滤（先做）

先过滤低质量点，避免噪声直接进入判定：
- `visibility_mean > 0.60`
- `confidence_mean > 0.50`

说明：
- 这里的 mean 是“同一 query 在多个时间上的平均值”。
- 如果你目前是单时刻输出，也可以先用当前帧值替代，但稳定性会差一些。

---

## Step B：统一到世界坐标系 + 计算时序一致性（主判据）

### B1. 坐标变换原则

若 `coords_3d` 在相机系（以 `t_cam` 为参考），则变到世界系：

`p_world = inv(cams_T_world[t_cam]) @ p_cam_h`

其中：
- `p_cam_h = [x, y, z, 1]^T`
- `cams_T_world[t]` 常表示 `world -> camera` 变换（按当前数据集实现注释）
- 所以要取逆矩阵得到 `camera -> world`

### B2. 轨迹离散度（静态主判据）

对同一 query 在多个目标时刻的世界点序列 `p_t`，计算离散度。  
本文默认用**中位位移**：

1. `c = median_t(p_t)`（轨迹中心）
2. `d_t = ||p_t - c||`
3. `sigma = median_t(d_t)`

`sigma` 越小，说明该点在全局空间越稳定，越像静态点。

> 你也可以用 MAD（Median Absolute Deviation）替代，逻辑一样，阈值需重调。

---

## Step C：联合判定（输出三类）

默认阈值如下：

- `static`：
  - `sigma < 0.03 m`
  - 且 `motion_median < 0.02`

- `dynamic`：
  - `sigma > 0.06 m`
  - 或 `motion_median > 0.05`

- `uncertain`：
  - 不满足上面两类的其余样本

为什么这样设：
- `sigma` 是主判据（几何时序一致性最可靠）；
- `motion` 只做“补充裁决”；
- 中间区域给 `uncertain`，避免误伤。

---

## Step D：导出结果

至少导出 3 份点云：
- `static_points.ply`
- `dynamic_points.ply`
- `uncertain_points.ply`

再导出 1 份统计报告（JSON/TXT 均可）：
- 三类点数量与占比
- `sigma` 分布（均值/中位数/分位数）
- `motion` 分布

---

## 5. 参数不是拍脑袋：每个阈值的物理意义

## 5.1 `visibility` 阈值

意义：点是否真的被有效观测到。  
过低会引入遮挡点和错误投影点。

## 5.2 `confidence` 阈值

意义：模型对自身预测的信心。  
过低会把“模型瞎猜”的点混入静态集合。

## 5.3 `sigma` 阈值（最关键）

意义：该点在世界系下的时序稳定性。  
是静/动态最直接的几何证据。

## 5.4 `motion` 阈值

意义：运动幅度提示。  
用于辅助，不建议单独决定静/动态。

### 误差来源（你看到异常结果时，优先想到这些）
- 深度噪声（尤其远距离/反光区域）
- 外参抖动（相机位姿不稳）
- 低纹理区（匹配不稳定）
- 单位混淆（m 和 mm）

### 固定调参顺序（推荐）
1. 先调 `visibility/confidence`（先保质量）
2. 再调 `sigma`（主判据）
3. 最后微调 `motion`

### 网格搜索建议
- `sigma`: `0.02 / 0.03 / 0.05`
- `motion`: `0.015 / 0.02 / 0.03`
- `visibility`: `0.5 / 0.6 / 0.7`
- `confidence`: `0.4 / 0.5 / 0.6`

---

## 6. 完整执行流程（不改代码版，任务清单）

这一节按“输入 -> 处理 -> 输出 -> 通过标准 -> 做错现象”写。

## 任务 1：选可用 checkpoint

输入：训练日志目录（如 `lightning_logs_*`）  
处理：选 `train/loss` 长期有限值、已成功保存的 ckpt  
输出：`*.ckpt` 路径  
通过标准：ckpt 文件存在且大小正常  
做错现象：后续推理直接失败或输出全噪声

---

## 任务 2：准备推理输出（已有导出就直接用）

输入：模型推理导出的 `npz`/`npy` 结果  
处理：检查是否包含最少字段（见第 10 节 Data Contract）  
输出：可用于分离的中间数据  
通过标准：字段齐全、shape 可解释  
做错现象：无法形成轨迹，无法算 `sigma`

> 关键提醒：要做“时序一致性”，你需要同一 query 在多个时间上的结果。  
> 如果当前导出是每次随机 query 且无法对应同一 query id，就先不能直接做轨迹判定。

---

## 任务 3：坐标统一（camera -> world）

输入：`coords_3d`, `t_cam`, `cams_T_world`  
处理：按变换公式逐点转换到 world  
输出：`coords_3d_world`  
通过标准：尺度（米）合理、点云方向正确  
做错现象：静态点像在“抖动”或整体翻转/漂移

---

## 任务 4：计算四类信号

输入：`visibility`, `confidence`, `motion`, `coords_3d_world`  
处理：
- 计算 `visibility_mean`
- 计算 `confidence_mean`
- 计算 `motion_median`
- 计算 `sigma`

输出：每个 query 的 4 个统计量  
通过标准：无 NaN/Inf，数值范围合理  
做错现象：阈值怎么调都不稳定

---

## 任务 5：按规则打标签

输入：4 个统计量 + 默认阈值  
处理：按 Step C 打 `static/dynamic/uncertain` 标签  
输出：三类点索引  
通过标准：三类都非空（至少 static 和 uncertain 通常非空）  
做错现象：`static` 或 `dynamic` 接近 0%/100%

---

## 任务 6：导出点云 + 报告

输入：标签 + 点坐标  
处理：分别写出三类点云并保存统计报告  
输出：`static_points.ply` 等文件 + 报告  
通过标准：可视化可打开，统计可读  
做错现象：输出文件为空或点数量不一致

---

## 7. 最低验收 + 进阶验收

## 7.1 最低验收（必须过）
- 能导出三类点云；
- `static` 占比不是 0% 也不是 100%；
- 主要动态主体（人/车）大部分落在 `dynamic`。

## 7.2 进阶验收（建议过）
- 把 `static_points` 用于下一步 mesh 重建时，连通性明显优于“未分离原始点云”；
- 调阈值后，结果变化方向可解释（阈值更严格 -> static 变少，dynamic/uncertain 变多）。

---

## 8. 常见问题排查（症状 -> 原因 -> 修复）

### 问题 1：`dynamic` 几乎为 0
- 常见原因：阈值太松，或 `motion` 尺度和单位不一致
- 修复方向：
  - 提高 `sigma` 动态判据敏感度（降低 dynamic 触发阈值）
  - 检查单位是否统一为米（m）

### 问题 2：`static` 几乎为 0
- 常见原因：阈值太紧，或坐标变换方向写反
- 修复方向：
  - 放宽 `sigma < 0.03m` 到 `0.05m` 试验
  - 核对是否使用了 `inv(cams_T_world)`

### 问题 3：分离结果“闪烁”或帧间跳变明显
- 常见原因：时间索引错位、轨迹未对齐、外参噪声大
- 修复方向：
  - 先验证同一 query 是否在多帧可追踪
  - 检查 `t_cam/t_tgt` 对齐关系

### 问题 4：输出尺度非常离谱
- 常见原因：米/毫米混用，或深度缩放处理错误
- 修复方向：
  - 统一到米（m）
  - 在可视化里量测已知物体尺寸做 sanity check

### 问题 5：训练日志里见到少量 NaN（如 val）是否不能做分离
- 结论：不一定。  
- 影响边界：
  - 少量验证 NaN（例如空 mask 边界 batch）不等于整体不可用；
  - 但在分离前必须做“导出结果有限值检查”。

建议在分离前做一次数据健康检查：
- `coords_3d / visibility / confidence / motion` 必须全部为有限值（非 NaN/Inf）；
- 非有限值样本先剔除或标记为 `uncertain`。

---

## 9. 快速复习图（文本版）

```text
[训练得到可用ckpt]
      |
      v
[推理导出4类信号: coords_3d/vis/conf/motion]
      |
      v
[坐标统一到world]
      |
      v
[质量预过滤(vis/conf)]
      |
      v
[计算sigma + motion_median]
      |
      v
[三分类: static / dynamic / uncertain]
      |
      v
[导出三类点云 + 统计报告]
      |
      v
[下一步: 仅用static建mesh]
```

---

## 10. 导出数据约定（Data Contract，建议固定）

为避免后续脚本口径不一致，推荐固定最小字段如下。

每个样本至少包含：
- `coords_3d`：`float32`，形状建议 `[T, Q, 3]` 或 `[Q, 3]`
- `visibility`：`float32`，形状 `[T, Q]` 或 `[Q]`，范围 `[0, 1]`
- `confidence`：`float32`，形状 `[T, Q]` 或 `[Q]`，范围 `[0, 1]`
- `motion`：`float32`，形状 `[T, Q, 3]` 或 `[Q, 3]`
- `t_src`：`int64`
- `t_tgt`：`int64`
- `t_cam`：`int64`
- `cams_T_world`：`float32`，形状 `[T, 4, 4]` 或 `[4, 4]`

硬性约定：
- 长度单位统一为米（m）
- 矩阵语义写入元数据（是否 `world->camera`）
- 若是时序分离，必须可定位“同一 query 在多时刻”的对应关系

---

## 11. 术语与公式附录

## 11.1 术语对照

- Query：查询点  
- Visibility：可见性  
- Confidence：置信度  
- Motion：运动向量  
- Trajectory consistency：轨迹时序一致性  
- Camera frame：相机坐标系  
- World frame：世界坐标系

## 11.2 最小公式集合

1) 坐标变换（相机到世界）
- `p_world = inv(T_cw) * p_cam_h`

2) 轨迹中心
- `c = median_t(p_t)`

3) 中位位移离散度
- `sigma = median_t(||p_t - c||)`

4) 三分类判定（默认）
- `static`: `sigma < 0.03m` and `motion_median < 0.02`
- `dynamic`: `sigma > 0.06m` or `motion_median > 0.05`
- `uncertain`: else

---

## 12. 一句话落地建议

第一次做时，不要追求一步到位：  
先用默认阈值跑通三分类并可视化，再按“先 vis/conf、再 sigma、后 motion”的顺序微调。  
只要你能稳定产出 `static_points.ply` 且动态主体主要落在 `dynamic`，这一步就算成功。
