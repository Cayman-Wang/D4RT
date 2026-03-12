# D4RT + GaussGym：静动态点云分离与静态 Mesh 重建实施指南（最小改动版）

## 1. 目标
在现有代码基础上，按最小风险路径完成以下闭环：
1. 用 D4RT 稳定训练并可推理导出时序 3D 结果
2. 在世界坐标系完成静/动态点云分离
3. 使用 GaussGym 的现有网格处理能力将静态点云重建为 mesh

---

## 2. 为什么采用这条路径
当前仓库状态下，建议先走“可导出 + 可分离 + 可建 mesh”而非先修完整 test 主链：
- D4RT 训练链路可作为稳定入口
- 模型已输出 `coords_3d`、`visibility`、`confidence`
- GaussGym 已有 Open3D/mesh 清理与补洞能力可复用

这意味着：先产出工程可用结果，再回补统一评估链路。

---

## 3. 环境与数据准备

### 3.1 D4RT 环境
```bash
cd D4RT
pip install -r requirements.txt
```

### 3.2 数据目录
保证 PointOdyssey 满足读取结构：
```text
<dataset_location>/<split>/<seq>/
  rgbs/
  depths/
  normals/
  info.npz
  anno.npz
```

---

## 4. 第一阶段：先把 D4RT 跑稳

目标：先确认训练/验证稳定，不追论文最优指标。

建议起步命令（单卡 24GB）：
```bash
cd D4RT
python scripts/train_d4rt.py \
  --dataset_location /path/to/PointOdyssey \
  --train_dset train \
  --val_dset val \
  --use_val \
  --S 8 \
  --N 32 \
  --num_queries 1024 \
  --img_size 256 \
  --batch_size 1 \
  --num_workers 4 \
  --encoder_embed_dim 768 \
  --encoder_depth 12 \
  --encoder_num_heads 12 \
  --decoder_dim 512 \
  --decoder_num_layers 6 \
  --devices 1 \
  --accelerator gpu \
  --precision bf16-mixed \
  --log_dir lightning_logs_repro
```

执行建议：
1. 先跑 500~2000 step，确认不炸 loss、无 NaN、显存稳定
2. 再跑 1~2 万 step 看趋势
3. 这一步只做稳定性验收，不急于调大模型

---

## 5. 第二阶段：新增“导出专用推理入口”

不要依赖现有 test 脚本作为唯一导出路径，直接新增一个导出脚本（例如 `D4RT/scripts/export_d4rt_predictions.py`）。

### 5.1 每个样本导出字段
至少导出：
- `coords_3d`
- `visibility`
- `confidence`
- `t_src`, `t_tgt`, `t_cam`
- `cams_T_world`
- 样本标识（`seq_name`、`sample_idx`）

### 5.2 推荐落盘组织
```text
exports/<run_name>/<split>/<seq_name>/
  sample_000000.npz
  sample_000001.npz
  ...
```

每个 `.npz` 中保存上述字段，后处理只读导出结果，不再依赖训练进程内状态。

---

## 6. 第三阶段：统一坐标系到世界系

在做静动态分离前，统一到世界坐标系。

建议流程：
1. 先明确 `coords_3d` 当前所在坐标系（通常为 `t_cam` 相机系）
2. 使用对应帧的外参将点变到世界系
3. 对转换结果做抽样可视化，检查尺度与方向是否合理

实现中务必固定同一套约定并写进代码注释，避免“相机到世界 / 世界到相机”语义混淆。

---

## 7. 第四阶段：静/动态点云分离

### 7.1 预过滤
先做置信过滤，建议起始阈值：
- `visibility > 0.5`
- `confidence > 0.5`

### 7.2 轨迹离散度判静态
对每个 query 在世界系轨迹计算时序离散度：
- 可选指标：中位位移、MAD
- 起始阈值：`0.02m`
- 网格搜索：`0.01 / 0.02 / 0.05`

### 7.3 导出结果
- `static_points.ply`
- `dynamic_points.ply`
- 统计文件（静/动态占比、离散度分布）

---

## 8. 第五阶段：在 GaussGym 侧做静态 Mesh 重建

建议将 mesh 重建放在 GaussGym 侧，复用已有工具链。

### 8.1 基础流程
1. 体素降采样（`0.01~0.02m`）
2. 法向估计
3. Ball Pivoting（半径 `[0.5v, v, 2v]`，`v` 为体素大小）
4. 保留最大连通域
5. 去退化面、简化、补洞
6. 导出 `static_mesh.ply`

### 8.2 建议输出
```text
mesh_outputs/<run_name>/<seq_name>/
  static_points.ply
  dynamic_points.ply
  static_mesh_raw.ply
  static_mesh_clean.ply
  metrics.json
  params.yaml
```

---

## 9. 验收标准

1. 训练稳定：loss 可下降、无 NaN、无频繁中断
2. 导出稳定：字段齐全，时间索引一致，样本可回溯
3. 分离有效：静态场景下 dynamic 点占比低，动态主体集中在 dynamic 点云
4. mesh 可用：连通、孔洞少、背景结构完整

---

## 10. 推荐里程碑

- M1：训练稳定（500~2000 step）
- M2：导出脚本完成并产出 `.npz`
- M3：静/动态分离完成并导出两类点云
- M4：第一版静态 mesh 产出
- M5：参数调优并固定默认配置

---

## 11. 常见问题与排查

1. **显存爆掉**：先减 `num_queries`，再减 `S/N`，最后调 `img_size`
2. **分离结果噪声多**：先提高 `visibility/confidence` 阈值，再调静态阈值
3. **mesh 破碎**：增大体素、加强连通域过滤、补洞后再简化
4. **坐标错位**：先单帧可视化验证坐标变换，再跑整段序列

---

## 12. 后续增强（可选）

在上述闭环稳定后，再做以下增强：
1. 回补并统一 `train/val/test` 评估链路
2. 将监督语义逐步对齐到 query 级 `gt_3d`
3. 增加基于语义或运动一致性的静动态联合判定

本指南优先工程可落地性：先有结果，再做统一与精化。