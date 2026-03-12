# D4RT 复现与静态 Mesh 重建指南（PointOdyssey，单卡 24GB）

## 1. 目标与路线
本指南分两步：
1. **先复现 D4RT 训练链路**（能稳定训练/验证，不先追论文最优指标）
2. 在稳定模型上做 **静/动态点云分离**，并把静态点云重建成 mesh

### 1.1 当前研究排期说明
当前静态 mesh 研究已经固定为 `S0 / S1a / S1b / S2a / S2b` 分层推进，详见 `research/guides/d4rt_static_mesh_research_reorder_zh.md`。

当前默认执行顺序是：
- `S0`：保留当前 smoke mesh，仅做链路健康检查
- `S1a`：`replay static points -> volumetric/static mesh`，是正式生产线第一优先
- `S2a`：`raw RGBD -> dense static mesh` low-cost oracle，只作诊断对照
- `S1b`：只在 authority replay 达到 `D4RT/pointcloud_result_guide_zh.md` 的“情况 C”后再进入
- `S2b`：只用于上界判断和论文 related work，不进入当前主线

---

## 2. 先复现 D4RT：最小可跑通路径

### 2.1 环境准备
在 `D4RT/` 单独创建环境并安装依赖：

```bash
cd D4RT
pip install -r requirements.txt
```

> 建议先不要混用 `gauss_gym` 环境，避免依赖冲突。

### 2.2 数据准备（PointOdyssey）
确保目录结构满足 `PointOdysseyDataset` 的读取逻辑：

```text
<dataset_location>/<split>/<seq>/
  rgbs/
  depths/
  normals/
  info.npz
  anno.npz
```

`<split>` 通常是 `train` 和 `val`。

### 2.3 建议的首轮训练命令（24GB 显存）

```bash
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

首轮目标：**先稳定跑通**，再逐步放大模型与查询数。

### 2.4 复现时必须关注的关键点
1. **README 参数与脚本参数不一致**：以 `scripts/train_d4rt.py` 的参数为准。
2. **`scripts/test_d4rt.py` 默认走 `D4RTDataModule`，而 `D4RTDataset.__getitem__` 仍是占位实现**，所以先用 `train.py --use_val` 的验证链路更稳。
3. **GT 语义一致性要盯紧**：
   - 数据集里已有按 `t_tgt -> t_cam` 定义的 `gt_3d`
   - 训练里当前是用深度反投影 `extract_gt_data()` 生成 GT
   两套语义不一致会影响时序几何稳定性与后续静动态分离。
4. **motion 头可用性**：当前训练通常没有 `gt_motion`，motion loss 常不生效，不要把 motion 头当主监督指标。
5. **先看稳定性，再看指标**：先确保 loss 不炸、无 NaN、显存稳定，再追指标。

---

## 3. 从 D4RT 到静态 Mesh：推荐流程

### 3.1 推理导出
对同一参考视角（固定 `t_cam`）采样多个 `t_tgt`，导出：
- `coords_3d`
- `visibility`
- `confidence`
- `t_src, t_tgt, t_cam`
- `cams_T_world`

### 3.2 统一坐标系
把预测点从相机坐标系转到世界坐标系：

- `p_world = inv(cams_T_world[t_cam]) @ p_cam`

### 3.3 静/动态分离
1. 先做置信过滤：`visibility > 0.5` 且 `confidence > 0.5`
2. 对每个 query 轨迹计算世界坐标时序离散度（如 MAD/中位位移）
3. 小于阈值判静态（建议起始 `0.02m`，再做 `0.01/0.02/0.05` 网格搜索）

### 3.4 静态点云建 Mesh（Open3D）
- 体素降采样（`0.01~0.02m`）
- 法向估计
- Ball Pivoting（半径 `[0.5v, v, 2v]`）
- 仅保留最大连通域，去退化面，简化，补洞
- 导出 `static_mesh.ply`

---

## 4. 验收标准（建议）
1. 训练 loss 稳定下降，无 NaN。
2. 验证集可稳定前向，不频繁 `sampling failed`。
3. 静态场景中动态点占比低，动态物体主要落在 dynamic 点云。
4. `static_mesh.ply` 连通、洞少、背景结构完整。

---

## 5. 建议的执行顺序（最小风险）
1. 用小配置跑通 500~2000 step（确认训练稳定）
2. 跑 1~2 万 step 看趋势（确认没有明显退化）
3. 增加导出脚本（先导出点云，不急着建 mesh）
4. 接入静动态分离
5. 最后做静态 mesh 重建与参数调优
