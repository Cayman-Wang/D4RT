# D4RT：4D 重建 Transformer

D4RT（4D Reconstruction Transformer）的实现，用于从视频序列进行 4D 重建。

## 概述

D4RT 是一个基于 Transformer 的 4D 重建模型，采用了：
- **基于 Query 的解码机制**：相互独立的查询向量，仅对编码器特征进行注意力计算
- **维度解耦**：在查询向量中解耦空间维度（u, v）与时间维度（t_src, t_tgt, t_cam）
- **编码器-解码器架构**：ViT 编码器（交替局部/全局注意力）+ 轻量级交叉注意力解码器

## 关键特性

### 编码器
- Vision Transformer（ViT），在帧内局部注意力与全局自注意力之间交替
- 额外引入一个 token 用于编码原始视频宽高比
- 固定方形分辨率（256x256）

### Query 构建
- 归一化 2D 坐标（u, v）+ Fourier 特征嵌入
- 三个时间维度采用可学习的离散嵌入：t_src（源帧）、t_tgt（目标帧）、t_cam（相机参考帧）
- 局部 RGB patch 嵌入（以查询位置为中心的 9x9 patch）

### 解码器
- 轻量级交叉注意力 Transformer（6-8 层）
- 独立查询机制：查询之间不相互作用，仅关注编码器特征
- 输出：通过线性投影得到 3D 坐标

### 损失函数
- **主损失（L_3D）**：L1 损失，带预处理（按平均深度归一化）与变换（sign(x) * log(1+|x|)）
- **辅助损失**：
  - 2D 投影损失
  - 表面法向余弦相似度损失
  - 可见性预测（Binary Cross-Entropy）
  - 运动位移损失
  - 置信度惩罚（-log(c)）

### 训练策略
- 每个 batch 随机采样 N=2048 个查询
- 30% 查询采样自深度不连续区域或运动边界（Sobel 算子）
- 40% 样本满足 t_tgt = t_cam
- 优化器：AdamW（weight decay 0.03）
- 学习率调度：余弦退火（LR：1e-4 → 1e-6）

## 安装与环境准备（先激活环境）

> 下述流程统一采用“先激活环境，再运行脚本”，不使用 `conda run -n d4rt ...`。

```bash
cd /home/grasp/Desktop/wym-project/4d-gaussgym/D4RT
conda activate d4rt
pip install -r requirements.txt
python -V
which python
```

## 项目结构（与动静分离相关）

```
D4RT/
├── d4rt/
│   ├── models/
│   │   └── d4rt_model.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── datamodule.py
│   ├── separation/
│   │   ├── motion_score.py
│   │   ├── instance_tracker.py
│   │   ├── mesh_builder.py
│   │   └── io_contract.py
│   ├── train.py
│   └── test.py
├── scripts/
│   ├── train_d4rt.py
│   ├── test_d4rt.py
│   ├── export_separation_stream.py
│   ├── run_separation_replay.py
│   ├── build_separation_meshes.py
│   ├── visualize_separation_frame.py
│   ├── visualize_separation_sequence.py
│   ├── visualize_separation_timeline.py
│   ├── visualize_dataset_rgbd_sequence.py
│   └── visualize_dataset_rgbd_timeline.py
├── pointcloud_result_guide_zh.md
├── rgbd_scene_guide_zh.md
└── README_zh.md
```

## 从零运行完整流程：D4RT 4D 重建 → 静态/动态点云分离

### 1) 设置数据路径与输出目录

```bash
export DATA_ROOT=<你的PointOdyssey根目录>
export OUT_ROOT=outputs/d4rt_separation_pipeline
mkdir -p ${OUT_ROOT}
```

建议先确认数据目录存在：

```bash
ls ${DATA_ROOT}/train
ls ${DATA_ROOT}/val
```

### 2) 从零训练并获取 ckpt（先 smoke 跑通）

> 关键约束：`num_queries <= N`。  
> 下面这组参数已经在 24G 显存机器上实测可跑通（约 4 分钟/epoch）。  
> 注意：当前代码里 Query 特征维度默认是 512，请保持 `--decoder_dim 512`。

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_d4rt.py \
  --dataset_location ${DATA_ROOT} \
  --train_dset train \
  --quick \
  --S 4 \
  --N 96 \
  --num_queries 96 \
  --img_size 160 \
  --strides 4 \
  --clip_step 16 \
  --encoder_embed_dim 512 \
  --encoder_depth 6 \
  --encoder_num_heads 8 \
  --decoder_dim 512 \
  --decoder_num_heads 8 \
  --decoder_num_layers 4 \
  --batch_size 1 \
  --num_workers 0 \
  --max_epochs 1 \
  --devices 1 \
  --accelerator gpu \
  --precision 16-mixed \
  --log_dir ${OUT_ROOT}/train_smoke
```

查找并设置 ckpt：

```bash
find ${OUT_ROOT}/train_smoke -type f -name "*.ckpt"
export CKPT=$(find ${OUT_ROOT}/train_smoke -type f -name "*.ckpt" | sort | tail -n 1)
echo ${CKPT}
```

`--ckpt` 文件即上一步训练产生的 Lightning checkpoint。

### 3) 导出分离输入流（world 坐标）

> `--img_size` 必须与训练时一致（本例是 `160`），否则会出现 checkpoint 位置编码尺寸不匹配。

```bash
python scripts/export_separation_stream.py \
  --test_data_path ${DATA_ROOT} \
  --test_dset val \
  --ckpt ${CKPT} \
  --output_npz ${OUT_ROOT}/separation_stream_smoke.npz \
  --quick \
  --img_size 160 \
  --S 4 \
  --N 96 \
  --num_queries 96 \
  --strides 4 \
  --clip_step 16 \
  --max_clips 5 \
  --batch_size 1 \
  --num_workers 0 \
  --device auto
```

### 4) 运行静态/动态点云分离回放

仅统计校验（不落盘每帧）：

```bash
python scripts/run_separation_replay.py \
  --input_npz ${OUT_ROOT}/separation_stream_smoke.npz \
  --output_dir ${OUT_ROOT}/replay_smoke_dry \
  --dry_run \
  --save_json
```

完整落盘（保存每帧分离结果）：

```bash
python scripts/run_separation_replay.py \
  --input_npz ${OUT_ROOT}/separation_stream_smoke.npz \
  --output_dir ${OUT_ROOT}/replay_smoke_full
```

### 4.5) 构建 M3a mesh smoke 结果

> 该步骤只验证 replay -> mesh 的最小闭环，不代表已经达到高质量 mesh 阶段。
> 默认会把 `instance_id=-1` 的动态点也作为 smoke 阶段的 fallback 实例流，避免稀疏 replay 时整条链路空跑。

```bash
python scripts/build_separation_meshes.py \
  --frames_dir ${OUT_ROOT}/replay_smoke_full/frames \
  --output_dir ${OUT_ROOT}/mesh_smoke
```

### 5) 验收检查

```bash
ls -lh ${OUT_ROOT}/separation_stream_smoke.npz
ls -lh ${OUT_ROOT}/replay_smoke_dry/summary.json
ls -lh ${OUT_ROOT}/replay_smoke_full/summary.json
ls ${OUT_ROOT}/replay_smoke_full/frames | head
ls -lh ${OUT_ROOT}/mesh_smoke/mesh_summary.json
find ${OUT_ROOT}/mesh_smoke/meshes -type f | head
```

请在 `summary.json` 中重点确认：

- `total_static_points > 0`
- `total_dynamic_points > 0`

### 6) 可视化查看分离后的点云

如果尚未安装可视化依赖：

```bash
pip install open3d matplotlib
```

可视化第 0 帧（优先使用 Open3D）：

```bash
python scripts/visualize_separation_frame.py \
  --frames_dir ${OUT_ROOT}/replay_smoke_full/frames \
  --frame_index 0 \
  --backend open3d
```

如需优先显示真实 RGB（缺失时自动回退语义色）：

```bash
python scripts/visualize_separation_frame.py \
  --frames_dir ${OUT_ROOT}/replay_smoke_full/frames \
  --frame_index 0 \
  --backend open3d \
  --color_mode rgb
```

如果你在远程环境没有图形窗口，可改用 matplotlib 并导出图片：

```bash
python scripts/visualize_separation_frame.py \
  --frames_dir ${OUT_ROOT}/replay_smoke_full/frames \
  --frame_index 0 \
  --backend matplotlib \
  --save_png ${OUT_ROOT}/replay_smoke_full/frame_000000.png \
  --no_show
```

也可以直接指定单个帧文件：

```bash
python scripts/visualize_separation_frame.py \
  --frame_npz ${OUT_ROOT}/replay_smoke_full/frames/frame_000000.npz \
  --backend auto
```

如果你想看“序列级完整静态场景 + 动态分离效果”，不要只看单帧。新脚本会把多帧静态点累计到同一 world 坐标系，并支持把动态点按最近窗口或全序列累计后导出为辅助查看文件：

```bash
python scripts/visualize_separation_sequence.py \
  --frames_dir ${OUT_ROOT}/replay_smoke_full/frames \
  --dynamic_mode window \
  --dynamic_window 4 \
  --voxel_size 0.02 \
  --color_mode rgb \
  --export_static_ply ${OUT_ROOT}/replay_smoke_full/static_scene_accumulated.ply \
  --export_dynamic_ply ${OUT_ROOT}/replay_smoke_full/dynamic_window_last4.ply \
  --export_combined_ply ${OUT_ROOT}/replay_smoke_full/combined_scene_window_last4.ply \
  --export_instances_dir ${OUT_ROOT}/replay_smoke_full/dynamic_instances \
  --export_summary_json ${OUT_ROOT}/replay_smoke_full/sequence_summary_window4.json \
  --backend open3d
```

远程无图形环境时可以只导出，不开窗口：

```bash
python scripts/visualize_separation_sequence.py \
  --frames_dir ${OUT_ROOT}/replay_smoke_full/frames \
  --dynamic_mode window \
  --dynamic_window 4 \
  --voxel_size 0.02 \
  --export_static_ply ${OUT_ROOT}/replay_smoke_full/static_scene_accumulated.ply \
  --export_dynamic_ply ${OUT_ROOT}/replay_smoke_full/dynamic_window_last4.ply \
  --export_combined_ply ${OUT_ROOT}/replay_smoke_full/combined_scene_window_last4.ply \
  --export_instances_dir ${OUT_ROOT}/replay_smoke_full/dynamic_instances \
  --export_summary_json ${OUT_ROOT}/replay_smoke_full/sequence_summary_window4.json \
  --backend none
```

注意：`run_separation_replay.py` 的正式产物只有 `summary.json + frames/frame_*.npz`。上面这些 `*.ply` 和 `sequence_summary_window4.json` 都是 `visualize_separation_sequence.py` 的辅助导出，用于人工查看，不是 M2.5 / M3 门禁的最小必备证据。

推荐默认理解：

- `static_scene_accumulated.ply`：整段静态点累计后的“完整场景”查看入口
- `dynamic_window_last4.ply`：最近几帧动态点的短窗口累计，避免全序列动态拖尾
- `dynamic_instances/instance_*.ply`：按 `instance_id` 拆开的动态实例点云
- `combined_scene_window_last4.ply`：静态灰色 + 动态彩色的总览点云
- `sequence_summary_window4.json`：这次辅助导出的统计摘要，不等同于 `run_separation_replay.py` 的 `summary.json`

如果你想拖动时间条查看不同时间点的动态点云，不要对 `.ply` 做时间播放，而是直接读取 `replay_full/frames/frame_*.npz`：

```bash
python scripts/visualize_separation_timeline.py \
  --frames_dir ${OUT_ROOT}/replay_smoke_full/frames \
  --static_mode all \
  --dynamic_mode window \
  --dynamic_window 4 \
  --color_mode rgb
```

如果你要检查 M3a 的 mesh smoke 产物，直接看下面这些真实存在的输出：

- `${OUT_ROOT}/mesh_smoke/mesh_summary.json`
- `${OUT_ROOT}/mesh_smoke/frames/frame_*.npz`（包含 `static_mesh_path` / `dynamic_meshes_json`）
- `${OUT_ROOT}/mesh_smoke/meshes/static/*.ply`
- `${OUT_ROOT}/mesh_smoke/meshes/dynamic/instance_*/dynamic_frame_*.ply`

补充说明：

- `--static_mode all`：静态背景固定为全序列累计场景
- `--static_mode upto`：静态背景会随着时间条逐步累积
- `--static_mode current`：只看当前帧静态点
- `--static_mode none`：只看动态点
- `--color_mode semantic`：默认语义配色（兼容旧流程）
- `--color_mode rgb`：优先使用真实 RGB；若旧 NPZ 不含 `*_colors_rgb` 字段会自动回退语义色

关于“怎么看当前结果是否合理、什么时候该继续训练”的判断方法，请直接看：[`pointcloud_result_guide_zh.md`](./pointcloud_result_guide_zh.md)。

如果你要看“数据集原始 RGB 外观”的稠密点云（而不是分离语义色），请看：[`rgbd_scene_guide_zh.md`](./rgbd_scene_guide_zh.md)。

如果你要看“数据集 RGBD 的 4D 动态变化”（静态全局 + 动态滑窗时间轴），直接运行：

```bash
python scripts/visualize_dataset_rgbd_timeline.py \
  --dataset_root ${DATA_ROOT} \
  --dset val \
  --sequence ani10_new_f \
  --start_frame 0 \
  --end_frame 128 \
  --frame_stride 1 \
  --pixel_stride 1 \
  --dynamic_mode window \
  --dynamic_window 8 \
  --voxel_size 0.01 \
  --max_static_points 250000 \
  --max_dynamic_points 150000 \
  --backend matplotlib
```

无 GUI 导出（逐帧 PLY + 摘要 JSON）：

```bash
python scripts/visualize_dataset_rgbd_timeline.py \
  --dataset_root ${DATA_ROOT} \
  --dset val \
  --sequence ani10_new_f \
  --max_frames 64 \
  --dynamic_mode window \
  --dynamic_window 8 \
  --backend none \
  --export_frames_dir ${OUT_ROOT}/rgbd_timeline_frames \
  --export_summary_json ${OUT_ROOT}/rgbd_timeline_summary.json
```

### 7) 正式训练建议（24G 显存三档参数 + 实测高利用率档）

Smoke 跑通后，建议升级到正式配置。先记住 4 条硬约束：

- 正式训练/导出都不要加 `--quick`
- 始终满足 `num_queries <= N`
- 当前代码里请固定 `--decoder_dim 512`
- 导出参数必须与训练保持一致：`img_size/S/N/num_queries/strides/clip_step`

#### 三档参数（RTX 4090 24G 建议）

- 稳过档（优先稳定）：`img_size=160, S=8, N=512, num_queries=512, encoder=512/6/8, decoder_layers=4, batch_size=2`
- 均衡档（质量/速度平衡）：`img_size=192, S=8, N=1024, num_queries=1024, encoder=768/12/12, decoder_layers=6, batch_size=2`
- 冲高档（质量优先）：`img_size=224, S=8, N=1536, num_queries=1536, encoder=1024/16/16, decoder_layers=8, batch_size=2`

#### 实测高利用率档（推荐你当前机器优先尝试）

> 在本机（RTX 4090 24G）`quick` 基准中，这组参数的 GPU 利用率最高且稳定可跑：  
> 平均利用率约 `26.5%`，P90 约 `44%`，显存峰值约 `15GB`。

- 高利用率档：`img_size=224, S=8, N=512, num_queries=512, encoder=1024/16/16, decoder_layers=8, batch_size=4, num_workers=12`

> 若冲高档 OOM，优先回退到均衡档；若仍不稳，再回退到稳过档。

#### 7.1 选择一档参数（示例：高利用率档）

```bash
export PROFILE=high_util

# high_util（实测推荐）
export IMG_SIZE=224
export S=8
export N=512
export NUM_QUERIES=512
export STRIDES="1 2 4"
export CLIP_STEP=2
export ENC_EMBED=1024
export ENC_DEPTH=16
export ENC_HEADS=16
export DEC_DIM=512
export DEC_HEADS=8
export DEC_LAYERS=8
export BATCH_SIZE=4
export NUM_WORKERS_TRAIN=12
export NUM_WORKERS_EXPORT=4
```

改成稳过档时，把上面变量替换为：

```bash
export PROFILE=stable
export IMG_SIZE=160
export S=8
export N=512
export NUM_QUERIES=512
export STRIDES="1 2 4"
export CLIP_STEP=2
export ENC_EMBED=512
export ENC_DEPTH=6
export ENC_HEADS=8
export DEC_DIM=512
export DEC_HEADS=8
export DEC_LAYERS=4
export BATCH_SIZE=2
export NUM_WORKERS_TRAIN=8
export NUM_WORKERS_EXPORT=4
```

改成冲高档时，把上面变量替换为：

```bash
export PROFILE=aggressive
export IMG_SIZE=224
export S=8
export N=1536
export NUM_QUERIES=1536
export STRIDES="1 2 4"
export CLIP_STEP=2
export ENC_EMBED=1024
export ENC_DEPTH=16
export ENC_HEADS=16
export DEC_DIM=512
export DEC_HEADS=8
export DEC_LAYERS=8
export BATCH_SIZE=2
export NUM_WORKERS_TRAIN=12
export NUM_WORKERS_EXPORT=2
```

改成均衡档时，把上面变量替换为：

```bash
export PROFILE=balanced
export IMG_SIZE=192
export S=8
export N=1024
export NUM_QUERIES=1024
export STRIDES="1 2 4"
export CLIP_STEP=2
export ENC_EMBED=768
export ENC_DEPTH=12
export ENC_HEADS=12
export DEC_DIM=512
export DEC_HEADS=8
export DEC_LAYERS=6
export BATCH_SIZE=2
export NUM_WORKERS_TRAIN=12
export NUM_WORKERS_EXPORT=4
```

#### 7.2 统一训练命令（按当前档位变量执行）

```bash
export OUT_ROOT=outputs/d4rt_formal_${PROFILE}
mkdir -p ${OUT_ROOT}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 多卡训练（如 --devices 2）建议加上，避免某些机器 NCCL 初始化后段错误
export NCCL_SOCKET_IFNAME=lo

python scripts/train_d4rt.py \
  --dataset_location ${DATA_ROOT} \
  --train_dset train \
  --S ${S} \
  --N ${N} \
  --num_queries ${NUM_QUERIES} \
  --img_size ${IMG_SIZE} \
  --strides ${STRIDES} \
  --clip_step ${CLIP_STEP} \
  --encoder_embed_dim ${ENC_EMBED} \
  --encoder_depth ${ENC_DEPTH} \
  --encoder_num_heads ${ENC_HEADS} \
  --decoder_dim ${DEC_DIM} \
  --decoder_num_heads ${DEC_HEADS} \
  --decoder_num_layers ${DEC_LAYERS} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS_TRAIN} \
  --max_epochs 20 \
  --devices 1 \
  --accelerator gpu \
  --precision 16-mixed \
  --log_dir ${OUT_ROOT}/train
```

多卡补充说明：

- 如果你用双卡，把命令中的 `--devices 1` 改为 `--devices 2`。
- `scripts/train_d4rt.py` 在 `--devices > 1` 且未显式传 `--strategy` 时，会自动使用 `ddp_find_unused_parameters_true`，以避免 DDP 的 unused parameters 报错。
- 如需手动指定，可在训练命令里添加：`--strategy ddp_find_unused_parameters_true`。

如需每个 epoch 做验证，可额外加上：`--val_dset val --use_val`（会略微降低训练吞吐）。

#### 7.3 导出与分离（按当前档位变量执行）

```bash
export CKPT=$(find ${OUT_ROOT}/train -type f -name "*.ckpt" | sort | tail -n 1)
echo ${CKPT}

python scripts/export_separation_stream.py \
  --test_data_path ${DATA_ROOT} \
  --test_dset val \
  --ckpt ${CKPT} \
  --output_npz ${OUT_ROOT}/separation_stream.npz \
  --img_size ${IMG_SIZE} \
  --S ${S} \
  --N ${N} \
  --num_queries ${NUM_QUERIES} \
  --strides ${STRIDES} \
  --clip_step ${CLIP_STEP} \
  --batch_size 1 \
  --num_workers ${NUM_WORKERS_EXPORT} \
  --device auto

python scripts/run_separation_replay.py \
  --input_npz ${OUT_ROOT}/separation_stream.npz \
  --output_dir ${OUT_ROOT}/replay_full
```

## 脚本输入输出契约

### `scripts/export_separation_stream.py`

输出 NPZ 主键：

- `points_world`：`(T, N, 3)`
- `motion_world`：`(T, N, 3)`
- `confidence`：`(T, N)`
- `visibility`：`(T, N)`
- `point_ids`：`(T, N)`
- `timestamps`：`(T,)`

（另含辅助键：`valid_mask`, `frame_point_counts`, `clip_indices`, `clip_frame_indices`, `annotation_paths`）

### `scripts/run_separation_replay.py`

输入：读取上面的 NPZ 主键。  
输出：

- `summary.json`
- `frames/frame_*.npz`（非 `--dry_run`）

### `scripts/build_separation_meshes.py`

输入：读取 `run_separation_replay.py` 导出的 `frames/frame_*.npz`。
输出：

- `mesh_summary.json`
- `frames/frame_*.npz`（逻辑上回填 `static_mesh_path` 与 `dynamic_meshes`；NPZ 落盘字段为 `static_mesh_path` 与 `dynamic_meshes_json`）
- `meshes/static/*.ply`
- `meshes/dynamic/instance_*/dynamic_frame_*.ply`

## 测试场景与验收标准

- 场景 A：GPU smoke（`max_epochs=1, S=4, N=96, num_queries=96, img_size=160`）能完整跑通三阶段流程。
- 场景 B：`--dry_run --save_json` 仅生成统计 JSON，不生成逐帧 NPZ。
- 场景 C：full-run 生成 `frames/frame_*.npz`。
- 场景 E：`build_separation_meshes.py` 基于 replay 结果生成非空 static/dynamic mesh 文件。
- 场景 D（CPU 兜底）：
  - 训练：将 `--accelerator gpu` 改为 `--accelerator cpu`
  - 导出：将 `--device auto` 改为 `--device cpu`

## 常见问题排查（FAQ）

1. 报错：`--num_queries (...) must be <= --N (...)`
   - 处理：确保 `N >= num_queries`（例如都设为 256）。

2. 找不到 `*.ckpt`
   - 处理：先执行 `find ${OUT_ROOT}/train_smoke -type f -name "*.ckpt"`；若为空，检查训练是否完成或 `--log_dir` 是否正确。

3. CUDA 不可用
   - 处理：使用 CPU 兜底参数继续 smoke 验证（见上节场景 D）。

4. 数据集路径错误或为空
   - 处理：先检查 `${DATA_ROOT}/train` 与 `${DATA_ROOT}/val` 是否存在并含数据。

## 默认假设

- 数据集为 PointOdyssey 格式。
- 先 smoke 再正式训练，优先保证流程跑通。
- 动静分离链路基于当前仓库中的 `export_separation_stream.py` 与 `run_separation_replay.py`。

## 引用

如果你使用了本代码，请引用 D4RT 论文。
