# RGBD 原始外观点云查看指南（PointOdyssey）

这份指南用于解决两个问题：
- 想看“数据集原始颜色/纹理外观”的场景点云，而不是分离语义色（静态灰/动态红）。
- 想拿到比 query 点更稠密的点云做对照验证。

核心做法：直接从 `rgbs/ + depths/ + anno.npz` 做 RGBD 回投，不走分离链路。

## 1. 先激活环境

```bash
cd /home/grasp/Desktop/wym-project/4d-gaussgym/D4RT
source /home/grasp/miniconda3/etc/profile.d/conda.sh
conda activate d4rt
```

## 2. Smoke 命令（先看一段）

```bash
export DATA_ROOT=/mnt/windows_data2/wym-project/4d-gaussgym/datasets/PointOdyssey
export OUT_ROOT=outputs/rgbd_scene_smoke
mkdir -p "${OUT_ROOT}"

python scripts/visualize_dataset_rgbd_sequence.py \
  --dataset_root "${DATA_ROOT}" \
  --dset val \
  --sequence ani10_new_f \
  --start_frame 0 \
  --end_frame 64 \
  --frame_stride 2 \
  --pixel_stride 1 \
  --voxel_size 0.01 \
  --max_points 1500000 \
  --output_ply "${OUT_ROOT}/ani10_new_f_rgbd_smoke.ply" \
  --backend none
```

说明：
- `--backend none` 适合远程/无 GUI；只导出文件不弹窗。
- `--pixel_stride 1` 是最稠密像素采样。
- `--voxel_size 0.01` 会做 1cm 体素压缩，减少重复点。

## 3. 正式导出（更完整）

```bash
export DATA_ROOT=/mnt/windows_data2/wym-project/4d-gaussgym/datasets/PointOdyssey
export OUT_ROOT=outputs/rgbd_scene_formal
mkdir -p "${OUT_ROOT}"

python scripts/visualize_dataset_rgbd_sequence.py \
  --dataset_root "${DATA_ROOT}" \
  --dset val \
  --sequence ani10_new_f \
  --start_frame 0 \
  --end_frame -1 \
  --frame_stride 1 \
  --pixel_stride 1 \
  --max_frames 256 \
  --voxel_size 0.01 \
  --max_points 3000000 \
  --output_ply "${OUT_ROOT}/ani10_new_f_rgbd_full.ply" \
  --backend none
```

建议先控制 `--max_frames`，确认效果后再增大。

## 4. 本地有 GUI 时直接预览

```bash
python scripts/visualize_dataset_rgbd_sequence.py \
  --dataset_root "${DATA_ROOT}" \
  --dset val \
  --sequence ani10_new_f \
  --start_frame 0 \
  --end_frame 64 \
  --frame_stride 2 \
  --pixel_stride 1 \
  --voxel_size 0.01 \
  --output_ply "${OUT_ROOT}/ani10_new_f_rgbd_preview.ply" \
  --backend open3d
```

## 5. 参数解释（只列关键）

- `--depth_scale`：深度缩放。默认 `1000/65535`，与 D4RT 当前 dataset loader 一致。
- `--frame_stride`：时间采样间隔，越大越快但时序覆盖更稀疏。
- `--pixel_stride`：像素采样间隔，`1` 最稠密。
- `--voxel_size`：世界坐标体素下采样，`<=0` 关闭。
- `--max_points`：最终点数上限，`<=0` 不限（会占较大内存）。

## 6. 常见问题

1. 输出点云为空  
检查是否选到了有效帧范围：`--start_frame/--end_frame`，以及深度阈值 `--depth_min/--depth_max`。

2. 点云颜色不对/偏差明显  
当前脚本直接使用同像素 RGB，不做光照补偿；如果你改了图像 resize/裁剪流程，需同步检查相机内参与坐标映射。

3. 运行内存占用过高  
优先调大 `--frame_stride` 或 `--pixel_stride`，并设置更小的 `--max_frames`、更大的 `--voxel_size`。

4. 预览不弹窗  
无桌面环境时使用 `--backend none`；有桌面环境需安装 `open3d` 并确保本机图形会话可用。

