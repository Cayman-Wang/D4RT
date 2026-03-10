# D4RT 动静点云结果判读指南

本文档用于判断 D4RT 在 `run_separation_replay.py` 之后输出的静态/动态点云结果是否合理，以及什么时候应该继续训练或扩大正式导出规模，而不是过早进入 mesh 调试。

当前权威流程里，formal 门禁的最小必备证据是 `replay_full/summary.json + replay_full/frames/frame_*.npz`。`visualize_separation_sequence.py` 导出的 `*.ply` 和 `sequence_summary*.json` 只用于辅助人工查看，不是 formal replay 的必备产物。

## 1. 先看什么，不要先看什么

先看：
- `replay_full/summary.json`
- `replay_full/frames/frame_*.npz` 的时间序列变化
- `visualize_separation_sequence.py` 按需导出的 `static_scene_accumulated.ply` / `dynamic_window_last4.ply` / `combined_scene_window_last4.ply`
- 存在时再看 `sequence_summary*.json`

不要只看：
- 单帧 `frame_000000.npz`
- 单帧可视化结果

原因：单帧动态点天然稀疏，只用单帧很容易误判“模型完全没学到”或“点云质量很差”。很多情况下，单帧稀疏是正常现象，要结合时间序列和累计结果一起判断。

## 2. 推荐查看顺序

### 2.1 看最终完整场景
目标：确认完整静态场景有没有成形，动态轨迹整体是否合理。

重点输入：
- `replay_full/summary.json`
- `replay_full/frames/frame_*.npz`

如需把完整静态场景和动态窗口累计导出成 PLY，可先运行：

```bash
python scripts/visualize_separation_sequence.py \
  --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames \
  --dynamic_mode window \
  --dynamic_window 4 \
  --color_mode rgb \
  --export_static_ply outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/static_scene_accumulated.ply \
  --export_dynamic_ply outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/dynamic_window_last4.ply \
  --export_combined_ply outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/combined_scene_window_last4.ply \
  --export_summary_json outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/sequence_summary_window4.json \
  --backend none
```

### 2.2 看分离效果
目标：确认静态背景和动态目标是否被正确分开。

推荐命令：

```bash
python scripts/visualize_separation_timeline.py \
  --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames \
  --static_mode all \
  --dynamic_mode window \
  --dynamic_window 4 \
  --color_mode rgb
```

含义：
- 静态点云保持为完整背景
- 动态点云显示最近 4 帧
- 拖动时间条时，主要观察动态点是否沿着真实运动目标变化
- `--color_mode rgb` 会优先显示真实 RGB；旧帧文件若无颜色字段会自动回退语义色

### 2.3 看时序构建过程
目标：确认静态场景是不是在逐步变完整，而不是一直停留在极稀疏状态。

推荐命令：

```bash
python scripts/visualize_separation_timeline.py \
  --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames \
  --static_mode upto \
  --dynamic_mode window \
  --dynamic_window 4
```

### 2.4 看单帧动态是否过稀
目标：确认“动态点少”到底是正常单帧稀疏，还是模型本身确实太弱。

推荐命令：

```bash
python scripts/visualize_separation_timeline.py \
  --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames \
  --static_mode none \
  --dynamic_mode frame
```

## 3. 什么表现算“基本正常”

下面这些表现说明当前 separation / replay 链路基本是正常的。

### 3.1 动态点基本贴着运动目标走
- 动态点主要落在移动物体周围，而不是大面积撒在墙、地面、桌面等静态背景上。
- 拖动时间条时，动态点的位置变化和物体运动方向基本一致。

### 3.2 动态点随时间变化连续
- 从一帧切到下一帧，动态点不会大范围瞬移。
- 不会频繁整段消失，然后又突然出现在完全不相关的位置。

### 3.3 静态背景主轮廓能认出来
- 墙面、地面、桌面、障碍物的大体结构能看出来。
- 没有明显的“双层墙”、“整片漂移”、“厚重重影”。

### 3.4 全序列或短窗口动态轨迹大体合理
- 看 timeline 的 `dynamic_mode window/all`，或额外导出的动态/combined PLY 时，动态轨迹沿着一条合理路线分布。
- 不应该出现“整场景到处洒点”的情况。

### 3.5 近邻窗口下仍能看到动态团块
- 用 `--dynamic_mode window --dynamic_window 4` 时，动态点最好能形成一个局部团块。
- 如果完全只剩几个离散点，说明单帧预测仍偏弱。

## 4. 什么表现说明还需要继续训练或扩大导出规模

下面这些现象更像是 checkpoint 不够强、导出规模不够大，或者输入点密度不足。

### 4.1 单帧动态几乎只有零散点
- `dynamic_mode frame` 下经常只有个位数到十几个点。
- 目标轮廓完全看不出来。

说明：
- 这时 mesh 基本没有可用输入。

### 4.2 短窗口动态仍然非常稀疏
- `dynamic_mode window --dynamic_window 4` 后仍然只有零散点。
- 看不到稳定的目标形状或局部团块。

说明：
- 当前结果更适合验证“分离逻辑跑通”，不适合做高质量 mesh。

### 4.3 静态场景只剩骨架，没有连续表面
- `visualize_separation_timeline.py --static_mode all/upto` 或导出的 `static_scene_accumulated.ply` 里只有非常稀的点骨架。
- 地面、墙面、桌面都不连续，缺口很多。

说明：
- 这通常不是 mesh builder 的问题，而是输入点云密度不够。

### 4.4 全序列动态像“红雾/拖带”
- 看 `dynamic_mode all` 或全序列累计的动态 PLY 时，动态点不是沿着物体走，而是拉出很厚的拖尾。
- 每一帧的位置误差较大，导致累积后形成模糊带。

说明：
- 模型定位噪声偏大，继续堆 mesh 只会得到 broken mesh。

### 4.5 动静错分明显
- 动态点大量粘在静态背景上。
- 静态点里混入明显移动物体。
- 拖动时间条时，经常看到同一块背景忽然被打成动态。

说明：
- 这时不应该先加训练轮数，而应先检查 replay 参数、阈值、checkpoint 是否匹配。

## 5. 统计指标怎么看

重点看以下文件：
- `replay_full/summary.json`
- `replay_full/frames/frame_*.npz`
- `replay_full/sequence_summary*.json`（仅当你显式运行 `visualize_separation_sequence.py --export_summary_json ...` 时存在）

### 5.1 `mean_active_tracks`
- 越稳定越好。
- 接近 1 表示当前序列里 tracker 基本能持续跟住一个动态实例。

### 5.2 `total_uncertain_points`
- 过高说明很多点落在“静/动都不够确定”的中间区间。
- 这会直接影响 mesh 输入质量。

### 5.3 `static_downsampled_points`
- 用来粗看完整静态场景是否已经足够致密。
- 如果这个数仍然很小，静态场景就很难变成连续表面。

### 5.4 `dynamic_downsampled_points`
- `window` 模式用于看“当前动态是否足够清楚”。
- `all` 模式用于看“整段动态轨迹是否足够完整”。

## 6. 对当前 authority replay 结果的判读示例

当前 authority 输出位于：
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/summary.json`
- `outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames/frame_*.npz`

当前统计：
- `processed_frames = 256`
- `total_static_points = 3734`
- `total_dynamic_points = 1549`
- `total_uncertain_points = 2141`
- `mean_active_tracks = 0.0`
- `dynamic_window_points(last4,total) = 24`
- `dynamic_points_per_frame(max) = 16`

这组结果更适合这样理解：
- `stream -> replay` 在 authority repo 内已经跑通，可以作为 `M3a mesh smoke` 的输入基线。
- 但它还没有达到“情况 C”。`mean_active_tracks = 0.0` 说明当前 tracker 没有形成稳定实例流。
- 最近窗口动态总点数只有 `24`，仍然更像“稀疏趋势信号”，不适合进入 `M3b` 的 mesh 质量阶段。
- 结论：保留 `M3a mesh smoke`，优先继续训练或扩大正式导出规格，而不是继续调 mesh 质量。

## 7. 什么时候继续训练，什么时候先别急着训练

### 应该继续训练或扩大正式导出规模
满足下面任一条件时，优先继续训练或提高正式导出规格：
- 单帧和短窗口动态都太稀。
- 静态累计场景仍然很碎。
- 想进入 mesh/M3，但当前输入点密度明显不够。

优先调整：
- 使用更强 checkpoint
- 正式训练时不要用 `--quick`
- 提高 `N`
- 提高 `num_queries`
- 增大 `max_clips`

### 不要先靠“加训练轮数”解决的问题
如果出现以下情况，先排查逻辑/参数，而不是盲目加 epoch：
- 动静错分严重
- 动态点总是粘背景
- 背景经常被错误打成动态
- replay 参数和导出数据不匹配

优先检查：
- `run_separation_replay.py` 的聚类参数
- 使用的 `ckpt` 是否和导出配置匹配
- `img_size / S / N / num_queries / strides / clip_step` 是否一致

## 8. 一个简单决策规则

可以用下面的经验规则快速判断：

### 情况 A：分离逻辑正常，但输入稀疏
表现：
- 动态点位置大体对
- 时序稳定
- 静态背景能认出轮廓
- 但点云整体还是稀

结论：
- 继续训练 / 提高正式导出规格
- 暂时不要花太多时间调 mesh

### 情况 B：分离逻辑本身不正常
表现：
- 动静大面积错分
- 动态点飘到背景上
- 时间条拖动时点云跳变严重

结论：
- 先调分离参数 / replay 参数 / checkpoint 匹配关系
- 先别进入 mesh builder

### 情况 C：点云已经较完整且稳定
表现：
- 静态场景连续性明显提升
- 动态窗口能形成稳定团块
- 轨迹不乱飘

结论：
- 可以进入 `M3b Mesh Quality` 或更高质量的 mesh 阶段

## 9. 推荐默认查看命令

看完整场景：

```bash
python scripts/visualize_separation_timeline.py \
  --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames \
  --static_mode all \
  --dynamic_mode window \
  --dynamic_window 4
```

看时序构建过程：

```bash
python scripts/visualize_separation_timeline.py \
  --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames \
  --static_mode upto \
  --dynamic_mode window \
  --dynamic_window 4
```

只看单帧动态是否过稀：

```bash
python scripts/visualize_separation_timeline.py \
  --frames_dir outputs/d4rt_formal_sequence/nonquick_val64_authority_gpu/replay_full/frames \
  --static_mode none \
  --dynamic_mode frame
```
