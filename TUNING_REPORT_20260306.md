# D4RT PointOdyssey 训练调参报告（2026-03-06）

## 1. 目标
在不修改模型代码（仅调运行参数和运行时环境变量）的前提下，提升本机（2 x RTX 3090）训练吞吐，并选出用于正式全量训练的参数组合。

## 2. 实验环境
- 代码目录：`/home/wangyumu/wym-project/D4RT`
- 数据集目录：`/home/wangyumu/wym-project/datasets/PointOdyssey`
- Python：`/home/wangyumu/anaconda3/envs/d4rt/bin/python`
- GPU：2 x NVIDIA GeForce RTX 3090
- 主要基础参数：
  - `S=8, N=512, num_queries=512, img_size=224`
  - `strides=1 2 4, clip_step=2`
  - `encoder=1024/16/16, decoder=512/8/8`
  - `batch_size=4, max_epochs=20, precision=16-mixed`

## 3. 调参方法
### 3.1 调参维度
固定模型/数据参数，仅比较：
- `devices in {2, 1}`
- `num_workers in {6, 4, 2}`（2卡）和 `num_workers=4`（1卡）

### 3.2 评估口径
- 指标主排序：`samples_per_sec`（越高越好）
- 辅助指标：`steps_per_min`、GPU 利用采样统计
- 为保证可比性，统一跑到 `metrics.csv` 的 `step >= 199`（约 200 step）后停止采样

### 3.3 运行时环境变量
所有调参实验统一使用：
- `NCCL_SOCKET_IFNAME=lo`
- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `NUMEXPR_NUM_THREADS=1`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 4. 实验结果
有效结果（200 step 对齐）如下：

| Run | devices | num_workers | steps/min | samples/s | 结论 |
|---|---:|---:|---:|---:|---|
| R1_2gpu_nw6 | 2 | 6 | 11.3205 | 1.5094 | 最优 |
| R2_2gpu_nw4_retry | 2 | 4 | 7.9805 | 1.0641 | 次优 |
| R3_2gpu_nw2_retry | 2 | 2 | 4.2499 | 0.5667 | 明显偏慢 |
| R4_1gpu_nw4 | 1 | 4 | 8.6329 | 0.5755 | 低于双卡最优 |

结论：
- 本机最佳组合是 `devices=2, num_workers=6`。
- 在当前数据/模型设置下，`num_workers` 从 6 降到 4 或 2 都会显著降低吞吐。

## 5. 本次遇到的问题与处理
1. 多卡训练初始化后出现段错误（Segmentation fault）
- 现象：NCCL 初始化后进程直接崩溃。
- 处理：设置 `NCCL_SOCKET_IFNAME=lo` 后稳定运行。

2. 实验间残留进程导致假性 OOM/污染
- 现象：前一组异常结束后，残留 DDP 子进程占显存，下一组启动失败或性能异常。
- 处理：每轮实验前后显式排查并清理残留 `train_d4rt.py` 进程。

3. `--max_steps` 与 Trainer 停止条件不一致
- 现象：脚本参数中有 `--max_steps`，但 Trainer 未设置该停止条件，实验难以按固定 step 自动停止。
- 处理：外部监控 `metrics.csv`，达到 `step>=199` 后终止进程并记录结果。

4. GPU 利用率看起来偏低
- 原因：该任务以短 burst 计算 + 数据读取/调度等待为主，1 秒粒度采样会看到很多 0 或低值；并且双卡存在 rank 不完全同步高负载的时刻。
- 处理：使用吞吐指标（samples/s）作为主判据，而非仅看瞬时 util。

## 6. 最优参数（用于正式训练）
建议正式训练使用：
- `--devices 2`
- `--num_workers 6`
- 其余参数保持用户给定配置

推荐启动命令：

```bash
export DATA_ROOT=/home/wangyumu/wym-project/datasets/PointOdyssey
export OUT_ROOT=/home/wangyumu/wym-project/D4RT/outputs

export NCCL_SOCKET_IFNAME=lo
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/wangyumu/anaconda3/envs/d4rt/bin/python scripts/train_d4rt.py \
  --dataset_location ${DATA_ROOT} \
  --train_dset train \
  --S 8 --N 512 --num_queries 512 --img_size 224 \
  --strides 1 2 4 --clip_step 2 \
  --encoder_embed_dim 1024 --encoder_depth 16 --encoder_num_heads 16 \
  --decoder_dim 512 --decoder_num_heads 8 --decoder_num_layers 8 \
  --batch_size 4 --num_workers 6 --max_epochs 20 \
  --devices 2 --accelerator gpu --precision 16-mixed \
  --log_dir ${OUT_ROOT}/train_best_2gpu_nw6
```

## 7. 原始结果文件
- 主调参目录：`/home/wangyumu/wym-project/D4RT/outputs/tune_runtime_stepcap_v2_20260306_162533`
- 汇总：
  - `summary.csv`
  - `retries/retry_summary.csv`
- 关键 run 结果：
  - `R1_2gpu_nw6/result.json`
  - `retries/R2_2gpu_nw4_retry/result.json`
  - `retries/R3_2gpu_nw2_retry/result.json`
  - `R4_1gpu_nw4/result.json`

## 8. 正式训练启动记录
- 启动时间：2026-03-06 18:25（Asia/Shanghai）
- 启动目录：`/home/wangyumu/wym-project/D4RT/outputs/train_best_2gpu_nw6_full_20260306_182552`
- 启动参数：`devices=2, num_workers=6`（其余与第 6 节一致）
- 启动后检查结果：
  - DDP 初始化成功（2/2 进程注册完成）
  - 数据集加载成功（`305912 clips`）
  - `metrics.csv` 已持续写入（已记录到 step 89，且持续增长）
  - 训练损失正常波动，无崩溃、无 OOM、无段错误
  - 观察到少量 `warning: sampling failed`（数据采样失败后会回退到零样本，不会中断训练），建议后续单独统计该告警比例

## 9. 2026-03-09 在线监控（续跑）
- 训练目录：`/home/wangyumu/wym-project/D4RT/outputs/train_best_2gpu_nw6_full_20260309_101100`
- 监控口径：先跟到 `step=199` 做阶段稳定性与吞吐估计
- 监控结果（`step=199`）：
  - `steps_per_min_est = 12.9932`
  - `samples_per_sec_est = 1.7324`（按全局 batch = `4 x 2` 计算）
  - GPU 平均利用率（采样窗口）：`gpu0_avg=19.05`, `gpu1_avg=85.1`
- 阶段结论：
  - 训练过程稳定（无 segfault / OOM / 进程退出）
  - loss 按预期下降并波动正常
  - 仍存在双卡负载不对称（与前述观察一致）
- 完成时间估计（以当前吞吐外推）：
  - 首个 epoch 结束（epoch 计数从 0 进入 1）预计约在 `2026-03-11 11:14` 左右
  - 若按“epoch=1 完整结束”（即第二个 epoch 结束）预计约在 `2026-03-13 12:17` 左右
