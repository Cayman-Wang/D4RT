#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Default paths for the current machine. Override with env vars if needed.
DEFAULT_PYTHON_BIN="/home/wangyumu/anaconda3/envs/d4rt/bin/python"
DEFAULT_DATA_ROOT="/home/wangyumu/wym-project/4d-gaussgym/datasets/PointOdyssey"
DEFAULT_OUTPUTS_ROOT="${REPO_ROOT}/outputs"
DEFAULT_RUN_PREFIX="m3_ckpt_hunt_balanced_2gpu"

PYTHON_BIN="${D4RT_PYTHON:-${DEFAULT_PYTHON_BIN}}"
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${DEFAULT_OUTPUTS_ROOT}}"
RUN_PREFIX="${RUN_PREFIX:-${DEFAULT_RUN_PREFIX}}"
RUN_ROOT="${RUN_ROOT:-}"

# Training defaults for checkpoint hunting on the current dual-GPU machine.
DEFAULT_TRAIN_VISIBLE_DEVICES="0,1"
DEFAULT_TRAIN_DEVICES="2"
DEFAULT_TRAIN_ACCELERATOR="gpu"
DEFAULT_TRAIN_PRECISION="16-mixed"
DEFAULT_TRAIN_BATCH_SIZE="2"
DEFAULT_TRAIN_NUM_WORKERS="6"
DEFAULT_TRAIN_MAX_EPOCHS="1"
DEFAULT_TRAIN_MAX_STEPS="800"
DEFAULT_TRAIN_WARMUP_STEPS="100"
DEFAULT_CHECKPOINT_EVERY="100"

TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES:-${DEFAULT_TRAIN_VISIBLE_DEVICES}}"
TRAIN_DEVICES="${TRAIN_DEVICES:-${DEFAULT_TRAIN_DEVICES}}"
TRAIN_ACCELERATOR="${TRAIN_ACCELERATOR:-${DEFAULT_TRAIN_ACCELERATOR}}"
TRAIN_PRECISION="${TRAIN_PRECISION:-${DEFAULT_TRAIN_PRECISION}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-${DEFAULT_TRAIN_NUM_WORKERS}}"
TRAIN_MAX_EPOCHS="${TRAIN_MAX_EPOCHS:-${DEFAULT_TRAIN_MAX_EPOCHS}}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-${DEFAULT_TRAIN_MAX_STEPS}}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-${DEFAULT_TRAIN_WARMUP_STEPS}}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-${DEFAULT_CHECKPOINT_EVERY}}"
RESUME_EXTRA_STEPS="${RESUME_EXTRA_STEPS:-800}"

# Data / model config already validated in this session.
DEFAULT_IMG_SIZE="192"
DEFAULT_S="8"
DEFAULT_N="1024"
DEFAULT_NUM_QUERIES="1024"
DEFAULT_STRIDES_TEXT="1 2 4"
DEFAULT_CLIP_STEP="2"
DEFAULT_ENCODER_EMBED_DIM="768"
DEFAULT_ENCODER_DEPTH="12"
DEFAULT_ENCODER_NUM_HEADS="12"
DEFAULT_DECODER_DIM="512"
DEFAULT_DECODER_NUM_HEADS="8"
DEFAULT_DECODER_NUM_LAYERS="6"
DEFAULT_PATCH_SIZE="16"
DEFAULT_MAX_FRAMES="100"

IMG_SIZE="${IMG_SIZE:-${DEFAULT_IMG_SIZE}}"
S="${S:-${DEFAULT_S}}"
N="${N:-${DEFAULT_N}}"
NUM_QUERIES="${NUM_QUERIES:-${DEFAULT_NUM_QUERIES}}"
STRIDES_TEXT="${STRIDES_TEXT:-${DEFAULT_STRIDES_TEXT}}"
CLIP_STEP="${CLIP_STEP:-${DEFAULT_CLIP_STEP}}"
ENCODER_EMBED_DIM="${ENCODER_EMBED_DIM:-${DEFAULT_ENCODER_EMBED_DIM}}"
ENCODER_DEPTH="${ENCODER_DEPTH:-${DEFAULT_ENCODER_DEPTH}}"
ENCODER_NUM_HEADS="${ENCODER_NUM_HEADS:-${DEFAULT_ENCODER_NUM_HEADS}}"
DECODER_DIM="${DECODER_DIM:-${DEFAULT_DECODER_DIM}}"
DECODER_NUM_HEADS="${DECODER_NUM_HEADS:-${DEFAULT_DECODER_NUM_HEADS}}"
DECODER_NUM_LAYERS="${DECODER_NUM_LAYERS:-${DEFAULT_DECODER_NUM_LAYERS}}"
PATCH_SIZE="${PATCH_SIZE:-${DEFAULT_PATCH_SIZE}}"
MAX_FRAMES="${MAX_FRAMES:-${DEFAULT_MAX_FRAMES}}"

# Export / replay defaults.
DEFAULT_EXPORT_VISIBLE_DEVICES="0"
DEFAULT_EXPORT_DEVICE="cuda:0"
DEFAULT_EXPORT_NUM_WORKERS="2"
DEFAULT_EVAL_MAX_CLIPS="16"
DEFAULT_FORMAL_MAX_CLIPS="64"
DEFAULT_EVAL_TEST_DSET="val"

EXPORT_VISIBLE_DEVICES="${EXPORT_VISIBLE_DEVICES:-${DEFAULT_EXPORT_VISIBLE_DEVICES}}"
EXPORT_DEVICE="${EXPORT_DEVICE:-${DEFAULT_EXPORT_DEVICE}}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS:-${DEFAULT_EXPORT_NUM_WORKERS}}"
EVAL_MAX_CLIPS="${EVAL_MAX_CLIPS:-${DEFAULT_EVAL_MAX_CLIPS}}"
FORMAL_MAX_CLIPS="${FORMAL_MAX_CLIPS:-${DEFAULT_FORMAL_MAX_CLIPS}}"
EVAL_TEST_DSET="${EVAL_TEST_DSET:-${DEFAULT_EVAL_TEST_DSET}}"
SWEEP_STEP_LIST="${SWEEP_STEP_LIST:-all}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Runtime env tuning.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

usage() {
  cat <<'EOF'
Usage:
  scripts/m3_checkpoint_pipeline.sh train-bg
  scripts/m3_checkpoint_pipeline.sh train-fg
  scripts/m3_checkpoint_pipeline.sh resume-bg [run_root] [ckpt]
  scripts/m3_checkpoint_pipeline.sh resume-fg [run_root] [ckpt]
  scripts/m3_checkpoint_pipeline.sh status [run_root]
  scripts/m3_checkpoint_pipeline.sh list-ckpts [run_root]
  scripts/m3_checkpoint_pipeline.sh eval-sweep [run_root]
  scripts/m3_checkpoint_pipeline.sh formal-one [run_root] [ckpt]
  scripts/m3_checkpoint_pipeline.sh formal-best [run_root]
  scripts/m3_checkpoint_pipeline.sh help

Defaults already match the current machine and dataset:
  DATA_ROOT=/home/wangyumu/wym-project/4d-gaussgym/datasets/PointOdyssey
  dual-GPU training
  img_size=192, S=8, N=1024, num_queries=1024

Useful overrides:
  RUN_ROOT=/path/to/existing/run
  TRAIN_MAX_STEPS=1200
  RESUME_EXTRA_STEPS=400
  EVAL_MAX_CLIPS=32
  FORMAL_MAX_CLIPS=-1
  SWEEP_STEP_LIST="100 300 500 700"
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

ensure_python() {
  [[ -x "${PYTHON_BIN}" ]] || die "Python not found: ${PYTHON_BIN}"
}

ensure_data_root() {
  [[ -d "${DATA_ROOT}/train" ]] || die "Missing dataset split: ${DATA_ROOT}/train"
  [[ -d "${DATA_ROOT}/val" ]] || die "Missing dataset split: ${DATA_ROOT}/val"
}

ensure_repo_scripts() {
  [[ -f "${SCRIPT_DIR}/train_d4rt.py" ]] || die "Missing train_d4rt.py"
  [[ -f "${SCRIPT_DIR}/export_separation_stream.py" ]] || die "Missing export_separation_stream.py"
  [[ -f "${SCRIPT_DIR}/run_separation_replay.py" ]] || die "Missing run_separation_replay.py"
  [[ -f "${SCRIPT_DIR}/build_separation_meshes.py" ]] || die "Missing build_separation_meshes.py"
}

load_strides_array() {
  read -r -a STRIDES_ARGS <<<"${STRIDES_TEXT}"
  [[ "${#STRIDES_ARGS[@]}" -gt 0 ]] || die "STRIDES_TEXT is empty."
}

default_new_run_root() {
  printf '%s/%s_%s\n' "${OUTPUTS_ROOT}" "${RUN_PREFIX}" "$(date +%Y%m%d_%H%M%S)"
}

find_latest_run_root() {
  find "${OUTPUTS_ROOT}" -maxdepth 1 -mindepth 1 -type d -name "${RUN_PREFIX}*" | sort | tail -n 1
}

resolve_run_root() {
  local candidate="${1:-${RUN_ROOT}}"
  candidate="${candidate#"${candidate%%[![:space:]]*}"}"
  candidate="${candidate%"${candidate##*[![:space:]]}"}"
  if [[ -n "${candidate}" ]]; then
    [[ -d "${candidate}" ]] || die "Run root does not exist: ${candidate}"
    printf '%s\n' "${candidate}"
    return
  fi

  candidate="$(find_latest_run_root || true)"
  [[ -n "${candidate}" ]] || die "No run root found under ${OUTPUTS_ROOT} with prefix ${RUN_PREFIX}"
  printf '%s\n' "${candidate}"
}

ckpt_dir_for_run() {
  printf '%s/train/checkpoints\n' "$1"
}

latest_ckpt_for_run() {
  local run_root="$1"
  local ckpt_dir
  ckpt_dir="$(ckpt_dir_for_run "${run_root}")"
  [[ -d "${ckpt_dir}" ]] || die "Checkpoint directory not found: ${ckpt_dir}"

  if [[ -f "${ckpt_dir}/last.ckpt" ]]; then
    printf '%s\n' "${ckpt_dir}/last.ckpt"
    return
  fi

  find "${ckpt_dir}" -maxdepth 1 -type f -name 'step-*.ckpt' | sort | tail -n 1
}

save_pipeline_config() {
  local run_root="$1"
  local cfg="${run_root}/pipeline_config.env"
  mkdir -p "${run_root}"
  cat >"${cfg}" <<EOF
PYTHON_BIN="${PYTHON_BIN}"
DATA_ROOT="${DATA_ROOT}"
RUN_PREFIX="${RUN_PREFIX}"
IMG_SIZE="${IMG_SIZE}"
S="${S}"
N="${N}"
NUM_QUERIES="${NUM_QUERIES}"
STRIDES_TEXT="${STRIDES_TEXT}"
CLIP_STEP="${CLIP_STEP}"
ENCODER_EMBED_DIM="${ENCODER_EMBED_DIM}"
ENCODER_DEPTH="${ENCODER_DEPTH}"
ENCODER_NUM_HEADS="${ENCODER_NUM_HEADS}"
DECODER_DIM="${DECODER_DIM}"
DECODER_NUM_HEADS="${DECODER_NUM_HEADS}"
DECODER_NUM_LAYERS="${DECODER_NUM_LAYERS}"
PATCH_SIZE="${PATCH_SIZE}"
MAX_FRAMES="${MAX_FRAMES}"
TRAIN_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}"
TRAIN_DEVICES="${TRAIN_DEVICES}"
TRAIN_ACCELERATOR="${TRAIN_ACCELERATOR}"
TRAIN_PRECISION="${TRAIN_PRECISION}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS}"
TRAIN_MAX_EPOCHS="${TRAIN_MAX_EPOCHS}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY}"
EXPORT_VISIBLE_DEVICES="${EXPORT_VISIBLE_DEVICES}"
EXPORT_DEVICE="${EXPORT_DEVICE}"
EXPORT_NUM_WORKERS="${EXPORT_NUM_WORKERS}"
EVAL_MAX_CLIPS="${EVAL_MAX_CLIPS}"
FORMAL_MAX_CLIPS="${FORMAL_MAX_CLIPS}"
EVAL_TEST_DSET="${EVAL_TEST_DSET}"
EOF
}

load_pipeline_config_if_present() {
  local run_root="$1"
  local cfg="${run_root}/pipeline_config.env"
  local keep_python_bin="${PYTHON_BIN}"
  local keep_data_root="${DATA_ROOT}"
  local keep_run_prefix="${RUN_PREFIX}"
  local keep_train_visible_devices="${TRAIN_VISIBLE_DEVICES}"
  local keep_train_devices="${TRAIN_DEVICES}"
  local keep_train_accelerator="${TRAIN_ACCELERATOR}"
  local keep_train_precision="${TRAIN_PRECISION}"
  local keep_train_batch_size="${TRAIN_BATCH_SIZE}"
  local keep_train_num_workers="${TRAIN_NUM_WORKERS}"
  local keep_train_max_epochs="${TRAIN_MAX_EPOCHS}"
  local keep_train_max_steps="${TRAIN_MAX_STEPS}"
  local keep_train_warmup_steps="${TRAIN_WARMUP_STEPS}"
  local keep_checkpoint_every="${CHECKPOINT_EVERY}"
  local keep_img_size="${IMG_SIZE}"
  local keep_s="${S}"
  local keep_n="${N}"
  local keep_num_queries="${NUM_QUERIES}"
  local keep_strides_text="${STRIDES_TEXT}"
  local keep_clip_step="${CLIP_STEP}"
  local keep_encoder_embed_dim="${ENCODER_EMBED_DIM}"
  local keep_encoder_depth="${ENCODER_DEPTH}"
  local keep_encoder_num_heads="${ENCODER_NUM_HEADS}"
  local keep_decoder_dim="${DECODER_DIM}"
  local keep_decoder_num_heads="${DECODER_NUM_HEADS}"
  local keep_decoder_num_layers="${DECODER_NUM_LAYERS}"
  local keep_patch_size="${PATCH_SIZE}"
  local keep_max_frames="${MAX_FRAMES}"
  local keep_export_visible_devices="${EXPORT_VISIBLE_DEVICES}"
  local keep_export_device="${EXPORT_DEVICE}"
  local keep_export_num_workers="${EXPORT_NUM_WORKERS}"
  local keep_eval_max_clips="${EVAL_MAX_CLIPS}"
  local keep_formal_max_clips="${FORMAL_MAX_CLIPS}"
  local keep_eval_test_dset="${EVAL_TEST_DSET}"
  if [[ -f "${cfg}" ]]; then
    # shellcheck disable=SC1090
    source "${cfg}"
  fi

  [[ "${keep_python_bin}" != "${DEFAULT_PYTHON_BIN}" ]] && PYTHON_BIN="${keep_python_bin}"
  [[ "${keep_data_root}" != "${DEFAULT_DATA_ROOT}" ]] && DATA_ROOT="${keep_data_root}"
  [[ "${keep_run_prefix}" != "${DEFAULT_RUN_PREFIX}" ]] && RUN_PREFIX="${keep_run_prefix}"
  [[ "${keep_train_visible_devices}" != "${DEFAULT_TRAIN_VISIBLE_DEVICES}" ]] && TRAIN_VISIBLE_DEVICES="${keep_train_visible_devices}"
  [[ "${keep_train_devices}" != "${DEFAULT_TRAIN_DEVICES}" ]] && TRAIN_DEVICES="${keep_train_devices}"
  [[ "${keep_train_accelerator}" != "${DEFAULT_TRAIN_ACCELERATOR}" ]] && TRAIN_ACCELERATOR="${keep_train_accelerator}"
  [[ "${keep_train_precision}" != "${DEFAULT_TRAIN_PRECISION}" ]] && TRAIN_PRECISION="${keep_train_precision}"
  [[ "${keep_train_batch_size}" != "${DEFAULT_TRAIN_BATCH_SIZE}" ]] && TRAIN_BATCH_SIZE="${keep_train_batch_size}"
  [[ "${keep_train_num_workers}" != "${DEFAULT_TRAIN_NUM_WORKERS}" ]] && TRAIN_NUM_WORKERS="${keep_train_num_workers}"
  [[ "${keep_train_max_epochs}" != "${DEFAULT_TRAIN_MAX_EPOCHS}" ]] && TRAIN_MAX_EPOCHS="${keep_train_max_epochs}"
  [[ "${keep_train_max_steps}" != "${DEFAULT_TRAIN_MAX_STEPS}" ]] && TRAIN_MAX_STEPS="${keep_train_max_steps}"
  [[ "${keep_train_warmup_steps}" != "${DEFAULT_TRAIN_WARMUP_STEPS}" ]] && TRAIN_WARMUP_STEPS="${keep_train_warmup_steps}"
  [[ "${keep_checkpoint_every}" != "${DEFAULT_CHECKPOINT_EVERY}" ]] && CHECKPOINT_EVERY="${keep_checkpoint_every}"
  [[ "${keep_img_size}" != "${DEFAULT_IMG_SIZE}" ]] && IMG_SIZE="${keep_img_size}"
  [[ "${keep_s}" != "${DEFAULT_S}" ]] && S="${keep_s}"
  [[ "${keep_n}" != "${DEFAULT_N}" ]] && N="${keep_n}"
  [[ "${keep_num_queries}" != "${DEFAULT_NUM_QUERIES}" ]] && NUM_QUERIES="${keep_num_queries}"
  [[ "${keep_strides_text}" != "${DEFAULT_STRIDES_TEXT}" ]] && STRIDES_TEXT="${keep_strides_text}"
  [[ "${keep_clip_step}" != "${DEFAULT_CLIP_STEP}" ]] && CLIP_STEP="${keep_clip_step}"
  [[ "${keep_encoder_embed_dim}" != "${DEFAULT_ENCODER_EMBED_DIM}" ]] && ENCODER_EMBED_DIM="${keep_encoder_embed_dim}"
  [[ "${keep_encoder_depth}" != "${DEFAULT_ENCODER_DEPTH}" ]] && ENCODER_DEPTH="${keep_encoder_depth}"
  [[ "${keep_encoder_num_heads}" != "${DEFAULT_ENCODER_NUM_HEADS}" ]] && ENCODER_NUM_HEADS="${keep_encoder_num_heads}"
  [[ "${keep_decoder_dim}" != "${DEFAULT_DECODER_DIM}" ]] && DECODER_DIM="${keep_decoder_dim}"
  [[ "${keep_decoder_num_heads}" != "${DEFAULT_DECODER_NUM_HEADS}" ]] && DECODER_NUM_HEADS="${keep_decoder_num_heads}"
  [[ "${keep_decoder_num_layers}" != "${DEFAULT_DECODER_NUM_LAYERS}" ]] && DECODER_NUM_LAYERS="${keep_decoder_num_layers}"
  [[ "${keep_patch_size}" != "${DEFAULT_PATCH_SIZE}" ]] && PATCH_SIZE="${keep_patch_size}"
  [[ "${keep_max_frames}" != "${DEFAULT_MAX_FRAMES}" ]] && MAX_FRAMES="${keep_max_frames}"
  [[ "${keep_export_visible_devices}" != "${DEFAULT_EXPORT_VISIBLE_DEVICES}" ]] && EXPORT_VISIBLE_DEVICES="${keep_export_visible_devices}"
  [[ "${keep_export_device}" != "${DEFAULT_EXPORT_DEVICE}" ]] && EXPORT_DEVICE="${keep_export_device}"
  [[ "${keep_export_num_workers}" != "${DEFAULT_EXPORT_NUM_WORKERS}" ]] && EXPORT_NUM_WORKERS="${keep_export_num_workers}"
  [[ "${keep_eval_max_clips}" != "${DEFAULT_EVAL_MAX_CLIPS}" ]] && EVAL_MAX_CLIPS="${keep_eval_max_clips}"
  [[ "${keep_formal_max_clips}" != "${DEFAULT_FORMAL_MAX_CLIPS}" ]] && FORMAL_MAX_CLIPS="${keep_formal_max_clips}"
  [[ "${keep_eval_test_dset}" != "${DEFAULT_EVAL_TEST_DSET}" ]] && EVAL_TEST_DSET="${keep_eval_test_dset}"
  return 0
}

write_command_file() {
  local output_path="$1"
  shift
  printf '#!/usr/bin/env bash\nset -euo pipefail\n' >"${output_path}"
  printf '%q ' "$@" >>"${output_path}"
  printf '\n' >>"${output_path}"
  chmod +x "${output_path}"
}

build_train_args() {
  load_strides_array
  TRAIN_ARGS=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/train_d4rt.py"
    --dataset_location "${DATA_ROOT}"
    --train_dset train
    --S "${S}"
    --N "${N}"
    --num_queries "${NUM_QUERIES}"
    --img_size "${IMG_SIZE}"
    --strides "${STRIDES_ARGS[@]}"
    --clip_step "${CLIP_STEP}"
    --encoder_embed_dim "${ENCODER_EMBED_DIM}"
    --encoder_depth "${ENCODER_DEPTH}"
    --encoder_num_heads "${ENCODER_NUM_HEADS}"
    --decoder_dim "${DECODER_DIM}"
    --decoder_num_heads "${DECODER_NUM_HEADS}"
    --decoder_num_layers "${DECODER_NUM_LAYERS}"
    --patch_size "${PATCH_SIZE}"
    --max_frames "${MAX_FRAMES}"
    --batch_size "${TRAIN_BATCH_SIZE}"
    --num_workers "${TRAIN_NUM_WORKERS}"
    --max_epochs "${TRAIN_MAX_EPOCHS}"
    --max_steps "${TRAIN_MAX_STEPS}"
    --warmup_steps "${TRAIN_WARMUP_STEPS}"
    --checkpoint_every_n_train_steps "${CHECKPOINT_EVERY}"
    --devices "${TRAIN_DEVICES}"
    --accelerator "${TRAIN_ACCELERATOR}"
    --precision "${TRAIN_PRECISION}"
  )
}

launch_train() {
  local mode="$1"
  local run_root="$2"
  local log_dir="${run_root}/train"
  local log_path="${run_root}/train.log"
  local pid_path="${run_root}/train.pid"

  mkdir -p "${log_dir}"
  save_pipeline_config "${run_root}"
  build_train_args

  TRAIN_ARGS+=(--log_dir "${log_dir}")

  write_command_file "${run_root}/launch_train_command.sh" env \
    CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}" \
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    "${TRAIN_ARGS[@]}"

  if [[ "${mode}" == "bg" ]]; then
    nohup env \
      CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}" \
      NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
      MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
      OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
      NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
      "${TRAIN_ARGS[@]}" >"${log_path}" 2>&1 &
    echo $! >"${pid_path}"
    ln -sfn "${run_root}" "${OUTPUTS_ROOT}/${RUN_PREFIX}_latest"
    echo "Background training started"
    echo "- run_root: ${run_root}"
    echo "- pid: $(cat "${pid_path}")"
    echo "- log: ${log_path}"
    return
  fi

  env \
    CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}" \
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    "${TRAIN_ARGS[@]}"
}

extract_ckpt_step() {
  local ckpt_path="$1"
  "${PYTHON_BIN}" - <<'PY' "${ckpt_path}"
import os
import sys
import torch

ckpt_path = sys.argv[1]
name = os.path.basename(ckpt_path)
if name.startswith("step-") and name.endswith(".ckpt"):
    try:
        print(int(name[len("step-"):-len(".ckpt")]))
        raise SystemExit(0)
    except ValueError:
        pass

payload = torch.load(ckpt_path, map_location="cpu")
print(int(payload.get("global_step", 0)))
PY
}

launch_resume() {
  local mode="$1"
  local run_root="$2"
  local resume_ckpt="$3"
  local log_dir="${run_root}/train"
  local log_path="${run_root}/train_resume.log"
  local pid_path="${run_root}/train.pid"
  local current_step
  local resume_max_steps

  [[ -f "${resume_ckpt}" ]] || die "Resume checkpoint not found: ${resume_ckpt}"
  load_pipeline_config_if_present "${run_root}"
  current_step="$(extract_ckpt_step "${resume_ckpt}")"
  resume_max_steps="$((current_step + RESUME_EXTRA_STEPS))"
  [[ "${resume_max_steps}" -gt "${TRAIN_WARMUP_STEPS}" ]] || die "resume max_steps must stay > warmup_steps"

  mkdir -p "${log_dir}"
  save_pipeline_config "${run_root}"
  build_train_args
  TRAIN_ARGS+=(--max_steps "${resume_max_steps}" --log_dir "${log_dir}" --resume_from_checkpoint "${resume_ckpt}")

  write_command_file "${run_root}/resume_train_command.sh" env \
    CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}" \
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    "${TRAIN_ARGS[@]}"

  if [[ "${mode}" == "bg" ]]; then
    nohup env \
      CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}" \
      NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
      MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
      OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
      NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
      "${TRAIN_ARGS[@]}" >"${log_path}" 2>&1 &
    echo $! >"${pid_path}"

    echo "Background resume started"
    echo "- run_root: ${run_root}"
    echo "- resume_ckpt: ${resume_ckpt}"
    echo "- current_step: ${current_step}"
    echo "- target_max_steps: ${resume_max_steps}"
    echo "- pid: $(cat "${pid_path}")"
    echo "- log: ${log_path}"
    return
  fi

  echo "Foreground resume starting"
  echo "- run_root: ${run_root}"
  echo "- resume_ckpt: ${resume_ckpt}"
  echo "- current_step: ${current_step}"
  echo "- target_max_steps: ${resume_max_steps}"

  env \
    CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_DEVICES}" \
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    "${TRAIN_ARGS[@]}"
}

warn_if_training_alive() {
  local run_root="$1"
  local pid_path="${run_root}/train.pid"
  if [[ -f "${pid_path}" ]]; then
    local pid
    pid="$(cat "${pid_path}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
      echo "Warning: training PID ${pid} is still running. Export may contend for GPU." >&2
    fi
  fi
  return 0
}

select_ckpts_for_sweep() {
  local run_root="$1"
  local ckpt_dir
  ckpt_dir="$(ckpt_dir_for_run "${run_root}")"
  [[ -d "${ckpt_dir}" ]] || die "Checkpoint directory not found: ${ckpt_dir}"

  CKPT_PATHS=()
  if [[ "${SWEEP_STEP_LIST}" == "all" ]]; then
    while IFS= read -r ckpt; do
      [[ -n "${ckpt}" ]] && CKPT_PATHS+=("${ckpt}")
    done < <(find "${ckpt_dir}" -maxdepth 1 -type f -name 'step-*.ckpt' | sort)
  else
    local step
    for step in ${SWEEP_STEP_LIST}; do
      local ckpt
      ckpt="${ckpt_dir}/$(printf 'step-%08d.ckpt' "${step}")"
      [[ -f "${ckpt}" ]] || die "Checkpoint not found: ${ckpt}"
      CKPT_PATHS+=("${ckpt}")
    done
  fi

  [[ "${#CKPT_PATHS[@]}" -gt 0 ]] || die "No checkpoints selected for sweep."
}

run_export_for_ckpt() {
  local ckpt_path="$1"
  local output_npz="$2"
  local max_clips="$3"
  load_strides_array
  mkdir -p "$(dirname "${output_npz}")"

  env CUDA_VISIBLE_DEVICES="${EXPORT_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/export_separation_stream.py" \
    --test_data_path "${DATA_ROOT}" \
    --test_dset "${EVAL_TEST_DSET}" \
    --ckpt "${ckpt_path}" \
    --output_npz "${output_npz}" \
    --img_size "${IMG_SIZE}" \
    --S "${S}" \
    --N "${N}" \
    --num_queries "${NUM_QUERIES}" \
    --strides "${STRIDES_ARGS[@]}" \
    --clip_step "${CLIP_STEP}" \
    --batch_size 1 \
    --num_workers "${EXPORT_NUM_WORKERS}" \
    --device "${EXPORT_DEVICE}" \
    --max_clips "${max_clips}"
}

run_eval_sweep() {
  local run_root="$1"
  local sweep_root="${run_root}/eval_sweep"
  load_pipeline_config_if_present "${run_root}"
  warn_if_training_alive "${run_root}"
  select_ckpts_for_sweep "${run_root}"
  mkdir -p "${sweep_root}"

  local ckpt
  for ckpt in "${CKPT_PATHS[@]}"; do
    local ckpt_name
    local ckpt_root
    local summary_path
    ckpt_name="$(basename "${ckpt}" .ckpt)"
    ckpt_root="${sweep_root}/${ckpt_name}"
    summary_path="${ckpt_root}/replay_dry/summary.json"

    if [[ "${SKIP_EXISTING}" == "1" && -f "${summary_path}" ]]; then
      echo "Skip existing summary: ${summary_path}"
      continue
    fi

    echo "Evaluating ${ckpt_name}"
    run_export_for_ckpt "${ckpt}" "${ckpt_root}/stream.npz" "${EVAL_MAX_CLIPS}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/run_separation_replay.py" \
      --input_npz "${ckpt_root}/stream.npz" \
      --output_dir "${ckpt_root}/replay_dry" \
      --dry_run \
      --save_json
  done

  "${PYTHON_BIN}" - <<'PY' "${sweep_root}"
import glob
import json
import os
import sys

sweep_root = sys.argv[1]
run_root = os.path.dirname(sweep_root)
summary_paths = sorted(glob.glob(os.path.join(sweep_root, "step-*", "replay_dry", "summary.json")))
rows = []
for path in summary_paths:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    ckpt_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    rows.append(
        {
            "ckpt_name": ckpt_name,
            "ckpt_path": os.path.join(run_root, "train", "checkpoints", ckpt_name + ".ckpt"),
            "processed_frames": int(payload["processed_frames"]),
            "total_static_points": int(payload["total_static_points"]),
            "total_dynamic_points": int(payload["total_dynamic_points"]),
            "total_uncertain_points": int(payload["total_uncertain_points"]),
            "mean_active_tracks": float(payload["mean_active_tracks"]),
            "summary_path": path,
        }
    )

rows.sort(
    key=lambda row: (
        -row["mean_active_tracks"],
        row["total_uncertain_points"],
        -row["total_dynamic_points"],
        -row["total_static_points"],
        row["ckpt_name"],
    )
)

scoreboard_tsv = os.path.join(sweep_root, "scoreboard.tsv")
with open(scoreboard_tsv, "w", encoding="utf-8") as fp:
    fp.write(
        "rank\tckpt_name\tframes\tstatic\tdynamic\tuncertain\tmean_active_tracks\tsummary_path\n"
    )
    for idx, row in enumerate(rows, start=1):
        fp.write(
            f"{idx}\t{row['ckpt_name']}\t{row['processed_frames']}\t"
            f"{row['total_static_points']}\t{row['total_dynamic_points']}\t"
            f"{row['total_uncertain_points']}\t{row['mean_active_tracks']:.6f}\t"
            f"{row['summary_path']}\n"
        )

best_ckpt_file = os.path.join(sweep_root, "best_ckpt.txt")
if rows:
    with open(best_ckpt_file, "w", encoding="utf-8") as fp:
        fp.write(rows[0]["ckpt_name"] + "\n")

print(f"scoreboard: {scoreboard_tsv}")
if rows:
    print(f"best_ckpt: {rows[0]['ckpt_name']}")
else:
    print("best_ckpt: none")
PY
}

resolve_ckpt_path() {
  local run_root="$1"
  local ckpt_arg="${2:-}"
  local ckpt_dir
  ckpt_dir="$(ckpt_dir_for_run "${run_root}")"
  ckpt_arg="${ckpt_arg#"${ckpt_arg%%[![:space:]]*}"}"
  ckpt_arg="${ckpt_arg%"${ckpt_arg##*[![:space:]]}"}"

  if [[ -z "${ckpt_arg}" ]]; then
    latest_ckpt_for_run "${run_root}"
    return
  fi

  if [[ -f "${ckpt_arg}" ]]; then
    printf '%s\n' "${ckpt_arg}"
    return
  fi

  if [[ -f "${ckpt_dir}/${ckpt_arg}" ]]; then
    printf '%s\n' "${ckpt_dir}/${ckpt_arg}"
    return
  fi

  if [[ -f "${ckpt_dir}/${ckpt_arg}.ckpt" ]]; then
    printf '%s\n' "${ckpt_dir}/${ckpt_arg}.ckpt"
    return
  fi

  die "Could not resolve checkpoint: ${ckpt_arg}"
}

best_ckpt_from_sweep() {
  local run_root="$1"
  local best_file="${run_root}/eval_sweep/best_ckpt.txt"
  [[ -f "${best_file}" ]] || die "Best checkpoint file not found: ${best_file}. Run eval-sweep first."
  local ckpt_name
  ckpt_name="$(head -n 1 "${best_file}")"
  resolve_ckpt_path "${run_root}" "${ckpt_name}"
}

run_formal_one() {
  local run_root="$1"
  local ckpt_path="$2"
  local ckpt_name
  local formal_label
  local formal_root

  load_pipeline_config_if_present "${run_root}"
  warn_if_training_alive "${run_root}"
  ckpt_name="$(basename "${ckpt_path}" .ckpt)"
  if [[ "${FORMAL_MAX_CLIPS}" == "-1" ]]; then
    formal_label="all"
  else
    formal_label="clips${FORMAL_MAX_CLIPS}"
  fi
  formal_root="${run_root}/formal/${ckpt_name}_${formal_label}"
  mkdir -p "${formal_root}"

  echo "Formal export for ${ckpt_name}"
  run_export_for_ckpt "${ckpt_path}" "${formal_root}/stream_full.npz" "${FORMAL_MAX_CLIPS}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/run_separation_replay.py" \
    --input_npz "${formal_root}/stream_full.npz" \
    --output_dir "${formal_root}/replay_full"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_separation_meshes.py" \
    --frames_dir "${formal_root}/replay_full/frames" \
    --output_dir "${formal_root}/mesh_smoke"

  echo "Formal pipeline finished"
  echo "- ckpt: ${ckpt_path}"
  echo "- output: ${formal_root}"
}

show_status() {
  local run_root="$1"
  local pid_path="${run_root}/train.pid"
  local log_path="${run_root}/train.log"
  local ckpt_dir
  ckpt_dir="$(ckpt_dir_for_run "${run_root}")"

  echo "run_root: ${run_root}"
  echo "ckpt_dir: ${ckpt_dir}"
  if [[ -f "${pid_path}" ]]; then
    local pid
    pid="$(cat "${pid_path}")"
    if ps -p "${pid}" >/dev/null 2>&1; then
      echo "train_pid: ${pid} (running)"
      ps -p "${pid}" -o pid=,etime=,cmd=
    else
      echo "train_pid: ${pid} (not running)"
    fi
  else
    echo "train_pid: none"
  fi

  if [[ -d "${ckpt_dir}" ]]; then
    find "${ckpt_dir}" -maxdepth 1 -type f -name '*.ckpt' | sort | tail -n 10
  fi

  if [[ -f "${log_path}" ]]; then
    echo "----- tail ${log_path} -----"
    tail -n 20 "${log_path}"
  fi
}

list_ckpts() {
  local run_root="$1"
  local ckpt_dir
  ckpt_dir="$(ckpt_dir_for_run "${run_root}")"
  [[ -d "${ckpt_dir}" ]] || die "Checkpoint directory not found: ${ckpt_dir}"
  find "${ckpt_dir}" -maxdepth 1 -type f -name '*.ckpt' | sort
}

main() {
  local cmd="${1:-help}"
  shift || true

  need_cmd find
  need_cmd sort
  need_cmd tail
  need_cmd ps
  ensure_python
  ensure_data_root
  ensure_repo_scripts
  mkdir -p "${OUTPUTS_ROOT}"

  case "${cmd}" in
    train-bg)
      local run_root
      run_root="${RUN_ROOT:-$(default_new_run_root)}"
      [[ ! -e "${run_root}" ]] || die "Run root already exists: ${run_root}"
      launch_train bg "${run_root}"
      ;;
    train-fg)
      local run_root
      run_root="${RUN_ROOT:-$(default_new_run_root)}"
      [[ ! -e "${run_root}" ]] || die "Run root already exists: ${run_root}"
      launch_train fg "${run_root}"
      ;;
    resume-bg)
      local run_root
      local resume_ckpt
      run_root="$(resolve_run_root "${1:-}")"
      if [[ $# -ge 2 ]]; then
        resume_ckpt="$(resolve_ckpt_path "${run_root}" "${2}")"
      else
        resume_ckpt="$(latest_ckpt_for_run "${run_root}")"
      fi
      launch_resume bg "${run_root}" "${resume_ckpt}"
      ;;
    resume-fg)
      local run_root
      local resume_ckpt
      run_root="$(resolve_run_root "${1:-}")"
      if [[ $# -ge 2 ]]; then
        resume_ckpt="$(resolve_ckpt_path "${run_root}" "${2}")"
      else
        resume_ckpt="$(latest_ckpt_for_run "${run_root}")"
      fi
      launch_resume fg "${run_root}" "${resume_ckpt}"
      ;;
    status)
      local run_root
      run_root="$(resolve_run_root "${1:-}")"
      show_status "${run_root}"
      ;;
    list-ckpts)
      local run_root
      run_root="$(resolve_run_root "${1:-}")"
      list_ckpts "${run_root}"
      ;;
    eval-sweep)
      local run_root
      run_root="$(resolve_run_root "${1:-}")"
      run_eval_sweep "${run_root}"
      ;;
    formal-one)
      local run_root
      local ckpt_path
      run_root="$(resolve_run_root "${1:-}")"
      ckpt_path="$(resolve_ckpt_path "${run_root}" "${2:-}")"
      run_formal_one "${run_root}" "${ckpt_path}"
      ;;
    formal-best)
      local run_root
      local ckpt_path
      run_root="$(resolve_run_root "${1:-}")"
      ckpt_path="$(best_ckpt_from_sweep "${run_root}")"
      run_formal_one "${run_root}" "${ckpt_path}"
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      usage
      die "Unknown command: ${cmd}"
      ;;
  esac
}

main "$@"
