#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash research/tools/pull_reference_projects.sh [target_dir]

Pull or update external reference projects used by the static-mesh / static-dynamic comparison guide.
Default target_dir: $HOME/reference-projects/mesh_static_dynamic
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

BASE_DIR="${1:-$HOME/reference-projects/mesh_static_dynamic}"
mkdir -p "$BASE_DIR"

clone_or_update() {
  local name="$1"
  local url="$2"
  local dir="$BASE_DIR/$name"

  if [[ -d "$dir/.git" ]]; then
    printf '[update] %s\n' "$name"
    env GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 git -C "$dir" pull --ff-only
    return 0
  fi

  if [[ -e "$dir" ]]; then
    printf '[error] %s exists but is not a git repo: %s\n' "$name" "$dir" >&2
    return 1
  fi

  printf '[clone] %s\n' "$name"
  env GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none "$url" "$dir"
}

PROJECTS=(
  'go-surf|https://github.com/JingwenWang95/go-surf.git'
  'BundleFusion|https://github.com/niessner/BundleFusion.git'
  'voxblox|https://github.com/ethz-asl/voxblox.git'
  'ppsurf|https://github.com/cg-tuwien/ppsurf.git'
  'BundleSDF|https://github.com/NVlabs/BundleSDF.git'
  'dynsurf|https://github.com/Mirgahney/dynsurf.git'
  'DynaSurfGS|https://github.com/Open3DVLab/DynaSurfGS.git'
  '4dtam|https://github.com/muskie82/4dtam.git'
  'GauSTAR|https://github.com/eth-ait/GauSTAR.git'
)

for item in "${PROJECTS[@]}"; do
  IFS='|' read -r name url <<< "$item"
  clone_or_update "$name" "$url"
done

printf '\nDone. Repos are in: %s\n' "$BASE_DIR"
