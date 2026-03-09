"""Build a dense RGBD world point cloud directly from PointOdyssey sequence files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproject PointOdyssey RGBD frames into world coordinates and export "
            "a colored sequence-level PLY."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="PointOdyssey root path that contains train/val directories.",
    )
    parser.add_argument("--dset", type=str, default="val", help="Split name, e.g. train/val.")
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence name under <dataset_root>/<dset>. Optional when --sequence_index is used.",
    )
    parser.add_argument(
        "--sequence_index",
        type=int,
        default=0,
        help="Sequence index used when --sequence is omitted.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        help="First frame index (inclusive).",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=-1,
        help="Last frame index (exclusive). -1 means no upper bound.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Frame subsampling stride.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=64,
        help="Process at most this many frames. <=0 means no limit.",
    )
    parser.add_argument(
        "--pixel_stride",
        type=int,
        default=1,
        help="Pixel subsampling stride for each frame. 1 means dense.",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=(1000.0 / 65535.0),
        help=(
            "Depth conversion scale from uint16 to metric depth. "
            "Default follows D4RT dataset loader: depth_float = depth16 * 1000 / 65535."
        ),
    )
    parser.add_argument(
        "--depth_min",
        type=float,
        default=1e-6,
        help="Drop depth <= depth_min.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=-1.0,
        help="Drop depth > depth_max when >0.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.0,
        help="Voxel downsampling size in world space. <=0 disables downsampling.",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=1_500_000,
        help="Cap final output points. <=0 keeps all points.",
    )
    parser.add_argument(
        "--output_ply",
        type=str,
        required=True,
        help="Output colored PLY path.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "open3d", "none"],
        default="auto",
        help="Preview backend. Use 'none' for export only.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open viewer window (useful for headless run).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling.")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be > 0.")
    if args.pixel_stride <= 0:
        raise ValueError("--pixel_stride must be > 0.")
    if args.start_frame < 0:
        raise ValueError("--start_frame must be >= 0.")
    if args.end_frame >= 0 and args.end_frame <= args.start_frame:
        raise ValueError("--end_frame must be > --start_frame, or set to -1.")
    if args.depth_scale <= 0:
        raise ValueError("--depth_scale must be > 0.")
    if args.depth_max > 0 and args.depth_max <= args.depth_min:
        raise ValueError("--depth_max must be > --depth_min when depth_max > 0.")


def _list_sequences(split_dir: Path) -> List[Path]:
    return sorted([path for path in split_dir.iterdir() if path.is_dir()])


def _resolve_sequence_dir(args: argparse.Namespace) -> Path:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    split_dir = dataset_root / args.dset
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if args.sequence is not None:
        seq_path = Path(args.sequence).expanduser()
        if seq_path.is_dir():
            return seq_path.resolve()
        candidate = split_dir / args.sequence
        if candidate.is_dir():
            return candidate.resolve()
        raise FileNotFoundError(
            f"Sequence not found. Tried absolute/relative path and split entry: {args.sequence}"
        )

    sequences = _list_sequences(split_dir)
    if not sequences:
        raise FileNotFoundError(f"No sequence directories found under: {split_dir}")
    if args.sequence_index < 0 or args.sequence_index >= len(sequences):
        raise IndexError(
            f"--sequence_index ({args.sequence_index}) out of range. "
            f"Available range: 0..{len(sequences) - 1}."
        )
    return sequences[args.sequence_index].resolve()


def _pick_npz_key(payload: np.lib.npyio.NpzFile, candidates: Iterable[str]) -> np.ndarray:
    for key in candidates:
        if key in payload:
            return np.asarray(payload[key])
    raise KeyError(
        f"None of the keys {list(candidates)} were found in anno.npz. "
        f"Available keys: {list(payload.files)}"
    )


def _load_intrinsics_extrinsics(anno_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(str(anno_path), allow_pickle=True) as payload:
        intrinsics = _pick_npz_key(payload, ["intrinsics", "pix_T_cams", "K"])
        extrinsics = _pick_npz_key(
            payload,
            ["extrinsics", "cams_T_world", "cam_T_world", "world_to_cam"],
        )

    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)

    if intrinsics.ndim == 2 and intrinsics.shape == (3, 3):
        intrinsics = intrinsics[None, ...]
    if extrinsics.ndim == 2 and extrinsics.shape == (4, 4):
        extrinsics = extrinsics[None, ...]

    if intrinsics.ndim != 3 or intrinsics.shape[1:] != (3, 3):
        raise ValueError(f"Invalid intrinsics shape: {intrinsics.shape}")
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"Invalid extrinsics shape: {extrinsics.shape}")

    return intrinsics, extrinsics


def _get_matrix_for_frame(matrices: np.ndarray, frame_idx: int) -> np.ndarray:
    if matrices.shape[0] == 1:
        return matrices[0]
    return matrices[frame_idx]


def _collect_indexed_files(directory: Path, prefix: str, suffix: str) -> Dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")
    indexed: Dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        indexed[int(match.group(1))] = path

    if not indexed:
        raise FileNotFoundError(
            f"No files matched pattern {prefix}_*{suffix} under: {directory}"
        )
    return indexed


def _select_frame_indices(
    candidate_indices: List[int],
    start_frame: int,
    end_frame: int,
    frame_stride: int,
    max_frames: int,
) -> List[int]:
    selected = [idx for idx in candidate_indices if idx >= start_frame and (end_frame < 0 or idx < end_frame)]
    selected = selected[::frame_stride]
    if max_frames > 0:
        selected = selected[:max_frames]
    return selected


def _invert_4x4(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def _reproject_frame_to_world(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    world_to_cam: np.ndarray,
    pixel_stride: int,
    depth_min: float,
    depth_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if fx == 0 or fy == 0:
        raise ValueError(f"Invalid intrinsics with zero focal length: fx={fx}, fy={fy}")

    u_coords = np.arange(width, dtype=np.float32) * float(pixel_stride)
    v_coords = np.arange(height, dtype=np.float32) * float(pixel_stride)
    uu, vv = np.meshgrid(u_coords, v_coords, indexing="xy")

    valid = np.isfinite(depth) & (depth > depth_min)
    if depth_max > 0:
        valid &= depth <= depth_max

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    z = depth[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1).astype(np.float32, copy=False)

    world_t_cam = _invert_4x4(world_to_cam)
    rot = world_t_cam[:3, :3]
    trans = world_t_cam[:3, 3]
    points_world = (points_cam @ rot.T) + trans[None, :]

    colors = rgb[valid].astype(np.uint8, copy=False)
    return points_world.astype(np.float32, copy=False), colors


def _voxel_downsample_with_color(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0 or points.shape[0] <= 1:
        return points, colors

    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)

    point_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    color_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(point_sums, inverse, points.astype(np.float64, copy=False))
    np.add.at(color_sums, inverse, colors.astype(np.float64, copy=False))

    down_points = (point_sums / counts[:, None]).astype(np.float32)
    down_colors = np.clip(np.round(color_sums / counts[:, None]), 0, 255).astype(np.uint8)
    return down_points, down_colors


def _random_subsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[keep], colors[keep]


def _write_colored_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"Point/color length mismatch: {points.shape[0]} vs {colors.shape[0]}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    with open(path, "wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def _preview_open3d(points: np.ndarray, colors: np.ndarray, title: str) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is not installed. Install it with: pip install open3d"
        ) from exc

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    cloud.colors = o3d.utility.Vector3dVector(
        colors.astype(np.float64, copy=False) / 255.0
    )
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries(
        [cloud, frame],
        window_name=title,
        width=1600,
        height=900,
    )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    rng = np.random.default_rng(args.seed)

    sequence_dir = _resolve_sequence_dir(args)
    anno_path = sequence_dir / "anno.npz"
    if not anno_path.is_file():
        raise FileNotFoundError(f"anno.npz not found: {anno_path}")

    intrinsics, extrinsics = _load_intrinsics_extrinsics(anno_path)
    extrinsic_count = extrinsics.shape[0]
    intrinsic_count = intrinsics.shape[0]
    if intrinsic_count not in (1, extrinsic_count):
        print(
            "Warning: intrinsics/extrinsics frame counts differ "
            f"(intrinsics={intrinsic_count}, extrinsics={extrinsic_count}). "
            "The script will use the common valid frame range."
        )

    if intrinsic_count == 1:
        frame_capacity = extrinsic_count
    else:
        frame_capacity = min(intrinsic_count, extrinsic_count)

    rgb_files = _collect_indexed_files(sequence_dir / "rgbs", "rgb", ".jpg")
    depth_files = _collect_indexed_files(sequence_dir / "depths", "depth", ".png")

    candidate_indices = sorted(set(rgb_files.keys()).intersection(depth_files.keys()))
    candidate_indices = [idx for idx in candidate_indices if idx < frame_capacity]
    if not candidate_indices:
        raise RuntimeError("No overlapping RGB/depth frames with valid intrinsics/extrinsics.")

    selected_indices = _select_frame_indices(
        candidate_indices=candidate_indices,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )
    if not selected_indices:
        raise RuntimeError("No frames selected. Check frame range/stride/max_frames.")

    print(f"Sequence: {sequence_dir.name}")
    print(
        f"Selected frames: {len(selected_indices)} "
        f"(min={selected_indices[0]}, max={selected_indices[-1]})"
    )
    print(f"Depth scale: {args.depth_scale}")
    print(f"Voxel size: {args.voxel_size}")

    point_chunks: List[np.ndarray] = []
    color_chunks: List[np.ndarray] = []
    buffered_points = 0

    for i, frame_idx in enumerate(selected_indices, start=1):
        rgb_bgr = cv2.imread(str(rgb_files[frame_idx]), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to load RGB frame: {rgb_files[frame_idx]}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(str(depth_files[frame_idx]), cv2.IMREAD_ANYDEPTH)
        if depth_raw is None:
            raise RuntimeError(f"Failed to load depth frame: {depth_files[frame_idx]}")
        depth = depth_raw.astype(np.float32) * float(args.depth_scale)

        if rgb.shape[:2] != depth.shape[:2]:
            rgb = cv2.resize(
                rgb,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        if args.pixel_stride > 1:
            rgb = rgb[:: args.pixel_stride, :: args.pixel_stride]
            depth = depth[:: args.pixel_stride, :: args.pixel_stride]

        frame_points, frame_colors = _reproject_frame_to_world(
            depth=depth,
            rgb=rgb,
            intrinsics=_get_matrix_for_frame(intrinsics, frame_idx),
            world_to_cam=_get_matrix_for_frame(extrinsics, frame_idx),
            pixel_stride=args.pixel_stride,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
        )

        if args.voxel_size > 0 and frame_points.shape[0] > 0:
            frame_points, frame_colors = _voxel_downsample_with_color(
                frame_points, frame_colors, args.voxel_size
            )

        if frame_points.shape[0] > 0:
            point_chunks.append(frame_points)
            color_chunks.append(frame_colors)
            buffered_points += int(frame_points.shape[0])

        if args.max_points > 0 and buffered_points > args.max_points * 4:
            merged_points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
            merged_colors = np.concatenate(color_chunks, axis=0).astype(np.uint8, copy=False)
            target = args.max_points * 2
            merged_points, merged_colors = _random_subsample_points(
                merged_points, merged_colors, target, rng
            )
            point_chunks = [merged_points]
            color_chunks = [merged_colors]
            buffered_points = int(merged_points.shape[0])

        if i % 10 == 0 or i == len(selected_indices):
            print(
                f"Processed {i}/{len(selected_indices)} frames, "
                f"buffered points: {buffered_points}"
            )

    if point_chunks:
        points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
        colors = np.concatenate(color_chunks, axis=0).astype(np.uint8, copy=False)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)

    if args.voxel_size > 0 and points.shape[0] > 0:
        points, colors = _voxel_downsample_with_color(points, colors, args.voxel_size)
        print(f"After global voxel downsample: {points.shape[0]} points")

    points, colors = _random_subsample_points(points, colors, args.max_points, rng)
    if points.shape[0] == 0:
        raise RuntimeError("No valid points were reconstructed from selected frames.")

    output_ply = Path(args.output_ply).expanduser().resolve()
    _write_colored_ply(output_ply, points, colors)
    print(f"Exported PLY: {output_ply}")
    print(f"Output points: {points.shape[0]}")

    if args.no_show or args.backend == "none":
        return

    if args.backend == "auto":
        try:
            _preview_open3d(points, colors, title=f"RGBD Scene: {sequence_dir.name}")
        except ImportError:
            print("Preview skipped: open3d not installed (backend=auto).")
        return

    _preview_open3d(points, colors, title=f"RGBD Scene: {sequence_dir.name}")


if __name__ == "__main__":
    main()
