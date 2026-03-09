"""Interactive RGBD timeline viewer for dense per-frame world point clouds."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FrameCloud:
    source_frame_index: int
    timestamp: float
    points: np.ndarray
    colors: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproject PointOdyssey RGBD frames into world coordinates and inspect "
            "dense per-frame dynamics with a timeline slider."
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
        "--dynamic_mode",
        type=str,
        choices=["frame", "window"],
        default="window",
        help="Dynamic layer mode: current frame only or trailing-window accumulation.",
    )
    parser.add_argument(
        "--dynamic_window",
        type=int,
        default=8,
        help="Trailing frame count used when --dynamic_mode=window.",
    )
    parser.add_argument(
        "--initial_index",
        type=int,
        default=-1,
        help="Initial timeline offset in selected frames. -1 means the last frame.",
    )
    parser.add_argument(
        "--max_static_points",
        type=int,
        default=200000,
        help="Subsample static global cloud to at most this many points for visualization.",
    )
    parser.add_argument(
        "--max_dynamic_points",
        type=int,
        default=150000,
        help="Subsample dynamic cloud to at most this many points for visualization.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.0,
        help="Scatter point size in matplotlib.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["matplotlib", "none"],
        default="matplotlib",
        help="Use 'none' for export/stat-only mode without interactive window.",
    )
    parser.add_argument(
        "--export_frames_dir",
        type=str,
        default=None,
        help="Optional output dir for per-frame colored PLY files (frame_XXXXX.ply).",
    )
    parser.add_argument(
        "--export_summary_json",
        type=str,
        default=None,
        help="Optional output JSON path for selected range and timeline summary.",
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
    if args.dynamic_mode == "window" and args.dynamic_window <= 0:
        raise ValueError("--dynamic_window must be > 0 when --dynamic_mode=window.")


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


def _concat_frame_clouds(
    frames: Sequence[FrameCloud],
    offsets: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    point_sets = [frames[offset].points for offset in offsets if frames[offset].points.shape[0] > 0]
    color_sets = [frames[offset].colors for offset in offsets if frames[offset].colors.shape[0] > 0]
    if not point_sets:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    points = np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(color_sets, axis=0).astype(np.uint8, copy=False)
    return points, colors


def _dynamic_frame_offsets(
    total_frames: int,
    offset: int,
    dynamic_mode: str,
    dynamic_window: int,
) -> List[int]:
    if offset < 0 or offset >= total_frames:
        raise IndexError(
            f"offset out of range: {offset}, valid range: 0..{total_frames - 1}"
        )
    if dynamic_mode == "frame":
        return [offset]
    if dynamic_window <= 0:
        raise ValueError("--dynamic_window must be > 0 when --dynamic_mode=window.")
    start = max(0, offset - dynamic_window + 1)
    return list(range(start, offset + 1))


def _resolve_initial_offset(frame_count: int, initial_index: int) -> int:
    if frame_count <= 0:
        raise ValueError("frame_count must be > 0.")
    if initial_index < 0:
        return frame_count - 1
    if initial_index >= frame_count:
        raise IndexError(
            f"--initial_index ({initial_index}) out of range. "
            f"Available range: 0..{frame_count - 1}."
        )
    return initial_index


def _compute_axis_limits(points: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    if points.shape[0] == 0:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = float(np.max(maxs - mins)) * 0.55
    radius = max(radius, 1e-3)
    return (
        (float(center[0] - radius), float(center[0] + radius)),
        (float(center[1] - radius), float(center[1] + radius)),
        (float(center[2] - radius), float(center[2] + radius)),
    )


def _set_equal_box_aspect(ax) -> None:
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _format_title(
    frame_index: int,
    timestamp: float,
    dynamic_mode: str,
    dynamic_window: int,
    static_count: int,
    dynamic_count: int,
) -> str:
    dynamic_tag = dynamic_mode
    if dynamic_mode == "window":
        dynamic_tag = f"window({dynamic_window})"
    return (
        f"frame={frame_index} | timestamp={timestamp:.3f} | "
        f"static={static_count} | dynamic={dynamic_count} ({dynamic_tag})"
    )


def _prepare_dynamic_cloud(
    frames: Sequence[FrameCloud],
    offset: int,
    dynamic_mode: str,
    dynamic_window: int,
    voxel_size: float,
    max_dynamic_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    offsets = _dynamic_frame_offsets(
        total_frames=len(frames),
        offset=offset,
        dynamic_mode=dynamic_mode,
        dynamic_window=dynamic_window,
    )
    dynamic_points, dynamic_colors = _concat_frame_clouds(frames, offsets)
    if voxel_size > 0 and dynamic_points.shape[0] > 0:
        dynamic_points, dynamic_colors = _voxel_downsample_with_color(
            dynamic_points, dynamic_colors, voxel_size
        )

    rng = np.random.default_rng(seed + offset)
    dynamic_points, dynamic_colors = _random_subsample_points(
        dynamic_points, dynamic_colors, max_dynamic_points, rng
    )
    return dynamic_points, dynamic_colors, offsets


def _write_frame_exports(frames: Sequence[FrameCloud], export_frames_dir: Path) -> List[str]:
    export_frames_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for offset, frame in enumerate(frames):
        output_path = export_frames_dir / f"frame_{offset:05d}.ply"
        _write_colored_ply(output_path, frame.points, frame.colors)
        paths.append(str(output_path))
    return paths


def _visualize_timeline_matplotlib(
    frames: Sequence[FrameCloud],
    static_points: np.ndarray,
    static_colors: np.ndarray,
    initial_offset: int,
    dynamic_mode: str,
    dynamic_window: int,
    voxel_size: float,
    max_dynamic_points: int,
    point_size: float,
    seed: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
    except ImportError as exc:
        raise ImportError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        ) from exc

    axis_limits = _compute_axis_limits(static_points)
    initial_dynamic_points, initial_dynamic_colors, _ = _prepare_dynamic_cloud(
        frames=frames,
        offset=initial_offset,
        dynamic_mode=dynamic_mode,
        dynamic_window=dynamic_window,
        voxel_size=voxel_size,
        max_dynamic_points=max_dynamic_points,
        seed=seed,
    )

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_axes([0.06, 0.18, 0.88, 0.76], projection="3d")
    slider_ax = fig.add_axes([0.18, 0.08, 0.58, 0.03])
    prev_ax = fig.add_axes([0.78, 0.075, 0.07, 0.045])
    next_ax = fig.add_axes([0.86, 0.075, 0.07, 0.045])

    ax.set_xlim(axis_limits[0])
    ax.set_ylim(axis_limits[1])
    ax.set_zlim(axis_limits[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_equal_box_aspect(ax)

    static_scatter = ax.scatter(
        static_points[:, 0] if static_points.shape[0] > 0 else [],
        static_points[:, 1] if static_points.shape[0] > 0 else [],
        static_points[:, 2] if static_points.shape[0] > 0 else [],
        s=point_size,
        c=static_colors / 255.0 if static_colors.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32),
        alpha=0.18,
        depthshade=False,
        label="static(global)",
    )
    dynamic_scatter = ax.scatter(
        initial_dynamic_points[:, 0] if initial_dynamic_points.shape[0] > 0 else [],
        initial_dynamic_points[:, 1] if initial_dynamic_points.shape[0] > 0 else [],
        initial_dynamic_points[:, 2] if initial_dynamic_points.shape[0] > 0 else [],
        s=point_size * 1.2,
        c=initial_dynamic_colors / 255.0 if initial_dynamic_colors.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32),
        alpha=0.95,
        depthshade=False,
        label="dynamic(window)",
    )
    ax.legend(loc="upper right")

    initial_frame = frames[initial_offset]
    ax.set_title(
        _format_title(
            frame_index=initial_frame.source_frame_index,
            timestamp=initial_frame.timestamp,
            dynamic_mode=dynamic_mode,
            dynamic_window=dynamic_window,
            static_count=static_points.shape[0],
            dynamic_count=initial_dynamic_points.shape[0],
        )
    )

    slider = Slider(
        ax=slider_ax,
        label="Frame",
        valmin=0,
        valmax=len(frames) - 1,
        valinit=initial_offset,
        valstep=1,
    )
    prev_button = Button(prev_ax, "Prev")
    next_button = Button(next_ax, "Next")

    def update(offset_value: int) -> None:
        offset = int(offset_value)
        dynamic_points, dynamic_colors, _ = _prepare_dynamic_cloud(
            frames=frames,
            offset=offset,
            dynamic_mode=dynamic_mode,
            dynamic_window=dynamic_window,
            voxel_size=voxel_size,
            max_dynamic_points=max_dynamic_points,
            seed=seed,
        )

        dynamic_scatter._offsets3d = (
            dynamic_points[:, 0] if dynamic_points.shape[0] > 0 else np.array([]),
            dynamic_points[:, 1] if dynamic_points.shape[0] > 0 else np.array([]),
            dynamic_points[:, 2] if dynamic_points.shape[0] > 0 else np.array([]),
        )
        dynamic_scatter.set_facecolors(dynamic_colors / 255.0)
        dynamic_scatter.set_edgecolors(dynamic_colors / 255.0)

        frame = frames[offset]
        ax.set_title(
            _format_title(
                frame_index=frame.source_frame_index,
                timestamp=frame.timestamp,
                dynamic_mode=dynamic_mode,
                dynamic_window=dynamic_window,
                static_count=static_points.shape[0],
                dynamic_count=dynamic_points.shape[0],
            )
        )
        fig.canvas.draw_idle()

    def on_prev(_event) -> None:
        slider.set_val(max(0, int(slider.val) - 1))

    def on_next(_event) -> None:
        slider.set_val(min(len(frames) - 1, int(slider.val) + 1))

    def on_key(event) -> None:
        if event.key == "left":
            on_prev(None)
        elif event.key == "right":
            on_next(None)

    slider.on_changed(update)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    # keep references live for interactive callbacks in some backends
    _ = static_scatter


def _build_summary(
    sequence_name: str,
    selected_indices: Sequence[int],
    frames: Sequence[FrameCloud],
    dynamic_mode: str,
    dynamic_window: int,
    voxel_size: float,
    pixel_stride: int,
    initial_offset: int,
    initial_dynamic_offsets: Sequence[int],
    initial_dynamic_points: np.ndarray,
    static_points: np.ndarray,
    exported_frame_files: Sequence[str],
) -> Dict[str, object]:
    frame_stats = [
        {
            "offset": int(offset),
            "source_frame_index": int(frame.source_frame_index),
            "timestamp": float(frame.timestamp),
            "point_count": int(frame.points.shape[0]),
        }
        for offset, frame in enumerate(frames)
    ]
    return {
        "sequence": sequence_name,
        "selected_frame_indices": [int(idx) for idx in selected_indices],
        "selected_frame_count": int(len(selected_indices)),
        "dynamic_mode": dynamic_mode,
        "dynamic_window": int(dynamic_window),
        "voxel_size": float(voxel_size),
        "pixel_stride": int(pixel_stride),
        "frames": frame_stats,
        "static_global_points": int(static_points.shape[0]),
        "initial_offset": int(initial_offset),
        "initial_source_frame_index": int(frames[initial_offset].source_frame_index),
        "initial_dynamic_frame_offsets": [int(idx) for idx in initial_dynamic_offsets],
        "initial_dynamic_source_indices": [
            int(frames[idx].source_frame_index) for idx in initial_dynamic_offsets
        ],
        "initial_dynamic_points": int(initial_dynamic_points.shape[0]),
        "exported_frame_files": list(exported_frame_files),
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)

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
    print(f"Dynamic mode: {args.dynamic_mode} (window={args.dynamic_window})")
    print(f"Depth scale: {args.depth_scale}")
    print(f"Voxel size: {args.voxel_size}")

    frames: List[FrameCloud] = []
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

        frames.append(
            FrameCloud(
                source_frame_index=int(frame_idx),
                timestamp=float(frame_idx),
                points=frame_points,
                colors=frame_colors,
            )
        )

        if i % 10 == 0 or i == len(selected_indices):
            print(
                f"Processed {i}/{len(selected_indices)} frames, "
                f"latest points: {frame_points.shape[0]}"
            )

    static_points_raw, static_colors_raw = _concat_frame_clouds(
        frames,
        offsets=list(range(len(frames))),
    )
    if args.voxel_size > 0 and static_points_raw.shape[0] > 0:
        static_points_raw, static_colors_raw = _voxel_downsample_with_color(
            static_points_raw, static_colors_raw, args.voxel_size
        )

    rng = np.random.default_rng(args.seed)
    static_points, static_colors = _random_subsample_points(
        static_points_raw, static_colors_raw, args.max_static_points, rng
    )

    initial_offset = _resolve_initial_offset(len(frames), args.initial_index)
    initial_dynamic_points, _initial_dynamic_colors, initial_dynamic_offsets = _prepare_dynamic_cloud(
        frames=frames,
        offset=initial_offset,
        dynamic_mode=args.dynamic_mode,
        dynamic_window=args.dynamic_window,
        voxel_size=args.voxel_size,
        max_dynamic_points=args.max_dynamic_points,
        seed=args.seed,
    )

    exported_frame_files: List[str] = []
    if args.export_frames_dir is not None:
        export_dir = Path(args.export_frames_dir).expanduser().resolve()
        exported_frame_files = _write_frame_exports(frames, export_dir)
        print(f"Exported {len(exported_frame_files)} frame PLY files to: {export_dir}")

    summary = _build_summary(
        sequence_name=sequence_dir.name,
        selected_indices=selected_indices,
        frames=frames,
        dynamic_mode=args.dynamic_mode,
        dynamic_window=args.dynamic_window,
        voxel_size=args.voxel_size,
        pixel_stride=args.pixel_stride,
        initial_offset=initial_offset,
        initial_dynamic_offsets=initial_dynamic_offsets,
        initial_dynamic_points=initial_dynamic_points,
        static_points=static_points,
        exported_frame_files=exported_frame_files,
    )

    print(f"Static(global) points for view: {static_points.shape[0]}")
    print(f"Initial dynamic points: {initial_dynamic_points.shape[0]}")
    print(f"Initial dynamic offsets: {initial_dynamic_offsets}")

    if args.export_summary_json is not None:
        summary_path = Path(args.export_summary_json).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Exported summary JSON: {summary_path}")

    if args.backend == "none":
        return

    _visualize_timeline_matplotlib(
        frames=frames,
        static_points=static_points,
        static_colors=static_colors,
        initial_offset=initial_offset,
        dynamic_mode=args.dynamic_mode,
        dynamic_window=args.dynamic_window,
        voxel_size=args.voxel_size,
        max_dynamic_points=args.max_dynamic_points,
        point_size=args.point_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

