"""Visualize and export sequence-level static/dynamic point clouds."""

from __future__ import annotations

import argparse
import colorsys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FrameData:
    path: Path
    frame_index: int
    timestamp: float
    static_points: np.ndarray
    dynamic_points: np.ndarray
    dynamic_instance_ids: np.ndarray
    static_colors_rgb: Optional[np.ndarray]
    dynamic_colors_rgb: Optional[np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Accumulate replay frames from scripts/run_separation_replay.py to "
            "inspect sequence-level static and dynamic point clouds."
        )
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory containing frame_*.npz files.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="First frame index (inclusive) inside frames_dir.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="Last frame index (exclusive). -1 means use all remaining frames.",
    )
    parser.add_argument(
        "--dynamic_mode",
        type=str,
        choices=["latest", "window", "all"],
        default="window",
        help="How to accumulate dynamic points for the sequence view.",
    )
    parser.add_argument(
        "--dynamic_window",
        type=int,
        default=4,
        help="Number of trailing frames used when --dynamic_mode=window.",
    )
    parser.add_argument(
        "--anchor_frame",
        type=int,
        default=-1,
        help="Anchor frame inside the selected range. -1 means the last selected frame.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Voxel size for accumulation downsampling. <=0 disables downsampling.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "open3d", "matplotlib", "none"],
        default="auto",
        help="Visualization backend. Use 'none' for export-only mode.",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        choices=["semantic", "rgb"],
        default="semantic",
        help="Point color mode: semantic or RGB loaded from frame NPZ.",
    )
    parser.add_argument(
        "--max_static_points",
        type=int,
        default=150000,
        help="Max static points shown in the viewer after subsampling.",
    )
    parser.add_argument(
        "--max_dynamic_points",
        type=int,
        default=100000,
        help="Max dynamic points shown in the viewer after subsampling.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for visualization subsampling.")
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.0,
        help="Scatter point size for matplotlib backend.",
    )
    parser.add_argument(
        "--save_png",
        type=str,
        default=None,
        help="Optional output image path for matplotlib backend.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    parser.add_argument(
        "--export_static_ply",
        type=str,
        default=None,
        help="Optional path for accumulated static point cloud PLY export.",
    )
    parser.add_argument(
        "--export_dynamic_ply",
        type=str,
        default=None,
        help="Optional path for accumulated dynamic point cloud PLY export.",
    )
    parser.add_argument(
        "--export_combined_ply",
        type=str,
        default=None,
        help="Optional path for a combined static/dynamic colored PLY export.",
    )
    parser.add_argument(
        "--export_instances_dir",
        type=str,
        default=None,
        help="Optional directory for per-instance dynamic point cloud PLY exports.",
    )
    parser.add_argument(
        "--min_instance_points",
        type=int,
        default=1,
        help="Minimum point count required to export a dynamic instance.",
    )
    parser.add_argument(
        "--export_summary_json",
        type=str,
        default=None,
        help="Optional path to write a JSON summary of the accumulated sequence.",
    )
    return parser.parse_args()


def _resolve_frame_files(frames_dir: Path) -> List[Path]:
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames_dir not found: {frames_dir}")
    frame_files = sorted(frames_dir.glob("frame_*.npz"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.npz found under: {frames_dir}")
    return frame_files


def _select_frame_files(frame_files: Sequence[Path], start_index: int, end_index: int) -> List[Path]:
    if start_index < 0:
        raise ValueError("--start_index must be >= 0.")

    resolved_end = len(frame_files) if end_index < 0 else end_index
    if resolved_end > len(frame_files):
        raise IndexError(
            f"--end_index ({resolved_end}) exceeds available frame count ({len(frame_files)})."
        )
    if start_index >= resolved_end:
        raise ValueError(
            f"Empty frame range: start_index={start_index}, end_index={resolved_end}."
        )
    return list(frame_files[start_index:resolved_end])


def _load_optional_colors(
    payload: np.lib.npyio.NpzFile,
    key: str,
    point_count: int,
    frame_path: Path,
) -> Optional[np.ndarray]:
    values = payload.get(key)
    if values is None:
        return None
    colors = np.asarray(values, dtype=np.uint8)
    if colors.shape != (point_count, 3):
        print(
            f"Warning: {frame_path.name} has invalid {key} shape {colors.shape}, "
            "falling back to semantic colors."
        )
        return None
    return colors


def _load_frames(frame_files: Sequence[Path], start_index: int) -> List[FrameData]:
    frames: List[FrameData] = []
    for offset, frame_path in enumerate(frame_files):
        with np.load(str(frame_path), allow_pickle=False) as payload:
            static_points = np.asarray(payload["static_points_world"], dtype=np.float32)
            dynamic_points = np.asarray(payload["dynamic_points_world"], dtype=np.float32)
            dynamic_instance_ids = np.asarray(payload["dynamic_instance_ids"], dtype=np.int64)
            timestamp = float(np.asarray(payload["timestamp"]).reshape(-1)[0])
            static_colors_rgb = _load_optional_colors(
                payload, "static_colors_rgb", static_points.shape[0], frame_path
            )
            dynamic_colors_rgb = _load_optional_colors(
                payload, "dynamic_colors_rgb", dynamic_points.shape[0], frame_path
            )

        if static_points.ndim != 2 or static_points.shape[1] != 3:
            raise ValueError(
                f"{frame_path}: static_points_world must have shape (N, 3), got {static_points.shape}."
            )
        if dynamic_points.ndim != 2 or dynamic_points.shape[1] != 3:
            raise ValueError(
                f"{frame_path}: dynamic_points_world must have shape (N, 3), got {dynamic_points.shape}."
            )
        if dynamic_instance_ids.ndim != 1:
            raise ValueError(
                f"{frame_path}: dynamic_instance_ids must be 1D, got {dynamic_instance_ids.shape}."
            )
        if dynamic_points.shape[0] != dynamic_instance_ids.shape[0]:
            raise ValueError(
                f"{frame_path}: dynamic point count ({dynamic_points.shape[0]}) does not match "
                f"instance id count ({dynamic_instance_ids.shape[0]})."
            )

        frames.append(
            FrameData(
                path=frame_path,
                frame_index=start_index + offset,
                timestamp=timestamp,
                static_points=static_points,
                dynamic_points=dynamic_points,
                dynamic_instance_ids=dynamic_instance_ids,
                static_colors_rgb=static_colors_rgb,
                dynamic_colors_rgb=dynamic_colors_rgb,
            )
        )
    return frames


def _resolve_anchor(frames: Sequence[FrameData], anchor_frame: int) -> int:
    if anchor_frame < 0:
        return len(frames) - 1
    if anchor_frame >= len(frames):
        raise IndexError(
            f"--anchor_frame ({anchor_frame}) exceeds selected frame count ({len(frames)})."
        )
    return anchor_frame


def _select_dynamic_frames(
    frames: Sequence[FrameData],
    dynamic_mode: str,
    dynamic_window: int,
    anchor_offset: int,
) -> List[FrameData]:
    if dynamic_mode == "all":
        return list(frames)
    if dynamic_mode == "latest":
        return [frames[anchor_offset]]
    if dynamic_window <= 0:
        raise ValueError("--dynamic_window must be > 0 when --dynamic_mode=window.")
    start = max(0, anchor_offset - dynamic_window + 1)
    return list(frames[start : anchor_offset + 1])


def _concat_points(point_sets: Sequence[np.ndarray]) -> np.ndarray:
    valid = [points for points in point_sets if points.shape[0] > 0]
    if not valid:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(valid, axis=0).astype(np.float32, copy=False)


def _static_semantic_colors(count: int) -> np.ndarray:
    return np.tile(np.array([[166, 166, 166]], dtype=np.uint8), (count, 1))


def _instance_color(instance_id: int) -> np.ndarray:
    if instance_id < 0:
        return np.array([229, 57, 53], dtype=np.uint8)
    hue = (instance_id * 0.61803398875) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
    return np.asarray([int(channel * 255) for channel in rgb], dtype=np.uint8)


def _dynamic_semantic_colors(instance_ids: np.ndarray) -> np.ndarray:
    if instance_ids.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    unique_ids = np.unique(instance_ids)
    color_map: Dict[int, np.ndarray] = {
        int(instance_id): _instance_color(int(instance_id)) for instance_id in unique_ids
    }
    return np.stack([color_map[int(instance_id)] for instance_id in instance_ids], axis=0)


def _concat_static_for_color_mode(
    frames: Sequence[FrameData],
    color_mode: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    point_sets = [frame.static_points for frame in frames if frame.static_points.shape[0] > 0]
    if not point_sets:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        if color_mode == "rgb":
            return empty_points, np.zeros((0, 3), dtype=np.uint8)
        return empty_points, None

    points = np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
    if color_mode != "rgb":
        return points, None

    color_sets: List[np.ndarray] = []
    for frame in frames:
        if frame.static_points.shape[0] == 0:
            continue
        if frame.static_colors_rgb is not None:
            color_sets.append(frame.static_colors_rgb.astype(np.uint8, copy=False))
        else:
            color_sets.append(_static_semantic_colors(frame.static_points.shape[0]))
    colors = np.concatenate(color_sets, axis=0).astype(np.uint8, copy=False)
    return points, colors


def _concat_dynamic_for_color_mode(
    frames: Sequence[FrameData],
    color_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    point_sets = [frame.dynamic_points for frame in frames if frame.dynamic_points.shape[0] > 0]
    id_sets = [frame.dynamic_instance_ids for frame in frames if frame.dynamic_instance_ids.shape[0] > 0]
    if not point_sets:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_ids = np.zeros((0,), dtype=np.int64)
        if color_mode == "rgb":
            return empty_points, empty_ids, np.zeros((0, 3), dtype=np.uint8)
        return empty_points, empty_ids, None

    points = np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
    instance_ids = np.concatenate(id_sets, axis=0).astype(np.int64, copy=False)
    if color_mode != "rgb":
        return points, instance_ids, None

    color_sets: List[np.ndarray] = []
    for frame in frames:
        point_count = frame.dynamic_points.shape[0]
        if point_count == 0:
            continue
        if frame.dynamic_colors_rgb is not None:
            color_sets.append(frame.dynamic_colors_rgb.astype(np.uint8, copy=False))
        else:
            color_sets.append(_dynamic_semantic_colors(frame.dynamic_instance_ids))
    colors = np.concatenate(color_sets, axis=0).astype(np.uint8, copy=False)
    return points, instance_ids, colors


def _voxel_downsample_points(
    points: np.ndarray,
    voxel_size: float,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    points_arr = points.astype(np.float32, copy=False)
    if colors is not None:
        colors_arr = np.asarray(colors, dtype=np.uint8)
        if colors_arr.shape != (points_arr.shape[0], 3):
            raise ValueError(
                f"colors must have shape ({points_arr.shape[0]}, 3), got {colors_arr.shape}."
            )
    else:
        colors_arr = None

    if voxel_size <= 0 or points_arr.shape[0] <= 1:
        return points_arr, None if colors_arr is None else colors_arr

    voxel_keys = np.floor(points_arr / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxel_keys, axis=0, return_inverse=True, return_counts=True)

    point_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(point_sums, inverse, points_arr.astype(np.float64, copy=False))
    down_points = (point_sums / counts[:, None]).astype(np.float32)

    if colors_arr is None:
        return down_points, None

    color_sums = np.zeros((counts.shape[0], 3), dtype=np.float64)
    np.add.at(color_sums, inverse, colors_arr.astype(np.float64, copy=False))
    down_colors = np.clip(np.round(color_sums / counts[:, None]), 0, 255).astype(np.uint8)
    return down_points, down_colors


def _voxel_downsample_dynamic(
    points: np.ndarray,
    instance_ids: np.ndarray,
    voxel_size: float,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    points_arr = points.astype(np.float32, copy=False)
    instance_ids_arr = instance_ids.astype(np.int64, copy=False)

    if colors is not None:
        colors_arr = np.asarray(colors, dtype=np.uint8)
        if colors_arr.shape != (points_arr.shape[0], 3):
            raise ValueError(
                f"colors must have shape ({points_arr.shape[0]}, 3), got {colors_arr.shape}."
            )
    else:
        colors_arr = None

    if voxel_size <= 0 or points_arr.shape[0] <= 1:
        return points_arr, instance_ids_arr, None if colors_arr is None else colors_arr

    down_points: List[np.ndarray] = []
    down_ids: List[np.ndarray] = []
    down_colors: List[np.ndarray] = []

    for instance_id in np.unique(instance_ids_arr):
        mask = instance_ids_arr == instance_id
        instance_points = points_arr[mask]
        instance_colors = None if colors_arr is None else colors_arr[mask]
        points_down, colors_down = _voxel_downsample_points(
            points=instance_points,
            voxel_size=voxel_size,
            colors=instance_colors,
        )
        if points_down.shape[0] == 0:
            continue
        down_points.append(points_down)
        down_ids.append(np.full((points_down.shape[0],), int(instance_id), dtype=np.int64))
        if colors_down is not None:
            down_colors.append(colors_down)

    if not down_points:
        empty_points = np.zeros((0, 3), dtype=np.float32)
        empty_ids = np.zeros((0,), dtype=np.int64)
        if colors_arr is None:
            return empty_points, empty_ids, None
        return empty_points, empty_ids, np.zeros((0, 3), dtype=np.uint8)

    points_out = np.concatenate(down_points, axis=0).astype(np.float32, copy=False)
    ids_out = np.concatenate(down_ids, axis=0).astype(np.int64, copy=False)

    if colors_arr is None:
        return points_out, ids_out, None

    colors_out = np.concatenate(down_colors, axis=0).astype(np.uint8, copy=False)
    return points_out, ids_out, colors_out


def _subsample(
    points: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if colors is not None and colors.shape != (points.shape[0], 3):
        raise ValueError(
            f"colors must have shape ({points.shape[0]}, 3), got {colors.shape}."
        )
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    points_down = points[keep]
    colors_down = None if colors is None else colors[keep]
    return points_down, colors_down


def _subsample_dynamic(
    points: np.ndarray,
    instance_ids: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if colors is not None and colors.shape != (points.shape[0], 3):
        raise ValueError(
            f"colors must have shape ({points.shape[0]}, 3), got {colors.shape}."
        )
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, instance_ids, colors
    keep = rng.choice(points.shape[0], size=max_points, replace=False)
    points_down = points[keep]
    ids_down = instance_ids[keep]
    colors_down = None if colors is None else colors[keep]
    return points_down, ids_down, colors_down


def _resolve_static_colors(
    color_mode: str,
    point_count: int,
    rgb_colors: Optional[np.ndarray],
) -> np.ndarray:
    if color_mode == "rgb" and rgb_colors is not None:
        return rgb_colors.astype(np.uint8, copy=False)
    return _static_semantic_colors(point_count)


def _resolve_dynamic_colors(
    color_mode: str,
    instance_ids: np.ndarray,
    rgb_colors: Optional[np.ndarray],
) -> np.ndarray:
    if color_mode == "rgb" and rgb_colors is not None:
        return rgb_colors.astype(np.uint8, copy=False)
    return _dynamic_semantic_colors(instance_ids)


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"Point/color count mismatch for {path}: {points.shape[0]} vs {colors.shape[0]}."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def _export_instances(
    export_dir: Path,
    points: np.ndarray,
    instance_ids: np.ndarray,
    colors: np.ndarray,
    min_instance_points: int,
) -> List[str]:
    if points.shape[0] != instance_ids.shape[0]:
        raise ValueError(
            f"instance_ids length ({instance_ids.shape[0]}) must match points ({points.shape[0]})."
        )
    if colors.shape != (points.shape[0], 3):
        raise ValueError(
            f"colors must have shape ({points.shape[0]}, 3), got {colors.shape}."
        )

    export_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for instance_id in sorted(np.unique(instance_ids).tolist()):
        mask = instance_ids == instance_id
        instance_points = points[mask]
        if instance_points.shape[0] < min_instance_points:
            continue

        if instance_id < 0:
            file_name = f"instance_neg{abs(int(instance_id)):04d}.ply"
        else:
            file_name = f"instance_{int(instance_id):04d}.ply"
        path = export_dir / file_name
        instance_colors = colors[mask]
        _write_ply(path, instance_points, instance_colors)
        written.append(str(path))
    return written


def _visualize_open3d(
    static_points: np.ndarray,
    dynamic_points: np.ndarray,
    static_colors: np.ndarray,
    dynamic_colors: np.ndarray,
    title: str,
) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is not installed. Install it with: pip install open3d"
        ) from exc

    geometries = []
    if static_points.shape[0] > 0:
        static_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(static_points)
        static_pcd.colors = o3d.utility.Vector3dVector(static_colors / 255.0)
        geometries.append(static_pcd)

    if dynamic_points.shape[0] > 0:
        dynamic_pcd = o3d.geometry.PointCloud()
        dynamic_pcd.points = o3d.utility.Vector3dVector(dynamic_points)
        dynamic_pcd.colors = o3d.utility.Vector3dVector(dynamic_colors / 255.0)
        geometries.append(dynamic_pcd)

    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    o3d.visualization.draw_geometries(geometries, window_name=title)


def _visualize_matplotlib(
    static_points: np.ndarray,
    dynamic_points: np.ndarray,
    static_colors: np.ndarray,
    dynamic_colors: np.ndarray,
    title: str,
    point_size: float,
    save_png: str | None,
    no_show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is not installed. Install it with: pip install matplotlib"
        ) from exc

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if static_points.shape[0] > 0:
        ax.scatter(
            static_points[:, 0],
            static_points[:, 1],
            static_points[:, 2],
            s=point_size,
            c=static_colors / 255.0,
            alpha=0.25,
            label="static",
        )

    if dynamic_points.shape[0] > 0:
        ax.scatter(
            dynamic_points[:, 0],
            dynamic_points[:, 1],
            dynamic_points[:, 2],
            s=point_size,
            c=dynamic_colors / 255.0,
            alpha=0.85,
            label="dynamic",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if save_png is not None:
        save_path = Path(save_png)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def _build_summary(
    frames: Sequence[FrameData],
    dynamic_frames: Sequence[FrameData],
    raw_static_count: int,
    raw_dynamic_count: int,
    static_points: np.ndarray,
    dynamic_points: np.ndarray,
    dynamic_instance_ids: np.ndarray,
) -> Dict[str, object]:
    instance_counts = {
        str(int(instance_id)): int((dynamic_instance_ids == instance_id).sum())
        for instance_id in np.unique(dynamic_instance_ids)
    }
    return {
        "selected_frame_indices": [int(frame.frame_index) for frame in frames],
        "selected_frame_count": int(len(frames)),
        "timestamps": [float(frame.timestamp) for frame in frames],
        "dynamic_frame_indices": [int(frame.frame_index) for frame in dynamic_frames],
        "dynamic_frame_count": int(len(dynamic_frames)),
        "static_raw_points": int(raw_static_count),
        "dynamic_raw_points": int(raw_dynamic_count),
        "static_downsampled_points": int(static_points.shape[0]),
        "dynamic_downsampled_points": int(dynamic_points.shape[0]),
        "dynamic_instance_counts": instance_counts,
    }


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    frames_dir = Path(args.frames_dir)
    frame_files = _resolve_frame_files(frames_dir)
    selected_files = _select_frame_files(frame_files, args.start_index, args.end_index)
    frames = _load_frames(selected_files, args.start_index)

    anchor_offset = _resolve_anchor(frames, args.anchor_frame)
    dynamic_frames = _select_dynamic_frames(
        frames=frames,
        dynamic_mode=args.dynamic_mode,
        dynamic_window=args.dynamic_window,
        anchor_offset=anchor_offset,
    )

    raw_static_points, raw_static_rgb = _concat_static_for_color_mode(frames, args.color_mode)
    raw_dynamic_points, raw_dynamic_instance_ids, raw_dynamic_rgb = _concat_dynamic_for_color_mode(
        dynamic_frames, args.color_mode
    )

    static_points, static_rgb = _voxel_downsample_points(
        raw_static_points, args.voxel_size, raw_static_rgb
    )
    dynamic_points, dynamic_instance_ids, dynamic_rgb = _voxel_downsample_dynamic(
        raw_dynamic_points,
        raw_dynamic_instance_ids,
        args.voxel_size,
        raw_dynamic_rgb,
    )

    summary = _build_summary(
        frames=frames,
        dynamic_frames=dynamic_frames,
        raw_static_count=raw_static_points.shape[0],
        raw_dynamic_count=raw_dynamic_points.shape[0],
        static_points=static_points,
        dynamic_points=dynamic_points,
        dynamic_instance_ids=dynamic_instance_ids,
    )

    print(f"Loaded {len(frames)} frames from: {frames_dir}")
    print(
        f"- selected frame indices: {summary['selected_frame_indices'][0]}"
        f"..{summary['selected_frame_indices'][-1]}"
    )
    print(
        f"- static points: raw={summary['static_raw_points']} "
        f"downsampled={summary['static_downsampled_points']}"
    )
    print(
        f"- dynamic mode: {args.dynamic_mode}"
        f" | raw={summary['dynamic_raw_points']}"
        f" | downsampled={summary['dynamic_downsampled_points']}"
    )
    print(f"- dynamic frames: {summary['dynamic_frame_indices']}")
    print(f"- dynamic instance counts: {summary['dynamic_instance_counts']}")
    print(f"- color mode: {args.color_mode}")

    static_colors = _resolve_static_colors(args.color_mode, static_points.shape[0], static_rgb)
    dynamic_colors = _resolve_dynamic_colors(args.color_mode, dynamic_instance_ids, dynamic_rgb)

    if args.export_static_ply is not None:
        path = Path(args.export_static_ply)
        _write_ply(path, static_points, static_colors)
        print(f"Exported static PLY to: {path}")

    if args.export_dynamic_ply is not None:
        path = Path(args.export_dynamic_ply)
        _write_ply(path, dynamic_points, dynamic_colors)
        print(f"Exported dynamic PLY to: {path}")

    if args.export_combined_ply is not None:
        path = Path(args.export_combined_ply)
        combined_points = np.concatenate([static_points, dynamic_points], axis=0)
        combined_colors = np.concatenate([static_colors, dynamic_colors], axis=0)
        _write_ply(path, combined_points, combined_colors)
        print(f"Exported combined PLY to: {path}")

    if args.export_instances_dir is not None:
        export_dir = Path(args.export_instances_dir)
        written = _export_instances(
            export_dir=export_dir,
            points=dynamic_points,
            instance_ids=dynamic_instance_ids,
            colors=dynamic_colors,
            min_instance_points=args.min_instance_points,
        )
        print(f"Exported {len(written)} instance PLY files to: {export_dir}")

    if args.export_summary_json is not None:
        summary_path = Path(args.export_summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Exported summary JSON to: {summary_path}")

    if args.backend == "none":
        return

    static_view, static_view_colors = _subsample(
        static_points, args.max_static_points, rng, static_colors
    )
    dynamic_view, dynamic_view_ids, dynamic_view_colors = _subsample_dynamic(
        dynamic_points,
        dynamic_instance_ids,
        args.max_dynamic_points,
        rng,
        dynamic_colors,
    )
    if static_view_colors is None:
        static_view_colors = _static_semantic_colors(static_view.shape[0])
    if dynamic_view_colors is None:
        dynamic_view_colors = _dynamic_semantic_colors(dynamic_view_ids)

    title = (
        f"frames {summary['selected_frame_indices'][0]}..{summary['selected_frame_indices'][-1]}"
        f" | static={static_view.shape[0]} dynamic={dynamic_view.shape[0]}"
        f" | dynamic_mode={args.dynamic_mode} | color={args.color_mode}"
    )

    backend = args.backend
    if backend == "auto":
        try:
            _visualize_open3d(
                static_view,
                dynamic_view,
                static_view_colors,
                dynamic_view_colors,
                title,
            )
            return
        except ImportError:
            backend = "matplotlib"

    if backend == "open3d":
        _visualize_open3d(
            static_view,
            dynamic_view,
            static_view_colors,
            dynamic_view_colors,
            title,
        )
    else:
        _visualize_matplotlib(
            static_points=static_view,
            dynamic_points=dynamic_view,
            static_colors=static_view_colors,
            dynamic_colors=dynamic_view_colors,
            title=title,
            point_size=args.point_size,
            save_png=args.save_png,
            no_show=args.no_show,
        )


if __name__ == "__main__":
    main()
