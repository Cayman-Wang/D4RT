"""Interactive timeline viewer for separation replay frame NPZ files."""

from __future__ import annotations

import argparse
import colorsys
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
            "Inspect replay_full/frames/frame_*.npz with an interactive timeline "
            "slider for dynamic point clouds."
        )
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory containing frame_*.npz files from run_separation_replay.py.",
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
        "--initial_index",
        type=int,
        default=-1,
        help="Initial slider position within the selected range. -1 means the last frame.",
    )
    parser.add_argument(
        "--static_mode",
        type=str,
        choices=["all", "upto", "current", "none"],
        default="all",
        help="How static points are shown while scrubbing the timeline.",
    )
    parser.add_argument(
        "--dynamic_mode",
        type=str,
        choices=["frame", "window"],
        default="frame",
        help="How dynamic points are shown for the selected frame.",
    )
    parser.add_argument(
        "--dynamic_window",
        type=int,
        default=4,
        help="Number of trailing frames used when --dynamic_mode=window.",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        choices=["semantic", "rgb"],
        default="semantic",
        help="Point color mode: semantic or RGB loaded from frame NPZ.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Voxel size used before visualization. <=0 disables downsampling.",
    )
    parser.add_argument(
        "--max_static_points",
        type=int,
        default=150000,
        help="Subsample static display points to at most this many points.",
    )
    parser.add_argument(
        "--max_dynamic_points",
        type=int,
        default=100000,
        help="Subsample dynamic display points to at most this many points.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=2.0,
        help="Scatter point size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for visualization subsampling.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["matplotlib", "none"],
        default="matplotlib",
        help="Use 'none' for headless validation of the selected timeline data.",
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


def _resolve_initial_offset(frames: Sequence[FrameData], initial_index: int) -> int:
    if initial_index < 0:
        return len(frames) - 1
    if initial_index >= len(frames):
        raise IndexError(
            f"--initial_index ({initial_index}) exceeds selected frame count ({len(frames)})."
        )
    return initial_index


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


def _concat_dynamic_with_rgb_fallback(
    frames: Sequence[FrameData],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_sets = [frame.dynamic_points for frame in frames if frame.dynamic_points.shape[0] > 0]
    id_sets = [frame.dynamic_instance_ids for frame in frames if frame.dynamic_instance_ids.shape[0] > 0]
    if not point_sets:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.uint8),
        )

    points = np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
    instance_ids = np.concatenate(id_sets, axis=0).astype(np.int64, copy=False)

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


def _select_static_data(
    frames: Sequence[FrameData],
    offset: int,
    static_mode: str,
    color_mode: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if static_mode == "none":
        empty = np.zeros((0, 3), dtype=np.float32)
        if color_mode == "rgb":
            return empty, np.zeros((0, 3), dtype=np.uint8)
        return empty, None

    if static_mode == "current":
        points = frames[offset].static_points.astype(np.float32, copy=False)
        if color_mode != "rgb":
            return points, None
        if frames[offset].static_colors_rgb is not None:
            return points, frames[offset].static_colors_rgb.astype(np.uint8, copy=False)
        return points, _static_semantic_colors(points.shape[0])

    frame_subset = frames if static_mode == "all" else frames[: offset + 1]
    points = _concat_points([frame.static_points for frame in frame_subset])
    if color_mode != "rgb":
        return points, None

    color_sets: List[np.ndarray] = []
    for frame in frame_subset:
        point_count = frame.static_points.shape[0]
        if point_count == 0:
            continue
        if frame.static_colors_rgb is not None:
            color_sets.append(frame.static_colors_rgb.astype(np.uint8, copy=False))
        else:
            color_sets.append(_static_semantic_colors(point_count))
    if color_sets:
        colors = np.concatenate(color_sets, axis=0).astype(np.uint8, copy=False)
    else:
        colors = np.zeros((0, 3), dtype=np.uint8)
    return points, colors


def _select_dynamic_data(
    frames: Sequence[FrameData],
    offset: int,
    dynamic_mode: str,
    dynamic_window: int,
    color_mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if dynamic_mode == "frame":
        points = frames[offset].dynamic_points.astype(np.float32, copy=False)
        instance_ids = frames[offset].dynamic_instance_ids.astype(np.int64, copy=False)
        if color_mode != "rgb":
            return points, instance_ids, None
        if frames[offset].dynamic_colors_rgb is not None:
            colors = frames[offset].dynamic_colors_rgb.astype(np.uint8, copy=False)
        else:
            colors = _dynamic_semantic_colors(instance_ids)
        return points, instance_ids, colors

    if dynamic_window <= 0:
        raise ValueError("--dynamic_window must be > 0 when --dynamic_mode=window.")
    start = max(0, offset - dynamic_window + 1)
    frame_subset = frames[start : offset + 1]
    if color_mode != "rgb":
        points = _concat_points([frame.dynamic_points for frame in frame_subset])
        instance_ids = np.concatenate(
            [frame.dynamic_instance_ids for frame in frame_subset if frame.dynamic_instance_ids.shape[0] > 0],
            axis=0,
        ).astype(np.int64, copy=False) if any(frame.dynamic_instance_ids.shape[0] > 0 for frame in frame_subset) else np.zeros((0,), dtype=np.int64)
        return points, instance_ids, None

    return _concat_dynamic_with_rgb_fallback(frame_subset)


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
        points_down, colors_down = _voxel_downsample_points(
            points_arr[mask],
            voxel_size=voxel_size,
            colors=None if colors_arr is None else colors_arr[mask],
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


def _prepare_view_state(
    frames: Sequence[FrameData],
    offset: int,
    static_mode: str,
    dynamic_mode: str,
    dynamic_window: int,
    color_mode: str,
    voxel_size: float,
    max_static_points: int,
    max_dynamic_points: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray | int | float]:
    static_points, static_rgb = _select_static_data(frames, offset, static_mode, color_mode)
    dynamic_points, dynamic_ids, dynamic_rgb = _select_dynamic_data(
        frames,
        offset,
        dynamic_mode,
        dynamic_window,
        color_mode,
    )

    static_points, static_rgb = _voxel_downsample_points(static_points, voxel_size, static_rgb)
    dynamic_points, dynamic_ids, dynamic_rgb = _voxel_downsample_dynamic(
        dynamic_points,
        dynamic_ids,
        voxel_size,
        dynamic_rgb,
    )

    static_points, static_rgb = _subsample(static_points, max_static_points, rng, static_rgb)
    dynamic_points, dynamic_ids, dynamic_rgb = _subsample_dynamic(
        dynamic_points,
        dynamic_ids,
        max_dynamic_points,
        rng,
        dynamic_rgb,
    )
    static_colors = _resolve_static_colors(color_mode, static_points.shape[0], static_rgb)
    dynamic_colors = _resolve_dynamic_colors(color_mode, dynamic_ids, dynamic_rgb)

    return {
        "static_points": static_points,
        "static_colors": static_colors,
        "dynamic_points": dynamic_points,
        "dynamic_ids": dynamic_ids,
        "dynamic_colors": dynamic_colors,
        "frame_index": int(frames[offset].frame_index),
        "timestamp": float(frames[offset].timestamp),
    }


def _compute_axis_limits(frames: Sequence[FrameData]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    point_sets = []
    for frame in frames:
        if frame.static_points.shape[0] > 0:
            point_sets.append(frame.static_points)
        if frame.dynamic_points.shape[0] > 0:
            point_sets.append(frame.dynamic_points)

    if not point_sets:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    points = np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
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


def _format_title(
    state: Dict[str, np.ndarray | int | float],
    static_mode: str,
    dynamic_mode: str,
    dynamic_window: int,
    color_mode: str,
) -> str:
    dynamic_suffix = dynamic_mode
    if dynamic_mode == "window":
        dynamic_suffix = f"window({dynamic_window})"
    return (
        f"frame={state['frame_index']} | timestamp={state['timestamp']:.3f} | "
        f"static={state['static_points'].shape[0]} ({static_mode}) | "
        f"dynamic={state['dynamic_points'].shape[0]} ({dynamic_suffix}) | "
        f"color={color_mode}"
    )


def _set_equal_box_aspect(ax) -> None:
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect((1.0, 1.0, 1.0))


def _visualize_matplotlib(
    frames: Sequence[FrameData],
    initial_offset: int,
    static_mode: str,
    dynamic_mode: str,
    dynamic_window: int,
    color_mode: str,
    voxel_size: float,
    max_static_points: int,
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

    axis_limits = _compute_axis_limits(frames)
    rng = np.random.default_rng(seed)
    initial_state = _prepare_view_state(
        frames=frames,
        offset=initial_offset,
        static_mode=static_mode,
        dynamic_mode=dynamic_mode,
        dynamic_window=dynamic_window,
        color_mode=color_mode,
        voxel_size=voxel_size,
        max_static_points=max_static_points,
        max_dynamic_points=max_dynamic_points,
        rng=rng,
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

    static_points = initial_state["static_points"]
    static_colors = initial_state["static_colors"]
    if static_points.shape[0] > 0:
        static_scatter = ax.scatter(
            static_points[:, 0],
            static_points[:, 1],
            static_points[:, 2],
            s=point_size,
            c=static_colors / 255.0,
            alpha=0.20,
            depthshade=False,
        )
    else:
        static_scatter = ax.scatter(
            [],
            [],
            [],
            s=point_size,
            c=np.zeros((0, 3), dtype=np.float32),
            alpha=0.20,
            depthshade=False,
        )

    dynamic_points = initial_state["dynamic_points"]
    dynamic_colors = initial_state["dynamic_colors"]
    if dynamic_points.shape[0] > 0:
        dynamic_scatter = ax.scatter(
            dynamic_points[:, 0],
            dynamic_points[:, 1],
            dynamic_points[:, 2],
            s=point_size * 1.4,
            c=dynamic_colors / 255.0,
            alpha=0.95,
            depthshade=False,
        )
    else:
        dynamic_scatter = ax.scatter(
            [],
            [],
            [],
            s=point_size * 1.4,
            c=np.zeros((0, 3), dtype=np.float32),
            alpha=0.95,
            depthshade=False,
        )

    ax.set_title(_format_title(initial_state, static_mode, dynamic_mode, dynamic_window, color_mode))

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
        state = _prepare_view_state(
            frames=frames,
            offset=offset,
            static_mode=static_mode,
            dynamic_mode=dynamic_mode,
            dynamic_window=dynamic_window,
            color_mode=color_mode,
            voxel_size=voxel_size,
            max_static_points=max_static_points,
            max_dynamic_points=max_dynamic_points,
            rng=np.random.default_rng(seed + offset),
        )

        static_points_local = state["static_points"]
        static_colors_local = state["static_colors"]
        dynamic_points_local = state["dynamic_points"]
        dynamic_colors_local = state["dynamic_colors"]

        static_scatter._offsets3d = (
            static_points_local[:, 0] if static_points_local.shape[0] > 0 else np.array([]),
            static_points_local[:, 1] if static_points_local.shape[0] > 0 else np.array([]),
            static_points_local[:, 2] if static_points_local.shape[0] > 0 else np.array([]),
        )
        dynamic_scatter._offsets3d = (
            dynamic_points_local[:, 0] if dynamic_points_local.shape[0] > 0 else np.array([]),
            dynamic_points_local[:, 1] if dynamic_points_local.shape[0] > 0 else np.array([]),
            dynamic_points_local[:, 2] if dynamic_points_local.shape[0] > 0 else np.array([]),
        )
        static_scatter.set_facecolors(static_colors_local / 255.0)
        static_scatter.set_edgecolors(static_colors_local / 255.0)
        dynamic_scatter.set_facecolors(dynamic_colors_local / 255.0)
        dynamic_scatter.set_edgecolors(dynamic_colors_local / 255.0)
        ax.set_title(_format_title(state, static_mode, dynamic_mode, dynamic_window, color_mode))
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


def main() -> None:
    args = _parse_args()

    frames_dir = Path(args.frames_dir)
    frame_files = _resolve_frame_files(frames_dir)
    selected_files = _select_frame_files(frame_files, args.start_index, args.end_index)
    frames = _load_frames(selected_files, args.start_index)
    initial_offset = _resolve_initial_offset(frames, args.initial_index)

    if args.backend == "none":
        state = _prepare_view_state(
            frames=frames,
            offset=initial_offset,
            static_mode=args.static_mode,
            dynamic_mode=args.dynamic_mode,
            dynamic_window=args.dynamic_window,
            color_mode=args.color_mode,
            voxel_size=args.voxel_size,
            max_static_points=args.max_static_points,
            max_dynamic_points=args.max_dynamic_points,
            rng=np.random.default_rng(args.seed),
        )
        print(f"Loaded {len(frames)} frames from: {frames_dir}")
        print(
            f"- initial frame={state['frame_index']} timestamp={state['timestamp']:.3f} "
            f"static={state['static_points'].shape[0]} dynamic={state['dynamic_points'].shape[0]}"
        )
        print(f"- color mode: {args.color_mode}")
        return

    _visualize_matplotlib(
        frames=frames,
        initial_offset=initial_offset,
        static_mode=args.static_mode,
        dynamic_mode=args.dynamic_mode,
        dynamic_window=args.dynamic_window,
        color_mode=args.color_mode,
        voxel_size=args.voxel_size,
        max_static_points=args.max_static_points,
        max_dynamic_points=args.max_dynamic_points,
        point_size=args.point_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
