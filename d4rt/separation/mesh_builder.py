"""Minimal static/dynamic mesh builder for replay outputs."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .io_contract import DynamicMeshInfo, SeparationFrame, load_frame_npz, save_frame_npz


def _estimate_frame_dt(timestamps: Sequence[float]) -> float:
    if len(timestamps) < 2:
        return 1.0
    diffs = np.diff(np.asarray(timestamps, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _interval_to_frames(interval_seconds: float, frame_dt: float) -> int:
    if interval_seconds <= 0:
        return 1
    safe_dt = max(float(frame_dt), 1e-6)
    return max(1, int(math.ceil(interval_seconds / safe_dt)))


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] <= 1 or voxel_size <= 0:
        return pts

    quantized = np.floor(pts / float(voxel_size)).astype(np.int64)
    _, unique_indices = np.unique(quantized, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return pts[unique_indices]


def _bbox_mesh(points: np.ndarray, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extent = np.maximum(maxs - mins, float(padding))
    half_pad = np.maximum(extent * 0.05, float(padding))
    mins = mins - half_pad
    maxs = maxs + half_pad

    vertices = np.array(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _build_mesh_geometry(
    points: np.ndarray,
    *,
    voxel_size: float,
    min_points_for_hull: int,
    bbox_padding: float,
) -> Tuple[np.ndarray, np.ndarray]:
    pts = _voxel_downsample(points, voxel_size=voxel_size)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    centered = pts - pts.mean(axis=0, keepdims=True)
    if pts.shape[0] >= min_points_for_hull and np.linalg.matrix_rank(centered) >= 3:
        try:
            hull = ConvexHull(pts)
            return pts.astype(np.float32), hull.simplices.astype(np.int32)
        except QhullError:
            pass

    return _bbox_mesh(pts, padding=bbox_padding)


def _write_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {vertices.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        fp.write(f"element face {faces.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        for vertex in vertices:
            fp.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            fp.write(f"3 {face[0]} {face[1]} {face[2]}\n")


@dataclass
class MeshBuilderConfig:
    static_export_interval_seconds: float = 2.0
    dynamic_export_interval_seconds: float = 0.5
    dynamic_window_frames: int = 8
    voxel_size: float = 0.05
    min_points_for_hull: int = 4
    bbox_padding: float = 0.02
    include_untracked_dynamic: bool = True


class SeparationMeshBuilder:
    """Build minimal meshes from replay frame outputs.

    The builder operates only on replay-frame NPZ files and writes a new output
    directory containing enriched frame files plus generated mesh assets.
    """

    def __init__(self, config: Optional[MeshBuilderConfig] = None):
        self.config = config or MeshBuilderConfig()

    def build_from_frames_dir(self, frames_dir: str | Path, output_dir: str | Path) -> Dict[str, object]:
        frames_path = Path(frames_dir)
        frame_paths = sorted(frames_path.glob("frame_*.npz"))
        frames = [load_frame_npz(path) for path in frame_paths]
        return self.build(frames, output_dir=output_dir, frame_names=[path.name for path in frame_paths])

    def build(
        self,
        frames: Sequence[SeparationFrame],
        *,
        output_dir: str | Path,
        frame_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        output_root = Path(output_dir)
        frames_out_dir = output_root / "frames"
        static_mesh_dir = output_root / "meshes" / "static"
        dynamic_mesh_root = output_root / "meshes" / "dynamic"
        output_root.mkdir(parents=True, exist_ok=True)
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        static_mesh_dir.mkdir(parents=True, exist_ok=True)
        dynamic_mesh_root.mkdir(parents=True, exist_ok=True)

        if frame_names is None:
            frame_names = [f"frame_{idx:06d}.npz" for idx in range(len(frames))]
        if len(frame_names) != len(frames):
            raise ValueError("frame_names length must match frames length.")

        timestamps = [frame.timestamp for frame in frames]
        frame_dt = _estimate_frame_dt(timestamps)
        static_interval_frames = _interval_to_frames(
            self.config.static_export_interval_seconds, frame_dt
        )
        dynamic_interval_frames = _interval_to_frames(
            self.config.dynamic_export_interval_seconds, frame_dt
        )

        static_accumulator: List[np.ndarray] = []
        dynamic_buffers: Dict[int, Deque[Tuple[int, np.ndarray]]] = {}
        latest_static_mesh: Optional[str] = None
        latest_dynamic_meshes: Dict[int, DynamicMeshInfo] = {}

        exported_static_meshes = 0
        exported_dynamic_meshes = 0
        per_frame_active_instances: List[int] = []

        for frame_index, (frame_name, frame) in enumerate(zip(frame_names, frames)):
            static_accumulator.append(frame.static_points_world)

            if frame.dynamic_count > 0:
                instance_ids = np.unique(frame.dynamic_instance_ids)
                for instance_id in instance_ids.tolist():
                    instance_id = int(instance_id)
                    if instance_id < 0 and not self.config.include_untracked_dynamic:
                        continue
                    mask = frame.dynamic_instance_ids == instance_id
                    instance_points = frame.dynamic_points_world[mask]
                    if instance_points.shape[0] == 0:
                        continue
                    dynamic_buffers.setdefault(instance_id, deque()).append((frame_index, instance_points))

            oldest_valid_frame = frame_index - self.config.dynamic_window_frames + 1
            active_instance_ids: List[int] = []
            for instance_id in list(dynamic_buffers.keys()):
                buffer = dynamic_buffers[instance_id]
                while buffer and buffer[0][0] < oldest_valid_frame:
                    buffer.popleft()
                if buffer:
                    active_instance_ids.append(instance_id)
                else:
                    dynamic_buffers.pop(instance_id, None)
                    latest_dynamic_meshes.pop(instance_id, None)

            should_export_static = latest_static_mesh is None or (frame_index % static_interval_frames == 0)
            if should_export_static:
                static_points = np.concatenate(static_accumulator, axis=0) if static_accumulator else np.zeros((0, 3), dtype=np.float32)
                vertices, faces = _build_mesh_geometry(
                    static_points,
                    voxel_size=self.config.voxel_size,
                    min_points_for_hull=self.config.min_points_for_hull,
                    bbox_padding=self.config.bbox_padding,
                )
                if vertices.shape[0] > 0 and faces.shape[0] > 0:
                    mesh_path = static_mesh_dir / f"static_frame_{frame_index:06d}.ply"
                    _write_ply_mesh(mesh_path, vertices, faces)
                    latest_static_mesh = str(mesh_path.relative_to(output_root))
                    exported_static_meshes += 1

            should_export_dynamic = frame_index % dynamic_interval_frames == 0
            if should_export_dynamic:
                for instance_id in active_instance_ids:
                    instance_points = np.concatenate(
                        [points for _, points in dynamic_buffers[instance_id]], axis=0
                    )
                    vertices, faces = _build_mesh_geometry(
                        instance_points,
                        voxel_size=self.config.voxel_size,
                        min_points_for_hull=self.config.min_points_for_hull,
                        bbox_padding=self.config.bbox_padding,
                    )
                    if vertices.shape[0] == 0 or faces.shape[0] == 0:
                        continue
                    mesh_path = (
                        dynamic_mesh_root
                        / f"instance_{instance_id}"
                        / f"dynamic_frame_{frame_index:06d}.ply"
                    )
                    _write_ply_mesh(mesh_path, vertices, faces)
                    centroid = vertices.mean(axis=0)
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, 3] = centroid
                    latest_dynamic_meshes[instance_id] = DynamicMeshInfo(
                        instance_id=instance_id,
                        mesh_path=str(mesh_path.relative_to(output_root)),
                        pose=pose,
                    )
                    exported_dynamic_meshes += 1

            active_dynamic_meshes = [
                latest_dynamic_meshes[instance_id]
                for instance_id in sorted(active_instance_ids)
                if instance_id in latest_dynamic_meshes
            ]
            per_frame_active_instances.append(len(active_dynamic_meshes))

            enriched_frame = SeparationFrame(
                timestamp=frame.timestamp,
                static_points_world=frame.static_points_world,
                dynamic_points_world=frame.dynamic_points_world,
                dynamic_instance_ids=frame.dynamic_instance_ids,
                dynamic_scores=frame.dynamic_scores,
                confidence=frame.confidence,
                visibility=frame.visibility,
                static_colors_rgb=frame.static_colors_rgb,
                dynamic_colors_rgb=frame.dynamic_colors_rgb,
                static_mesh_path=latest_static_mesh,
                dynamic_meshes=active_dynamic_meshes,
            )
            save_frame_npz(enriched_frame, frames_out_dir / frame_name)

        summary: Dict[str, object] = {
            "input_frame_count": len(frames),
            "frame_dt": frame_dt,
            "static_export_interval_frames": static_interval_frames,
            "dynamic_export_interval_frames": dynamic_interval_frames,
            "exported_static_meshes": exported_static_meshes,
            "exported_dynamic_meshes": exported_dynamic_meshes,
            "per_frame_active_mesh_instances": per_frame_active_instances,
            "config": asdict(self.config),
        }
        with (output_root / "mesh_summary.json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, indent=2)
        return summary

