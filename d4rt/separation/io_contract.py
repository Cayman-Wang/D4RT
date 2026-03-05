"""I/O contract for static/dynamic separation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json

import numpy as np


def _as_points(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {arr.shape}.")
    return arr


def _as_1d(values: np.ndarray, name: str, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got {arr.shape}.")
    return arr


@dataclass
class DynamicMeshInfo:
    """Mesh metadata for one dynamic instance."""

    instance_id: int
    mesh_path: str
    pose: np.ndarray

    def __post_init__(self) -> None:
        self.instance_id = int(self.instance_id)
        self.mesh_path = str(self.mesh_path)
        self.pose = np.asarray(self.pose, dtype=np.float32)
        if self.pose.shape != (4, 4):
            raise ValueError(f"pose must have shape (4, 4), got {self.pose.shape}.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "mesh_path": self.mesh_path,
            "pose": self.pose.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DynamicMeshInfo":
        return cls(
            instance_id=payload["instance_id"],
            mesh_path=payload["mesh_path"],
            pose=np.asarray(payload["pose"], dtype=np.float32),
        )


@dataclass
class SeparationFrame:
    """
    Per-frame separation output shared across downstream stages.

    `static_mesh_path` and `dynamic_meshes` are optional placeholders in M2 and can
    stay empty until mesh generation is introduced in M3.
    """

    timestamp: float
    static_points_world: np.ndarray
    dynamic_points_world: np.ndarray
    dynamic_instance_ids: np.ndarray
    dynamic_scores: np.ndarray
    confidence: np.ndarray
    visibility: np.ndarray
    static_mesh_path: Optional[str] = None
    dynamic_meshes: List[DynamicMeshInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.timestamp = float(self.timestamp)
        self.static_points_world = _as_points(self.static_points_world, "static_points_world")
        self.dynamic_points_world = _as_points(self.dynamic_points_world, "dynamic_points_world")
        self.dynamic_instance_ids = _as_1d(
            self.dynamic_instance_ids, "dynamic_instance_ids", dtype=np.int64
        )
        self.dynamic_scores = _as_1d(self.dynamic_scores, "dynamic_scores", dtype=np.float32)
        self.confidence = _as_1d(self.confidence, "confidence", dtype=np.float32)
        self.visibility = _as_1d(self.visibility, "visibility", dtype=np.float32)
        self.static_mesh_path = None if self.static_mesh_path is None else str(self.static_mesh_path)

        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite.")

        dynamic_count = self.dynamic_points_world.shape[0]
        for name, values in {
            "dynamic_instance_ids": self.dynamic_instance_ids,
            "dynamic_scores": self.dynamic_scores,
            "confidence": self.confidence,
            "visibility": self.visibility,
        }.items():
            if values.shape[0] != dynamic_count:
                raise ValueError(
                    f"{name} length ({values.shape[0]}) must match dynamic point count ({dynamic_count})."
                )

        if self.dynamic_meshes is None:
            self.dynamic_meshes = []
        else:
            self.dynamic_meshes = [
                mesh if isinstance(mesh, DynamicMeshInfo) else DynamicMeshInfo.from_dict(mesh)
                for mesh in self.dynamic_meshes
            ]

    @property
    def dynamic_count(self) -> int:
        return int(self.dynamic_points_world.shape[0])

    @property
    def static_count(self) -> int:
        return int(self.static_points_world.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "static_points_world": self.static_points_world.tolist(),
            "dynamic_points_world": self.dynamic_points_world.tolist(),
            "dynamic_instance_ids": self.dynamic_instance_ids.tolist(),
            "dynamic_scores": self.dynamic_scores.tolist(),
            "confidence": self.confidence.tolist(),
            "visibility": self.visibility.tolist(),
            "static_mesh_path": self.static_mesh_path,
            "dynamic_meshes": [mesh.to_dict() for mesh in self.dynamic_meshes],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SeparationFrame":
        return cls(
            timestamp=payload["timestamp"],
            static_points_world=np.asarray(payload["static_points_world"], dtype=np.float32),
            dynamic_points_world=np.asarray(payload["dynamic_points_world"], dtype=np.float32),
            dynamic_instance_ids=np.asarray(payload["dynamic_instance_ids"], dtype=np.int64),
            dynamic_scores=np.asarray(payload["dynamic_scores"], dtype=np.float32),
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            visibility=np.asarray(payload["visibility"], dtype=np.float32),
            static_mesh_path=payload.get("static_mesh_path"),
            dynamic_meshes=[
                DynamicMeshInfo.from_dict(item)
                for item in payload.get("dynamic_meshes", [])
            ],
        )


def save_frame_npz(frame: SeparationFrame, path: str | Path) -> None:
    """Persist one separation frame as a compressed NPZ file."""

    frame = frame if isinstance(frame, SeparationFrame) else SeparationFrame.from_dict(frame)
    dynamic_meshes_json = json.dumps([mesh.to_dict() for mesh in frame.dynamic_meshes])

    np.savez_compressed(
        str(path),
        timestamp=np.asarray(frame.timestamp, dtype=np.float64),
        static_points_world=frame.static_points_world,
        dynamic_points_world=frame.dynamic_points_world,
        dynamic_instance_ids=frame.dynamic_instance_ids,
        dynamic_scores=frame.dynamic_scores,
        confidence=frame.confidence,
        visibility=frame.visibility,
        static_mesh_path=np.asarray(frame.static_mesh_path or "", dtype=np.str_),
        dynamic_meshes_json=np.asarray(dynamic_meshes_json, dtype=np.str_),
    )


def load_frame_npz(path: str | Path) -> SeparationFrame:
    """Load one separation frame from NPZ."""

    with np.load(str(path), allow_pickle=False) as payload:
        mesh_payload = payload.get("dynamic_meshes_json")
        if mesh_payload is None:
            dynamic_meshes: Sequence[Dict[str, Any]] = []
        else:
            dynamic_meshes = json.loads(str(mesh_payload.item()))

        static_mesh_path = payload.get("static_mesh_path")
        if static_mesh_path is None:
            static_mesh_value: Optional[str] = None
        else:
            static_mesh_raw = str(static_mesh_path.item())
            static_mesh_value = static_mesh_raw or None

        return SeparationFrame(
            timestamp=float(payload["timestamp"].item()),
            static_points_world=payload["static_points_world"],
            dynamic_points_world=payload["dynamic_points_world"],
            dynamic_instance_ids=payload["dynamic_instance_ids"],
            dynamic_scores=payload["dynamic_scores"],
            confidence=payload["confidence"],
            visibility=payload["visibility"],
            static_mesh_path=static_mesh_value,
            dynamic_meshes=[DynamicMeshInfo.from_dict(item) for item in dynamic_meshes],
        )
