"""Point-level static/dynamic scoring utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


_STATIC_LABEL = np.int8(0)
_DYNAMIC_LABEL = np.int8(1)
_UNCERTAIN_LABEL = np.int8(2)


@dataclass
class MotionScoreConfig:
    """Hyper-parameters for point-level motion scoring."""

    dispersion_weight: float = 0.45
    residual_weight: float = 0.35
    occupancy_weight: float = 0.20

    dispersion_scale: float = 0.10
    residual_scale: float = 0.10

    voxel_size: float = 0.20
    history_window: int = 8
    max_idle_frames: int = 16

    confidence_threshold: float = 0.60
    visibility_threshold: float = 0.50

    static_threshold: float = 0.35
    dynamic_threshold: float = 0.55

    def __post_init__(self) -> None:
        if self.history_window < 2:
            raise ValueError("history_window must be >= 2.")
        if self.max_idle_frames < 1:
            raise ValueError("max_idle_frames must be >= 1.")
        if self.dispersion_scale <= 0 or self.residual_scale <= 0:
            raise ValueError("normalization scales must be > 0.")
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be > 0.")
        if self.dynamic_threshold <= self.static_threshold:
            raise ValueError("dynamic_threshold must be larger than static_threshold.")


@dataclass
class MotionScoreResult:
    """Scoring and label outputs for one frame."""

    scores: np.ndarray
    trajectory_dispersion: np.ndarray
    flow_residual: np.ndarray
    occupancy_instability: np.ndarray
    quality_mask: np.ndarray
    static_mask: np.ndarray
    dynamic_mask: np.ndarray
    uncertain_mask: np.ndarray
    labels: np.ndarray


def _normalize(values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float32) / np.float32(scale)
    normalized = np.where(np.isfinite(normalized), normalized, 1.0)
    return np.clip(normalized, 0.0, 1.0)


def classify_scores(
    scores: np.ndarray,
    quality_mask: np.ndarray,
    static_threshold: float,
    dynamic_threshold: float,
) -> MotionScoreResult:
    """Classify points into static/dynamic/uncertain using score thresholds."""

    if dynamic_threshold <= static_threshold:
        raise ValueError("dynamic_threshold must be larger than static_threshold.")

    scores = np.asarray(scores, dtype=np.float32)
    quality_mask = np.asarray(quality_mask, dtype=bool)
    if scores.ndim != 1 or quality_mask.ndim != 1 or scores.shape[0] != quality_mask.shape[0]:
        raise ValueError("scores and quality_mask must be 1D arrays with the same length.")

    static_mask = quality_mask & (scores <= static_threshold)
    dynamic_mask = quality_mask & (scores >= dynamic_threshold)
    uncertain_mask = ~(static_mask | dynamic_mask)

    labels = np.full(scores.shape[0], _UNCERTAIN_LABEL, dtype=np.int8)
    labels[static_mask] = _STATIC_LABEL
    labels[dynamic_mask] = _DYNAMIC_LABEL

    zeros = np.zeros_like(scores, dtype=np.float32)
    return MotionScoreResult(
        scores=scores,
        trajectory_dispersion=zeros,
        flow_residual=zeros,
        occupancy_instability=zeros,
        quality_mask=quality_mask,
        static_mask=static_mask,
        dynamic_mask=dynamic_mask,
        uncertain_mask=uncertain_mask,
        labels=labels,
    )


class MotionScoreCalculator:
    """Stateful calculator for online per-point static/dynamic scoring."""

    def __init__(self, config: Optional[MotionScoreConfig] = None):
        self.config = config or MotionScoreConfig()
        self._position_history: Dict[int, deque[np.ndarray]] = {}
        self._voxel_history: Dict[int, deque[np.ndarray]] = {}
        self._prev_positions: Dict[int, np.ndarray] = {}
        self._prev_labels: Dict[int, np.int8] = {}
        self._last_seen: Dict[int, int] = {}
        self._frame_idx = 0

    def reset(self) -> None:
        self._position_history.clear()
        self._voxel_history.clear()
        self._prev_positions.clear()
        self._prev_labels.clear()
        self._last_seen.clear()
        self._frame_idx = 0

    def _prune_inactive(self) -> None:
        stale_ids = [
            point_id
            for point_id, last_seen in self._last_seen.items()
            if self._frame_idx - last_seen > self.config.max_idle_frames
        ]
        for point_id in stale_ids:
            self._position_history.pop(point_id, None)
            self._voxel_history.pop(point_id, None)
            self._prev_positions.pop(point_id, None)
            self._prev_labels.pop(point_id, None)
            self._last_seen.pop(point_id, None)

    def update(
        self,
        points_world: np.ndarray,
        motion_world: np.ndarray,
        confidence: np.ndarray,
        visibility: np.ndarray,
        point_ids: Optional[np.ndarray] = None,
    ) -> MotionScoreResult:
        """
        Update scorer with one frame of points.

        Args:
            points_world: (N, 3) points in world frame.
            motion_world: (N, 3) predicted displacement in world frame.
            confidence: (N,) confidence scores.
            visibility: (N,) visibility scores.
            point_ids: optional (N,) stable IDs across frames.
        """

        points = np.asarray(points_world, dtype=np.float32)
        motion = np.asarray(motion_world, dtype=np.float32)
        conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        vis = np.asarray(visibility, dtype=np.float32).reshape(-1)

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_world must have shape (N, 3), got {points.shape}.")
        if motion.shape != points.shape:
            raise ValueError(
                f"motion_world must have shape {points.shape}, got {motion.shape}."
            )

        num_points = points.shape[0]
        if conf.shape[0] != num_points or vis.shape[0] != num_points:
            raise ValueError("confidence/visibility must have length N.")

        if point_ids is None:
            ids = np.arange(num_points, dtype=np.int64)
        else:
            ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
            if ids.shape[0] != num_points:
                raise ValueError("point_ids must have length N.")

        trajectory_dispersion_raw = np.zeros(num_points, dtype=np.float32)
        flow_residual_raw = np.zeros(num_points, dtype=np.float32)
        occupancy_instability = np.zeros(num_points, dtype=np.float32)

        for idx, point_id in enumerate(ids):
            point = points[idx]
            motion_vec = motion[idx]
            point_key = int(point_id)

            history = self._position_history.setdefault(
                point_key, deque(maxlen=self.config.history_window)
            )
            voxel_history = self._voxel_history.setdefault(
                point_key, deque(maxlen=self.config.history_window)
            )

            prev_point = self._prev_positions.get(point_key)
            if prev_point is not None:
                observed_delta = point - prev_point
                flow_residual_raw[idx] = np.linalg.norm(observed_delta - motion_vec)
            else:
                flow_residual_raw[idx] = 0.0

            candidate_points = list(history)
            candidate_points.append(point)
            if len(candidate_points) >= 2:
                candidate_stack = np.stack(candidate_points, axis=0)
                center = np.median(candidate_stack, axis=0)
                radial_distance = np.linalg.norm(candidate_stack - center, axis=1)
                trajectory_dispersion_raw[idx] = np.median(radial_distance)
            else:
                trajectory_dispersion_raw[idx] = 0.0

            voxel_index = np.floor(point / np.float32(self.config.voxel_size)).astype(np.int32)
            candidate_voxels = list(voxel_history)
            candidate_voxels.append(voxel_index)
            if len(candidate_voxels) >= 2:
                transitions = 0
                for a, b in zip(candidate_voxels[:-1], candidate_voxels[1:]):
                    if not np.array_equal(a, b):
                        transitions += 1
                occupancy_instability[idx] = transitions / float(len(candidate_voxels) - 1)
            else:
                occupancy_instability[idx] = 0.0

            history.append(point.copy())
            voxel_history.append(voxel_index)
            self._prev_positions[point_key] = point.copy()
            self._last_seen[point_key] = self._frame_idx

        trajectory_dispersion = _normalize(
            trajectory_dispersion_raw, self.config.dispersion_scale
        )
        flow_residual = _normalize(flow_residual_raw, self.config.residual_scale)

        scores = (
            self.config.dispersion_weight * trajectory_dispersion
            + self.config.residual_weight * flow_residual
            + self.config.occupancy_weight * occupancy_instability
        )
        scores = np.clip(scores.astype(np.float32), 0.0, 1.0)

        valid_points = np.isfinite(points).all(axis=1)
        quality_mask = (
            (conf >= self.config.confidence_threshold)
            & (vis >= self.config.visibility_threshold)
            & valid_points
        )

        static_mask_hard = quality_mask & (scores <= self.config.static_threshold)
        dynamic_mask_hard = quality_mask & (scores >= self.config.dynamic_threshold)
        middle_mask = quality_mask & ~(static_mask_hard | dynamic_mask_hard)

        labels = np.full(num_points, _UNCERTAIN_LABEL, dtype=np.int8)
        labels[static_mask_hard] = _STATIC_LABEL
        labels[dynamic_mask_hard] = _DYNAMIC_LABEL

        # Apply hysteresis: keep previous static/dynamic state for middle scores.
        for idx in np.where(middle_mask)[0]:
            prev_label = self._prev_labels.get(int(ids[idx]))
            if prev_label in (_STATIC_LABEL, _DYNAMIC_LABEL):
                labels[idx] = prev_label

        static_mask = labels == _STATIC_LABEL
        dynamic_mask = labels == _DYNAMIC_LABEL
        uncertain_mask = labels == _UNCERTAIN_LABEL

        for idx, point_id in enumerate(ids):
            label = labels[idx]
            if label in (_STATIC_LABEL, _DYNAMIC_LABEL):
                self._prev_labels[int(point_id)] = label

        self._frame_idx += 1
        self._prune_inactive()

        return MotionScoreResult(
            scores=scores,
            trajectory_dispersion=trajectory_dispersion,
            flow_residual=flow_residual,
            occupancy_instability=occupancy_instability,
            quality_mask=quality_mask,
            static_mask=static_mask,
            dynamic_mask=dynamic_mask,
            uncertain_mask=uncertain_mask,
            labels=labels,
        )
