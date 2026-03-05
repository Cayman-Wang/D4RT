"""Instance-level clustering and temporal association for dynamic points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None


def _voxelize(points: np.ndarray, voxel_size: float) -> Set[Tuple[int, int, int]]:
    if points.size == 0:
        return set()
    quantized = np.floor(points / np.float32(voxel_size)).astype(np.int32)
    return {tuple(cell) for cell in quantized}


def _voxel_iou(a: Set[Tuple[int, int, int]], b: Set[Tuple[int, int, int]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return float(inter) / float(max(union, 1))


def dbscan_cluster(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Simple NumPy DBSCAN implementation to avoid extra runtime deps."""

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}.")
    if eps <= 0:
        raise ValueError("eps must be > 0.")
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1.")

    n_points = pts.shape[0]
    if n_points == 0:
        return np.zeros((0,), dtype=np.int64)

    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    neighbor_mask = dist_sq <= np.float32(eps * eps)

    unvisited = np.int64(-99)
    noise = np.int64(-1)
    labels = np.full(n_points, unvisited, dtype=np.int64)
    cluster_id = 0

    for point_idx in range(n_points):
        if labels[point_idx] != unvisited:
            continue

        neighbors = np.where(neighbor_mask[point_idx])[0]
        if neighbors.size < min_samples:
            labels[point_idx] = noise
            continue

        labels[point_idx] = cluster_id
        seed_queue: List[int] = [int(x) for x in neighbors if int(x) != point_idx]

        while seed_queue:
            current = seed_queue.pop()
            if labels[current] == noise:
                labels[current] = cluster_id
            if labels[current] != unvisited:
                continue

            labels[current] = cluster_id
            current_neighbors = np.where(neighbor_mask[current])[0]
            if current_neighbors.size >= min_samples:
                for cand in current_neighbors:
                    cand_int = int(cand)
                    if labels[cand_int] in (unvisited, noise):
                        seed_queue.append(cand_int)

        cluster_id += 1

    return labels


def _hungarian_or_greedy(cost_matrix: np.ndarray) -> List[Tuple[int, int]]:
    if cost_matrix.size == 0:
        return []

    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost_matrix)
        return list(zip(rows.tolist(), cols.tolist()))

    # Fallback when scipy is unavailable.
    matches: List[Tuple[int, int]] = []
    used_rows: Set[int] = set()
    used_cols: Set[int] = set()

    flat_indices = np.argsort(cost_matrix, axis=None)
    n_rows, n_cols = cost_matrix.shape
    for flat_idx in flat_indices:
        row = int(flat_idx // n_cols)
        col = int(flat_idx % n_cols)
        if row in used_rows or col in used_cols:
            continue
        matches.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
        if len(used_rows) == n_rows or len(used_cols) == n_cols:
            break
    return matches


@dataclass
class TrackerConfig:
    """Hyper-parameters for dynamic instance clustering/tracking."""

    eps: float = 0.25
    min_samples: int = 30

    center_weight: float = 0.6
    iou_weight: float = 0.4
    max_center_distance: float = 1.0
    match_cost_threshold: float = 0.75
    iou_voxel_size: float = 0.15

    max_missed_frames: int = 5

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")
        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1.")
        if self.max_center_distance <= 0:
            raise ValueError("max_center_distance must be > 0.")
        if self.match_cost_threshold < 0:
            raise ValueError("match_cost_threshold must be >= 0.")
        if self.iou_voxel_size <= 0:
            raise ValueError("iou_voxel_size must be > 0.")
        if self.max_missed_frames < 0:
            raise ValueError("max_missed_frames must be >= 0.")


@dataclass
class _Detection:
    cluster_id: int
    point_indices: np.ndarray
    points: np.ndarray
    centroid: np.ndarray
    voxel_set: Set[Tuple[int, int, int]]
    mean_score: float


@dataclass
class _TrackState:
    track_id: int
    centroid: np.ndarray
    points: np.ndarray
    voxel_set: Set[Tuple[int, int, int]]
    last_timestamp: float
    missed_frames: int = 0
    age: int = 1


@dataclass
class TrackingResult:
    """Tracking result for one frame."""

    instance_ids: np.ndarray
    cluster_labels: np.ndarray
    active_track_ids: np.ndarray


class DynamicInstanceTracker:
    """Cluster dynamic points and keep consistent instance IDs over time."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self._tracks: Dict[int, _TrackState] = {}
        self._next_track_id = 0

    @property
    def active_tracks(self) -> Dict[int, _TrackState]:
        return self._tracks

    def reset(self) -> None:
        self._tracks = {}
        self._next_track_id = 0

    def _build_detections(
        self,
        points_world: np.ndarray,
        dynamic_scores: Optional[np.ndarray],
        cluster_labels: np.ndarray,
    ) -> List[_Detection]:
        detections: List[_Detection] = []
        unique_clusters = sorted([int(x) for x in np.unique(cluster_labels) if int(x) >= 0])

        for cluster_id in unique_clusters:
            indices = np.where(cluster_labels == cluster_id)[0]
            cluster_points = points_world[indices]
            mean_score = 0.0
            if dynamic_scores is not None and dynamic_scores.size > 0:
                mean_score = float(np.mean(dynamic_scores[indices]))
            detections.append(
                _Detection(
                    cluster_id=cluster_id,
                    point_indices=indices,
                    points=cluster_points,
                    centroid=np.mean(cluster_points, axis=0),
                    voxel_set=_voxelize(cluster_points, self.config.iou_voxel_size),
                    mean_score=mean_score,
                )
            )

        return detections

    def _match(
        self,
        detections: List[_Detection],
    ) -> Dict[int, int]:
        if not self._tracks or not detections:
            return {}

        track_ids = list(self._tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        cost_matrix = np.full((num_tracks, num_dets), 1e6, dtype=np.float32)

        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            for col, det in enumerate(detections):
                center_dist = np.linalg.norm(track.centroid - det.centroid)
                if center_dist > self.config.max_center_distance:
                    continue

                center_term = center_dist / self.config.max_center_distance
                iou = _voxel_iou(track.voxel_set, det.voxel_set)
                iou_term = 1.0 - iou
                cost = self.config.center_weight * center_term + self.config.iou_weight * iou_term
                cost_matrix[row, col] = float(cost)

        proposed = _hungarian_or_greedy(cost_matrix)

        assignments: Dict[int, int] = {}
        for row, col in proposed:
            cost = float(cost_matrix[row, col])
            if not np.isfinite(cost) or cost > self.config.match_cost_threshold:
                continue
            track_id = track_ids[row]
            assignments[track_id] = col

        return assignments

    def _spawn_track(self, det: _Detection, timestamp: float) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = _TrackState(
            track_id=track_id,
            centroid=det.centroid.copy(),
            points=det.points.copy(),
            voxel_set=set(det.voxel_set),
            last_timestamp=timestamp,
            missed_frames=0,
            age=1,
        )
        return track_id

    def update(
        self,
        timestamp: float,
        points_world: np.ndarray,
        dynamic_scores: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        """
        Update tracker with one frame of dynamic points.

        Args:
            timestamp: frame timestamp.
            points_world: (N, 3) dynamic points in world frame.
            dynamic_scores: optional (N,) per-point dynamic scores.
        """

        points = np.asarray(points_world, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_world must have shape (N, 3), got {points.shape}.")

        num_points = points.shape[0]
        if dynamic_scores is None:
            scores = None
        else:
            scores = np.asarray(dynamic_scores, dtype=np.float32).reshape(-1)
            if scores.shape[0] != num_points:
                raise ValueError("dynamic_scores must have length N.")

        if num_points == 0:
            stale_track_ids = []
            for track_id, track in self._tracks.items():
                track.missed_frames += 1
                if track.missed_frames > self.config.max_missed_frames:
                    stale_track_ids.append(track_id)
            for track_id in stale_track_ids:
                self._tracks.pop(track_id, None)
            return TrackingResult(
                instance_ids=np.zeros((0,), dtype=np.int64),
                cluster_labels=np.zeros((0,), dtype=np.int64),
                active_track_ids=np.asarray(sorted(self._tracks.keys()), dtype=np.int64),
            )

        cluster_labels = dbscan_cluster(
            points,
            eps=self.config.eps,
            min_samples=self.config.min_samples,
        )
        detections = self._build_detections(points, scores, cluster_labels)

        assignments = self._match(detections)
        matched_track_ids = set(assignments.keys())
        matched_det_ids = set(assignments.values())

        det_to_track_id: Dict[int, int] = {}
        for track_id, det_idx in assignments.items():
            det = detections[det_idx]
            track = self._tracks[track_id]
            track.centroid = det.centroid.copy()
            track.points = det.points.copy()
            track.voxel_set = set(det.voxel_set)
            track.last_timestamp = float(timestamp)
            track.missed_frames = 0
            track.age += 1
            det_to_track_id[det_idx] = track_id

        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_ids:
                continue
            new_id = self._spawn_track(det, float(timestamp))
            det_to_track_id[det_idx] = new_id

        stale_track_ids = []
        for track_id, track in self._tracks.items():
            if track_id in matched_track_ids:
                continue
            if track_id in det_to_track_id.values():
                continue
            track.missed_frames += 1
            if track.missed_frames > self.config.max_missed_frames:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            self._tracks.pop(track_id, None)

        instance_ids = np.full(num_points, -1, dtype=np.int64)
        for det_idx, track_id in det_to_track_id.items():
            det = detections[det_idx]
            instance_ids[det.point_indices] = track_id

        return TrackingResult(
            instance_ids=instance_ids,
            cluster_labels=cluster_labels.astype(np.int64),
            active_track_ids=np.asarray(sorted(self._tracks.keys()), dtype=np.int64),
        )
