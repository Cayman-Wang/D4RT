"""Offline replay for static/dynamic point separation and instance tracking."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

# Add project root for local imports when the script is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from d4rt.separation import (  # noqa: E402
    DynamicInstanceTracker,
    MotionScoreCalculator,
    MotionScoreConfig,
    SeparationFrame,
    TrackerConfig,
    save_frame_npz,
)


def _get_array(payload: Dict[str, np.ndarray], names: Iterable[str], required: bool = True) -> np.ndarray:
    for name in names:
        if name in payload:
            return payload[name]
    if required:
        raise KeyError(f"Missing required key. Tried: {list(names)}")
    return None


def _squeeze_last_dim(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay D4RT predictions to generate static/dynamic separation frames."
    )
    parser.add_argument("--input_npz", type=str, required=True, help="Input prediction npz path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Process at most this many frames (-1 means all)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only validate inputs and compute stats without writing frame files.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Save summary JSON in dry-run mode.",
    )

    # Motion score config
    parser.add_argument("--dispersion_weight", type=float, default=0.45)
    parser.add_argument("--residual_weight", type=float, default=0.35)
    parser.add_argument("--occupancy_weight", type=float, default=0.20)
    parser.add_argument("--dispersion_scale", type=float, default=0.10)
    parser.add_argument("--residual_scale", type=float, default=0.10)
    parser.add_argument("--voxel_size", type=float, default=0.20)
    parser.add_argument("--history_window", type=int, default=8)
    parser.add_argument("--max_idle_frames", type=int, default=16)
    parser.add_argument("--confidence_threshold", type=float, default=0.60)
    parser.add_argument("--visibility_threshold", type=float, default=0.50)
    parser.add_argument("--static_threshold", type=float, default=0.35)
    parser.add_argument("--dynamic_threshold", type=float, default=0.55)

    # Tracker config
    parser.add_argument("--cluster_eps", type=float, default=0.25)
    parser.add_argument("--cluster_min_samples", type=int, default=30)
    parser.add_argument("--center_weight", type=float, default=0.6)
    parser.add_argument("--iou_weight", type=float, default=0.4)
    parser.add_argument("--max_center_distance", type=float, default=1.0)
    parser.add_argument("--match_cost_threshold", type=float, default=0.75)
    parser.add_argument("--iou_voxel_size", type=float, default=0.15)
    parser.add_argument("--max_missed_frames", type=int, default=5)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    if not args.dry_run:
        frame_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input_npz, allow_pickle=False) as payload:
        points = _get_array(payload, ["points_world", "coords_3d_world"])
        motion = _get_array(payload, ["motion_world", "motion"], required=False)
        confidence = _get_array(payload, ["confidence", "confidences"])
        visibility = _get_array(payload, ["visibility", "visibilities"])
        timestamps = _get_array(payload, ["timestamps", "timestamp"], required=False)
        point_ids = _get_array(payload, ["point_ids", "query_ids"], required=False)

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                f"Expected points shape (T, N, 3) from points_world/coords_3d_world, got {points.shape}."
            )

        num_frames, num_points = points.shape[:2]

        if motion is None:
            motion = np.zeros_like(points, dtype=np.float32)
        else:
            motion = np.asarray(motion, dtype=np.float32)
            if motion.shape != points.shape:
                raise ValueError(
                    f"motion shape must match points shape {points.shape}, got {motion.shape}."
                )

        confidence = _squeeze_last_dim(np.asarray(confidence, dtype=np.float32))
        visibility = _squeeze_last_dim(np.asarray(visibility, dtype=np.float32))

        if confidence.shape != (num_frames, num_points):
            raise ValueError(
                f"confidence must have shape (T, N), got {confidence.shape}."
            )
        if visibility.shape != (num_frames, num_points):
            raise ValueError(
                f"visibility must have shape (T, N), got {visibility.shape}."
            )

        if timestamps is None:
            timestamps = np.arange(num_frames, dtype=np.float64)
        else:
            timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if timestamps.shape[0] != num_frames:
                raise ValueError(
                    f"timestamps must have length T={num_frames}, got {timestamps.shape}."
                )
        if not np.isfinite(timestamps).all():
            raise ValueError("timestamps contain non-finite values.")
        if timestamps.shape[0] > 1 and np.any(np.diff(timestamps) < 0):
            raise ValueError("timestamps must be non-decreasing.")

        if point_ids is None:
            point_ids = np.broadcast_to(
                np.arange(num_points, dtype=np.int64)[None, :], (num_frames, num_points)
            )
        else:
            point_ids = np.asarray(point_ids, dtype=np.int64)
            if point_ids.ndim == 1 and point_ids.shape[0] == num_points:
                point_ids = np.broadcast_to(point_ids[None, :], (num_frames, num_points))
            if point_ids.shape != (num_frames, num_points):
                raise ValueError(
                    f"point_ids must have shape (T, N) or (N,), got {point_ids.shape}."
                )

    motion_config = MotionScoreConfig(
        dispersion_weight=args.dispersion_weight,
        residual_weight=args.residual_weight,
        occupancy_weight=args.occupancy_weight,
        dispersion_scale=args.dispersion_scale,
        residual_scale=args.residual_scale,
        voxel_size=args.voxel_size,
        history_window=args.history_window,
        max_idle_frames=args.max_idle_frames,
        confidence_threshold=args.confidence_threshold,
        visibility_threshold=args.visibility_threshold,
        static_threshold=args.static_threshold,
        dynamic_threshold=args.dynamic_threshold,
    )
    tracker_config = TrackerConfig(
        eps=args.cluster_eps,
        min_samples=args.cluster_min_samples,
        center_weight=args.center_weight,
        iou_weight=args.iou_weight,
        max_center_distance=args.max_center_distance,
        match_cost_threshold=args.match_cost_threshold,
        iou_voxel_size=args.iou_voxel_size,
        max_missed_frames=args.max_missed_frames,
    )

    scorer = MotionScoreCalculator(config=motion_config)
    tracker = DynamicInstanceTracker(config=tracker_config)

    frame_limit = num_frames if args.max_frames < 0 else min(num_frames, args.max_frames)
    if frame_limit <= 0:
        raise ValueError("No frames selected for processing.")

    summary: Dict[str, List[int]] = {
        "static_counts": [],
        "dynamic_counts": [],
        "uncertain_counts": [],
        "active_track_counts": [],
    }

    for frame_idx in range(frame_limit):
        frame_points = points[frame_idx]
        frame_motion = motion[frame_idx]
        frame_conf = confidence[frame_idx]
        frame_vis = visibility[frame_idx]
        frame_ids = point_ids[frame_idx]
        frame_ts = float(timestamps[frame_idx])

        score_result = scorer.update(
            points_world=frame_points,
            motion_world=frame_motion,
            confidence=frame_conf,
            visibility=frame_vis,
            point_ids=frame_ids,
        )

        static_points = frame_points[score_result.static_mask]
        dynamic_points = frame_points[score_result.dynamic_mask]
        dynamic_scores = score_result.scores[score_result.dynamic_mask]
        dynamic_confidence = frame_conf[score_result.dynamic_mask]
        dynamic_visibility = frame_vis[score_result.dynamic_mask]

        tracking_result = tracker.update(
            timestamp=frame_ts,
            points_world=dynamic_points,
            dynamic_scores=dynamic_scores,
        )

        frame = SeparationFrame(
            timestamp=frame_ts,
            static_points_world=static_points,
            dynamic_points_world=dynamic_points,
            dynamic_instance_ids=tracking_result.instance_ids,
            dynamic_scores=dynamic_scores,
            confidence=dynamic_confidence,
            visibility=dynamic_visibility,
            static_mesh_path=None,
            dynamic_meshes=[],
        )

        if not args.dry_run:
            save_frame_npz(frame, frame_dir / f"frame_{frame_idx:06d}.npz")

        summary["static_counts"].append(int(score_result.static_mask.sum()))
        summary["dynamic_counts"].append(int(score_result.dynamic_mask.sum()))
        summary["uncertain_counts"].append(int(score_result.uncertain_mask.sum()))
        summary["active_track_counts"].append(int(tracking_result.active_track_ids.shape[0]))

    aggregate = {
        "input_npz": str(args.input_npz),
        "processed_frames": frame_limit,
        "total_static_points": int(np.sum(summary["static_counts"])),
        "total_dynamic_points": int(np.sum(summary["dynamic_counts"])),
        "total_uncertain_points": int(np.sum(summary["uncertain_counts"])),
        "mean_active_tracks": float(np.mean(summary["active_track_counts"])),
        "per_frame": summary,
        "motion_score_config": vars(motion_config),
        "tracker_config": vars(tracker_config),
    }

    should_save_json = (not args.dry_run) or args.save_json
    if should_save_json:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "summary.json", "w", encoding="utf-8") as fp:
            json.dump(aggregate, fp, indent=2)

    print("Separation replay finished")
    print(f"- dry run: {args.dry_run}")
    print(f"- frames: {frame_limit}")
    print(f"- total static points: {aggregate['total_static_points']}")
    print(f"- total dynamic points: {aggregate['total_dynamic_points']}")
    print(f"- output: {output_dir}")


if __name__ == "__main__":
    main()
