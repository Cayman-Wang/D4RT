"""Build minimal static/dynamic meshes from replay frame outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from d4rt.separation.mesh_builder import MeshBuilderConfig, SeparationMeshBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consume run_separation_replay.py frame_*.npz outputs and build minimal "
            "static/dynamic mesh assets for M3a smoke validation."
        )
    )
    parser.add_argument("--frames_dir", required=True, help="Replay frames directory.")
    parser.add_argument("--output_dir", required=True, help="Mesh output directory.")
    parser.add_argument(
        "--static_export_interval_seconds",
        type=float,
        default=2.0,
        help="Static mesh export interval in replay-timestamp seconds.",
    )
    parser.add_argument(
        "--dynamic_export_interval_seconds",
        type=float,
        default=0.5,
        help="Dynamic mesh export interval in replay-timestamp seconds.",
    )
    parser.add_argument(
        "--dynamic_window_frames",
        type=int,
        default=8,
        help="Number of trailing frames kept for each dynamic instance.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size applied before mesh generation. <=0 disables downsampling.",
    )
    parser.add_argument(
        "--bbox_padding",
        type=float,
        default=0.02,
        help="Minimum padding used by the bbox fallback mesh builder.",
    )
    parser.add_argument(
        "--min_points_for_hull",
        type=int,
        default=4,
        help="Minimum point count before trying ConvexHull.",
    )
    parser.add_argument(
        "--exclude_untracked_dynamic",
        action="store_true",
        help="Ignore dynamic points whose replay instance id is -1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MeshBuilderConfig(
        static_export_interval_seconds=args.static_export_interval_seconds,
        dynamic_export_interval_seconds=args.dynamic_export_interval_seconds,
        dynamic_window_frames=args.dynamic_window_frames,
        voxel_size=args.voxel_size,
        min_points_for_hull=args.min_points_for_hull,
        bbox_padding=args.bbox_padding,
        include_untracked_dynamic=not args.exclude_untracked_dynamic,
    )
    builder = SeparationMeshBuilder(config=config)
    summary = builder.build_from_frames_dir(args.frames_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

