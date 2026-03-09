"""Export D4RT predictions to a separation replay stream NPZ."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project root for local imports when the script is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from d4rt.data.datamodule import PointOdysseyDataModule
from d4rt.models.d4rt_model import D4RTModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run D4RT inference and export a world-frame stream NPZ that can be "
            "consumed by scripts/run_separation_replay.py."
        )
    )

    # Data arguments
    parser.add_argument(
        "--test_data_path",
        type=str,
        required=True,
        help="Path to PointOdyssey dataset root.",
    )
    parser.add_argument("--test_dset", type=str, default="val", help="Test split name.")
    parser.add_argument("--num_queries", type=int, default=2048, help="Queries per clip.")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size.")
    parser.add_argument("--S", type=int, default=8, help="Frames per clip.")
    parser.add_argument("--N", type=int, default=2048, help="Trajectory budget per clip.")
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Stride values for clip sampling.",
    )
    parser.add_argument("--clip_step", type=int, default=2, help="Step size for clip sampling.")
    parser.add_argument("--quick", action="store_true", help="Quick mode (first sequence only).")
    parser.add_argument("--verbose", action="store_true", help="Verbose dataset logging.")
    parser.add_argument("--boundary_ratio", type=float, default=0.3)
    parser.add_argument("--t_tgt_eq_t_cam_ratio", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size. Use 1 for export.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument(
        "--max_clips",
        type=int,
        default=-1,
        help="Process at most this many clips (-1 means all).",
    )

    # Checkpoint/model arguments
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--patch_size", type=int, default=16, help="Encoder patch size.")
    parser.add_argument("--encoder_embed_dim", type=int, default=None)
    parser.add_argument("--encoder_depth", type=int, default=None)
    parser.add_argument("--encoder_num_heads", type=int, default=None)
    parser.add_argument("--decoder_dim", type=int, default=None)
    parser.add_argument("--decoder_num_heads", type=int, default=None)
    parser.add_argument("--decoder_num_layers", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)

    # Output/runtime arguments
    parser.add_argument("--output_npz", type=str, required=True, help="Exported stream NPZ path.")
    parser.add_argument(
        "--point_id_stride",
        type=int,
        default=10_000_000,
        help="Global point-id stride per clip.",
    )
    parser.add_argument(
        "--no_dedup",
        action="store_true",
        help="Disable per-frame deduplication by point_id.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto/cpu/cuda/cuda:0 ...",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_queries > args.N:
        raise ValueError(
            f"--num_queries ({args.num_queries}) must be <= --N ({args.N}) for PointOdyssey sampling."
        )
    if args.batch_size != 1:
        raise ValueError("--batch_size must be 1 for export_separation_stream.py.")
    if args.point_id_stride <= 0:
        raise ValueError("--point_id_stride must be > 0.")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _extract_hparams(ckpt_payload: Dict[str, Any]) -> Dict[str, Any]:
    hparams = ckpt_payload.get("hyper_parameters", {})
    if isinstance(hparams, dict):
        return hparams
    return {}


def _select_model_arg(
    cli_value: Optional[int],
    hparams: Dict[str, Any],
    key: str,
    default_value: int,
) -> int:
    if cli_value is not None:
        return int(cli_value)
    if key in hparams:
        return int(hparams[key])
    return int(default_value)


def _build_model(args: argparse.Namespace, ckpt_payload: Dict[str, Any]) -> D4RTModel:
    hparams = _extract_hparams(ckpt_payload)
    model_kwargs = {
        "img_size": int(args.img_size),
        "patch_size": int(args.patch_size),
        "encoder_embed_dim": _select_model_arg(
            args.encoder_embed_dim, hparams, "encoder_embed_dim", 1408
        ),
        "encoder_depth": _select_model_arg(args.encoder_depth, hparams, "encoder_depth", 40),
        "encoder_num_heads": _select_model_arg(
            args.encoder_num_heads, hparams, "encoder_num_heads", 16
        ),
        "decoder_dim": _select_model_arg(args.decoder_dim, hparams, "decoder_dim", 512),
        "decoder_num_heads": _select_model_arg(
            args.decoder_num_heads, hparams, "decoder_num_heads", 8
        ),
        "decoder_num_layers": _select_model_arg(
            args.decoder_num_layers, hparams, "decoder_num_layers", 8
        ),
        "max_frames": _select_model_arg(args.max_frames, hparams, "max_frames", 100),
    }
    return D4RTModel(**model_kwargs)


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]

        cleaned[new_key] = value

    return cleaned


def _load_checkpoint(
    model: D4RTModel,
    ckpt_payload: Dict[str, Any],
    ckpt_path: str,
) -> None:
    if "state_dict" in ckpt_payload and isinstance(ckpt_payload["state_dict"], dict):
        raw_state = ckpt_payload["state_dict"]
    elif "model" in ckpt_payload and isinstance(ckpt_payload["model"], dict):
        raw_state = ckpt_payload["model"]
    elif isinstance(ckpt_payload, dict):
        raw_state = ckpt_payload
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    cleaned_state = _clean_state_dict_keys(raw_state)
    model_keys = set(model.state_dict().keys())
    model_state = {k: v for k, v in cleaned_state.items() if k in model_keys}
    if not model_state:
        raise ValueError(
            "No model weights matched D4RTModel keys. "
            "Please verify checkpoint and model config."
        )

    load_info = model.load_state_dict(model_state, strict=False)
    print("Loaded checkpoint:")
    print(f"- path: {ckpt_path}")
    print(f"- matched tensors: {len(model_state)}")
    print(f"- missing keys: {len(load_info.missing_keys)}")
    print(f"- unexpected keys: {len(load_info.unexpected_keys)}")


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _invert_extrinsics(cams_t_world: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(cams_t_world)
    except np.linalg.LinAlgError:
        # Fallback in case a matrix is close to singular.
        return np.linalg.pinv(cams_t_world)


def _transform_to_world(
    coords_cam: np.ndarray,
    motion_cam: np.ndarray,
    t_cam: np.ndarray,
    cams_t_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    world_t_cam = _invert_extrinsics(cams_t_world)

    t_cam_clamped = np.clip(t_cam.astype(np.int64), 0, world_t_cam.shape[0] - 1)
    rot = world_t_cam[t_cam_clamped, :3, :3]
    trans = world_t_cam[t_cam_clamped, :3, 3]

    coords_world = np.einsum("nij,nj->ni", rot, coords_cam) + trans
    motion_world = np.einsum("nij,nj->ni", rot, motion_cam)

    return coords_world.astype(np.float32), motion_world.astype(np.float32)


def _sample_query_colors(
    video: np.ndarray,
    t_tgt: np.ndarray,
    gt_2d_tgt: Optional[np.ndarray],
    coords_uv: Optional[np.ndarray],
) -> np.ndarray:
    """
    Sample RGB colors for queries from resized model-input frames.

    Args:
        video: (S, 3, H, W) float32 tensor in [0, 1] range.
        t_tgt: (N,) target frame index per query.
        gt_2d_tgt: optional (N, 2) pixel coordinates in model input space.
        coords_uv: optional (N, 2) normalized [0, 1] coordinates as fallback.

    Returns:
        colors_rgb: (N, 3) uint8.
    """
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(f"video must have shape (S, 3, H, W), got {video.shape}.")
    num_frames, _, height, width = video.shape
    query_count = t_tgt.shape[0]

    if gt_2d_tgt is not None:
        coords_xy = np.asarray(gt_2d_tgt, dtype=np.float32)
        if coords_xy.shape != (query_count, 2):
            raise ValueError(
                f"gt_2d_tgt must have shape ({query_count}, 2), got {coords_xy.shape}."
            )
        x = coords_xy[:, 0]
        y = coords_xy[:, 1]
    else:
        if coords_uv is None:
            raise ValueError("coords_uv is required when gt_2d_tgt is not available.")
        uv = np.asarray(coords_uv, dtype=np.float32)
        if uv.shape != (query_count, 2):
            raise ValueError(
                f"coords_uv must have shape ({query_count}, 2), got {uv.shape}."
            )
        x = uv[:, 0] * float(width - 1)
        y = uv[:, 1] * float(height - 1)

    center_x = 0.5 * float(width - 1)
    center_y = 0.5 * float(height - 1)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not finite_mask.all():
        x = x.copy()
        y = y.copy()
        x[~finite_mask] = center_x
        y[~finite_mask] = center_y

    x = np.clip(x, 0.0, float(width - 1))
    y = np.clip(y, 0.0, float(height - 1))

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    wx = (x - x0.astype(np.float32)).astype(np.float32)
    wy = (y - y0.astype(np.float32)).astype(np.float32)

    t_clamped = np.clip(t_tgt.astype(np.int64), 0, num_frames - 1)

    video_clamped = np.clip(video.astype(np.float32, copy=False), 0.0, 1.0)

    c00 = video_clamped[t_clamped, :, y0, x0]
    c10 = video_clamped[t_clamped, :, y0, x1]
    c01 = video_clamped[t_clamped, :, y1, x0]
    c11 = video_clamped[t_clamped, :, y1, x1]

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy

    colors = (
        c00 * w00[:, None]
        + c10 * w10[:, None]
        + c01 * w01[:, None]
        + c11 * w11[:, None]
    )
    colors = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    return colors


def _dedup_frame_entries(
    point_ids: np.ndarray,
    points_world: np.ndarray,
    motion_world: np.ndarray,
    confidence: np.ndarray,
    visibility: np.ndarray,
    colors_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    best_index_by_id: Dict[int, int] = {}

    for idx, point_id in enumerate(point_ids.tolist()):
        point_key = int(point_id)
        prev_idx = best_index_by_id.get(point_key)
        if prev_idx is None or confidence[idx] > confidence[prev_idx]:
            best_index_by_id[point_key] = idx

    keep_indices = np.array(sorted(best_index_by_id.values()), dtype=np.int64)
    return (
        point_ids[keep_indices],
        points_world[keep_indices],
        motion_world[keep_indices],
        confidence[keep_indices],
        visibility[keep_indices],
        colors_rgb[keep_indices],
    )


def _extract_clip_frames(
    clip_index: int,
    frame_counter: int,
    batch: Dict[str, Any],
    outputs: Dict[str, torch.Tensor],
    point_id_stride: int,
    dedup: bool,
) -> List[Dict[str, Any]]:
    coords_cam = outputs["coords_3d"][0].detach().cpu().numpy().astype(np.float32)
    motion_cam = outputs["motion"][0].detach().cpu().numpy().astype(np.float32)
    confidence = outputs["confidence"][0].detach().cpu().numpy().astype(np.float32)
    visibility = outputs["visibility"][0].detach().cpu().numpy().astype(np.float32)

    if confidence.ndim == 2 and confidence.shape[-1] == 1:
        confidence = confidence[:, 0]
    if visibility.ndim == 2 and visibility.shape[-1] == 1:
        visibility = visibility[:, 0]

    t_tgt = batch["t_tgt"][0].detach().cpu().numpy().astype(np.int64)
    t_cam = batch["t_cam"][0].detach().cpu().numpy().astype(np.int64)
    cams_t_world = batch["cams_T_world"][0].detach().cpu().numpy().astype(np.float32)
    video = batch["video"][0].detach().cpu().numpy().astype(np.float32)

    if "traj_indices" in batch:
        traj_indices = batch["traj_indices"][0].detach().cpu().numpy().astype(np.int64)
    else:
        traj_indices = np.arange(coords_cam.shape[0], dtype=np.int64)

    gt_2d_tgt = batch.get("gt_2d_tgt")
    if isinstance(gt_2d_tgt, torch.Tensor):
        gt_2d_tgt = gt_2d_tgt[0].detach().cpu().numpy().astype(np.float32)
    else:
        gt_2d_tgt = None

    coords_uv = batch.get("coords_uv")
    if isinstance(coords_uv, torch.Tensor):
        coords_uv = coords_uv[0].detach().cpu().numpy().astype(np.float32)
    else:
        coords_uv = None

    query_colors = _sample_query_colors(
        video=video,
        t_tgt=t_tgt,
        gt_2d_tgt=gt_2d_tgt,
        coords_uv=coords_uv,
    )

    points_world, motion_world = _transform_to_world(
        coords_cam=coords_cam,
        motion_cam=motion_cam,
        t_cam=t_cam,
        cams_t_world=cams_t_world,
    )

    t_tgt_clamped = np.clip(t_tgt, 0, cams_t_world.shape[0] - 1)
    point_ids = clip_index * point_id_stride + traj_indices

    annotation_path = ""
    annotation_values = batch.get("annotations_path")
    if isinstance(annotation_values, list) and len(annotation_values) > 0:
        annotation_path = str(annotation_values[0])

    frame_entries: List[Dict[str, Any]] = []
    for clip_frame_idx in range(cams_t_world.shape[0]):
        frame_mask = t_tgt_clamped == clip_frame_idx

        frame_ids = point_ids[frame_mask]
        frame_points = points_world[frame_mask]
        frame_motion = motion_world[frame_mask]
        frame_conf = confidence[frame_mask]
        frame_vis = visibility[frame_mask]
        frame_colors = query_colors[frame_mask]

        if dedup and frame_ids.shape[0] > 0:
            (
                frame_ids,
                frame_points,
                frame_motion,
                frame_conf,
                frame_vis,
                frame_colors,
            ) = _dedup_frame_entries(
                point_ids=frame_ids,
                points_world=frame_points,
                motion_world=frame_motion,
                confidence=frame_conf,
                visibility=frame_vis,
                colors_rgb=frame_colors,
            )

        frame_entries.append(
            {
                "timestamp": float(frame_counter + clip_frame_idx),
                "clip_index": int(clip_index),
                "clip_frame_index": int(clip_frame_idx),
                "annotation_path": annotation_path,
                "point_ids": frame_ids,
                "points_world": frame_points,
                "motion_world": frame_motion,
                "confidence": frame_conf,
                "visibility": frame_vis,
                "colors_rgb": frame_colors,
            }
        )

    return frame_entries


def _pack_frames(frame_entries: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not frame_entries:
        raise ValueError("No frames were exported. Check dataset path or --max_clips.")

    point_counts = np.asarray(
        [entry["points_world"].shape[0] for entry in frame_entries], dtype=np.int32
    )
    max_points = int(point_counts.max(initial=0))
    if max_points <= 0:
        raise ValueError("All exported frames are empty.")

    num_frames = len(frame_entries)

    points_world = np.zeros((num_frames, max_points, 3), dtype=np.float32)
    motion_world = np.zeros((num_frames, max_points, 3), dtype=np.float32)
    confidence = np.zeros((num_frames, max_points), dtype=np.float32)
    visibility = np.zeros((num_frames, max_points), dtype=np.float32)
    colors_rgb = np.zeros((num_frames, max_points, 3), dtype=np.uint8)
    point_ids = np.zeros((num_frames, max_points), dtype=np.int64)
    valid_mask = np.zeros((num_frames, max_points), dtype=np.bool_)

    timestamps = np.zeros((num_frames,), dtype=np.float64)
    clip_indices = np.zeros((num_frames,), dtype=np.int32)
    clip_frame_indices = np.zeros((num_frames,), dtype=np.int32)

    for frame_idx, entry in enumerate(frame_entries):
        count = entry["points_world"].shape[0]

        timestamps[frame_idx] = entry["timestamp"]
        clip_indices[frame_idx] = entry["clip_index"]
        clip_frame_indices[frame_idx] = entry["clip_frame_index"]

        if count > 0:
            points_world[frame_idx, :count] = entry["points_world"]
            motion_world[frame_idx, :count] = entry["motion_world"]
            confidence[frame_idx, :count] = entry["confidence"]
            visibility[frame_idx, :count] = entry["visibility"]
            colors_rgb[frame_idx, :count] = entry["colors_rgb"]
            point_ids[frame_idx, :count] = entry["point_ids"]
            valid_mask[frame_idx, :count] = True

        # Use frame-unique negative ids for padding slots.
        if count < max_points:
            pad = max_points - count
            point_ids[frame_idx, count:] = -(
                frame_idx * max_points + np.arange(1, pad + 1, dtype=np.int64)
            )

    annotation_paths = np.asarray(
        [entry["annotation_path"] for entry in frame_entries], dtype=np.str_
    )

    return {
        "points_world": points_world,
        "motion_world": motion_world,
        "confidence": confidence,
        "visibility": visibility,
        "colors_rgb": colors_rgb,
        "point_ids": point_ids,
        "timestamps": timestamps,
        "valid_mask": valid_mask,
        "frame_point_counts": point_counts,
        "clip_indices": clip_indices,
        "clip_frame_indices": clip_frame_indices,
        "annotation_paths": annotation_paths,
    }


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    ckpt_payload = torch.load(args.ckpt, map_location="cpu")

    model = _build_model(args, ckpt_payload)
    _load_checkpoint(model, ckpt_payload, args.ckpt)
    model.to(device)
    model.eval()

    datamodule = PointOdysseyDataModule(
        dataset_location=args.test_data_path,
        train_dset=args.test_dset,
        val_dset=args.test_dset,
        use_augs=False,
        use_val=False,
        S=args.S,
        N=args.N,
        strides=args.strides,
        clip_step=args.clip_step,
        quick=args.quick,
        verbose=args.verbose,
        num_queries=args.num_queries,
        img_size=args.img_size,
        boundary_ratio=args.boundary_ratio,
        t_tgt_eq_t_cam_ratio=args.t_tgt_eq_t_cam_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    datamodule.setup("test")
    dataloader = datamodule.test_dataloader()

    frame_entries: List[Dict[str, Any]] = []
    clip_count = 0
    frame_counter = 0

    for batch_idx, batch in enumerate(dataloader):
        if args.max_clips > 0 and clip_count >= args.max_clips:
            break

        annotations = batch.get("annotations_path")
        if isinstance(annotations, list) and len(annotations) > 0 and annotations[0] == "":
            continue

        batch_device = _to_device(batch, device)
        with torch.no_grad():
            outputs = model(
                video=batch_device["video"],
                coords_uv=batch_device["coords_uv"],
                t_src=batch_device["t_src"],
                t_tgt=batch_device["t_tgt"],
                t_cam=batch_device["t_cam"],
                aspect_ratio=batch_device.get("aspect_ratio"),
                video_orig=batch_device.get("video_orig"),
            )

        clip_frames = _extract_clip_frames(
            clip_index=clip_count,
            frame_counter=frame_counter,
            batch=batch,
            outputs=outputs,
            point_id_stride=args.point_id_stride,
            dedup=not args.no_dedup,
        )
        frame_entries.extend(clip_frames)

        clip_count += 1
        frame_counter += len(clip_frames)
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {batch_idx + 1} clips...")

    packed = _pack_frames(frame_entries)

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **packed)

    total_points = int(packed["frame_point_counts"].sum())
    print("Export complete")
    print(f"- clips: {clip_count}")
    print(f"- frames: {packed['points_world'].shape[0]}")
    print(f"- max points/frame: {packed['points_world'].shape[1]}")
    print(f"- total valid points: {total_points}")
    print(f"- output: {output_path}")


if __name__ == "__main__":
    main()
