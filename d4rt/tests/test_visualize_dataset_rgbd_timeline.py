import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class TestVisualizeDatasetRgbdTimeline(unittest.TestCase):
    @staticmethod
    def _load_script_module():
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "visualize_dataset_rgbd_timeline.py"
        spec = importlib.util.spec_from_file_location("visualize_dataset_rgbd_timeline", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _script_path() -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "scripts" / "visualize_dataset_rgbd_timeline.py"

    @staticmethod
    def _make_fake_dataset(
        root_dir: Path,
        frame_count: int,
        empty_depth_indices: set[int] | None = None,
    ) -> tuple[Path, str]:
        if empty_depth_indices is None:
            empty_depth_indices = set()

        dataset_root = root_dir / "dataset"
        sequence_name = "seq_demo"
        seq_dir = dataset_root / "val" / sequence_name
        rgb_dir = seq_dir / "rgbs"
        depth_dir = seq_dir / "depths"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        height, width = 6, 8
        intrinsics = np.array(
            [[40.0, 0.0, width * 0.5], [0.0, 40.0, height * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )[None, ...]
        extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], frame_count, axis=0)
        np.savez_compressed(seq_dir / "anno.npz", intrinsics=intrinsics, extrinsics=extrinsics)

        for frame_idx in range(frame_count):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0] = np.uint8((frame_idx + 1) * 20)
            rgb[:, :, 1] = np.uint8(80 + frame_idx)
            rgb[:, :, 2] = np.uint8(150 - frame_idx)
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(rgb_dir / f"rgb_{frame_idx:05d}.jpg"), rgb_bgr)

            depth_value = 0 if frame_idx in empty_depth_indices else 2000
            depth = np.full((height, width), depth_value, dtype=np.uint16)
            cv2.imwrite(str(depth_dir / f"depth_{frame_idx:05d}.png"), depth)

        return dataset_root, sequence_name

    @staticmethod
    def _read_ply_header(path: Path) -> tuple[int, list[str]]:
        lines: list[str] = []
        with open(path, "rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                decoded = line.decode("ascii", errors="ignore").strip()
                lines.append(decoded)
                if decoded == "end_header":
                    break

        vertex_count = -1
        for line in lines:
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                break
        return vertex_count, lines

    def test_selection_and_window_offsets(self):
        module = self._load_script_module()
        selected = module._select_frame_indices(
            candidate_indices=[0, 1, 2, 3, 4, 5],
            start_frame=1,
            end_frame=6,
            frame_stride=2,
            max_frames=2,
        )
        self.assertEqual(selected, [1, 3])

        offsets_window = module._dynamic_frame_offsets(
            total_frames=3,
            offset=1,
            dynamic_mode="window",
            dynamic_window=8,
        )
        self.assertEqual(offsets_window, [0, 1])

        offsets_frame = module._dynamic_frame_offsets(
            total_frames=3,
            offset=1,
            dynamic_mode="frame",
            dynamic_window=8,
        )
        self.assertEqual(offsets_frame, [1])

    def test_backend_none_exports_frames_and_summary(self):
        script_path = self._script_path()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_root, sequence_name = self._make_fake_dataset(tmp_path, frame_count=4)

            export_frames_dir = tmp_path / "exports"
            summary_json = tmp_path / "summary.json"
            cmd = [
                sys.executable,
                str(script_path),
                "--dataset_root",
                str(dataset_root),
                "--dset",
                "val",
                "--sequence",
                sequence_name,
                "--start_frame",
                "0",
                "--end_frame",
                "-1",
                "--frame_stride",
                "1",
                "--max_frames",
                "3",
                "--dynamic_mode",
                "window",
                "--dynamic_window",
                "8",
                "--backend",
                "none",
                "--export_frames_dir",
                str(export_frames_dir),
                "--export_summary_json",
                str(summary_json),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            ply_files = sorted(export_frames_dir.glob("frame_*.ply"))
            self.assertEqual(len(ply_files), 3)
            for ply_path in ply_files:
                vertex_count, header_lines = self._read_ply_header(ply_path)
                self.assertGreater(vertex_count, 0)
                self.assertIn("property uchar red", header_lines)
                self.assertIn("property uchar green", header_lines)
                self.assertIn("property uchar blue", header_lines)

            self.assertTrue(summary_json.exists())
            with open(summary_json, "r", encoding="utf-8") as handle:
                summary = json.load(handle)

            self.assertEqual(summary["selected_frame_count"], 3)
            self.assertEqual(len(summary["frames"]), 3)
            self.assertEqual(summary["initial_offset"], 2)
            self.assertEqual(summary["initial_dynamic_frame_offsets"], [0, 1, 2])
            self.assertGreater(summary["initial_dynamic_points"], 0)

    def test_backend_none_handles_empty_dynamic_frame(self):
        script_path = self._script_path()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_root, sequence_name = self._make_fake_dataset(
                tmp_path,
                frame_count=3,
                empty_depth_indices={1},
            )
            summary_json = tmp_path / "summary_empty.json"
            cmd = [
                sys.executable,
                str(script_path),
                "--dataset_root",
                str(dataset_root),
                "--dset",
                "val",
                "--sequence",
                sequence_name,
                "--start_frame",
                "0",
                "--end_frame",
                "-1",
                "--frame_stride",
                "1",
                "--max_frames",
                "3",
                "--initial_index",
                "1",
                "--dynamic_mode",
                "frame",
                "--backend",
                "none",
                "--export_summary_json",
                str(summary_json),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            with open(summary_json, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["initial_source_frame_index"], 1)
            self.assertEqual(summary["initial_dynamic_points"], 0)


if __name__ == "__main__":
    unittest.main()

