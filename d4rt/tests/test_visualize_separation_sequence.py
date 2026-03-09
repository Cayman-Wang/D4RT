import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from d4rt.separation.io_contract import SeparationFrame, save_frame_npz


class TestVisualizeSeparationSequence(unittest.TestCase):
    @staticmethod
    def _load_script_module():
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "visualize_separation_sequence.py"
        spec = importlib.util.spec_from_file_location("visualize_separation_sequence", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def _write_frame(
        self,
        path: Path,
        timestamp: float,
        static_points: np.ndarray,
        dynamic_points: np.ndarray,
        dynamic_instance_ids: np.ndarray,
        static_colors_rgb: np.ndarray | None = None,
        dynamic_colors_rgb: np.ndarray | None = None,
    ) -> None:
        frame = SeparationFrame(
            timestamp=timestamp,
            static_points_world=static_points.astype(np.float32),
            dynamic_points_world=dynamic_points.astype(np.float32),
            dynamic_instance_ids=dynamic_instance_ids.astype(np.int64),
            dynamic_scores=np.full((dynamic_points.shape[0],), 0.8, dtype=np.float32),
            confidence=np.full((dynamic_points.shape[0],), 0.9, dtype=np.float32),
            visibility=np.full((dynamic_points.shape[0],), 0.95, dtype=np.float32),
            static_colors_rgb=None if static_colors_rgb is None else static_colors_rgb.astype(np.uint8),
            dynamic_colors_rgb=None if dynamic_colors_rgb is None else dynamic_colors_rgb.astype(np.uint8),
        )
        save_frame_npz(frame, path)

    def test_export_sequence_point_clouds_without_viewer(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "visualize_separation_sequence.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            frames_dir = tmp_dir_path / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            self._write_frame(
                frames_dir / "frame_000000.npz",
                timestamp=0.0,
                static_points=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.10, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([7], dtype=np.int64),
            )
            self._write_frame(
                frames_dir / "frame_000001.npz",
                timestamp=1.0,
                static_points=np.array([[0.02, 0.0, 0.0], [2.00, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.15, 0.0, 0.0], [5.00, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([7, 9], dtype=np.int64),
            )
            self._write_frame(
                frames_dir / "frame_000002.npz",
                timestamp=2.0,
                static_points=np.array([[2.05, 0.0, 0.0], [3.00, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.16, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([7], dtype=np.int64),
            )

            static_ply = tmp_dir_path / "static.ply"
            dynamic_ply = tmp_dir_path / "dynamic.ply"
            combined_ply = tmp_dir_path / "combined.ply"
            instances_dir = tmp_dir_path / "instances"
            summary_json = tmp_dir_path / "summary.json"

            cmd = [
                sys.executable,
                str(script_path),
                "--frames_dir",
                str(frames_dir),
                "--backend",
                "none",
                "--dynamic_mode",
                "all",
                "--voxel_size",
                "0.2",
                "--export_static_ply",
                str(static_ply),
                "--export_dynamic_ply",
                str(dynamic_ply),
                "--export_combined_ply",
                str(combined_ply),
                "--export_instances_dir",
                str(instances_dir),
                "--export_summary_json",
                str(summary_json),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
            self.assertTrue(static_ply.exists())
            self.assertTrue(dynamic_ply.exists())
            self.assertTrue(combined_ply.exists())
            self.assertTrue((instances_dir / "instance_0007.ply").exists())
            self.assertTrue((instances_dir / "instance_0009.ply").exists())
            self.assertTrue(summary_json.exists())

            with open(summary_json, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertEqual(payload["selected_frame_count"], 3)
            self.assertEqual(payload["static_raw_points"], 6)
            self.assertEqual(payload["dynamic_raw_points"], 4)
            self.assertEqual(payload["static_downsampled_points"], 3)
            self.assertEqual(payload["dynamic_downsampled_points"], 2)
            self.assertEqual(payload["dynamic_instance_counts"], {"7": 1, "9": 1})

    def test_rgb_mode_falls_back_to_semantic_when_missing(self):
        module = self._load_script_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            frames_dir = tmp_dir_path / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            self._write_frame(
                frames_dir / "frame_000000.npz",
                timestamp=0.0,
                static_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([5], dtype=np.int64),
                static_colors_rgb=np.array([[10, 11, 12]], dtype=np.uint8),
                dynamic_colors_rgb=np.array([[20, 21, 22]], dtype=np.uint8),
            )
            self._write_frame(
                frames_dir / "frame_000001.npz",
                timestamp=1.0,
                static_points=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.1, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([5], dtype=np.int64),
                static_colors_rgb=None,
                dynamic_colors_rgb=None,
            )

            frame_files = module._resolve_frame_files(frames_dir)
            selected_files = module._select_frame_files(frame_files, 0, -1)
            frames = module._load_frames(selected_files, 0)

            static_points, static_rgb = module._concat_static_for_color_mode(frames, "rgb")
            dynamic_points, dynamic_ids, dynamic_rgb = module._concat_dynamic_for_color_mode(frames, "rgb")

            self.assertEqual(static_points.shape[0], 2)
            self.assertEqual(dynamic_points.shape[0], 2)
            np.testing.assert_array_equal(static_rgb[0], np.array([10, 11, 12], dtype=np.uint8))
            np.testing.assert_array_equal(static_rgb[1], np.array([166, 166, 166], dtype=np.uint8))
            np.testing.assert_array_equal(dynamic_rgb[0], np.array([20, 21, 22], dtype=np.uint8))
            np.testing.assert_array_equal(dynamic_rgb[1], module._instance_color(int(dynamic_ids[1])))

    def test_rgb_mode_exports_real_rgb_in_ply(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "visualize_separation_sequence.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            frames_dir = tmp_dir_path / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            self._write_frame(
                frames_dir / "frame_000000.npz",
                timestamp=0.0,
                static_points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([2], dtype=np.int64),
                static_colors_rgb=np.array([[101, 102, 103]], dtype=np.uint8),
                dynamic_colors_rgb=np.array([[201, 202, 203]], dtype=np.uint8),
            )

            combined_ply = tmp_dir_path / "combined_rgb.ply"
            cmd = [
                sys.executable,
                str(script_path),
                "--frames_dir",
                str(frames_dir),
                "--backend",
                "none",
                "--color_mode",
                "rgb",
                "--voxel_size",
                "0",
                "--dynamic_mode",
                "latest",
                "--export_combined_ply",
                str(combined_ply),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
            self.assertTrue(combined_ply.exists())

            with open(combined_ply, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            data_start = lines.index("end_header") + 1
            colors = {
                tuple(int(v) for v in line.split()[3:6])
                for line in lines[data_start:]
            }
            self.assertIn((101, 102, 103), colors)
            self.assertIn((201, 202, 203), colors)


if __name__ == "__main__":
    unittest.main()
