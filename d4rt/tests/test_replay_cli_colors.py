import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestReplayCliColors(unittest.TestCase):
    def _make_input_npz(self, path: Path, colors: np.ndarray) -> None:
        points = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
                [[0.1, 0.0, 1.0], [1.1, 0.0, 1.0], [2.1, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        motion = np.zeros_like(points, dtype=np.float32)
        confidence = np.full((2, 3), 0.95, dtype=np.float32)
        visibility = np.full((2, 3), 0.95, dtype=np.float32)
        timestamps = np.array([0.0, 0.1], dtype=np.float64)
        np.savez_compressed(
            path,
            points_world=points,
            motion_world=motion,
            confidence=confidence,
            visibility=visibility,
            timestamps=timestamps,
            colors_rgb=colors,
        )

    def _run_replay(self, input_path: Path, output_dir: Path) -> subprocess.CompletedProcess:
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "run_separation_replay.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--input_npz",
            str(input_path),
            "--output_dir",
            str(output_dir),
            "--static_threshold",
            "-1.0",
            "--dynamic_threshold",
            "-0.5",
            "--cluster_min_samples",
            "1",
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_replay_writes_dynamic_colors_rgb_uint8(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            input_path = tmp_dir_path / "input_uint8.npz"
            output_dir = tmp_dir_path / "out_uint8"
            colors = np.array(
                [
                    [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
                    [[41, 42, 43], [51, 52, 53], [61, 62, 63]],
                ],
                dtype=np.uint8,
            )
            self._make_input_npz(input_path, colors)

            result = self._run_replay(input_path, output_dir)
            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            frame0_path = output_dir / "frames" / "frame_000000.npz"
            self.assertTrue(frame0_path.exists())
            with np.load(frame0_path, allow_pickle=False) as frame0:
                dynamic_points = frame0["dynamic_points_world"]
                dynamic_colors = frame0["dynamic_colors_rgb"]
                static_colors = frame0["static_colors_rgb"]
                self.assertEqual(dynamic_points.shape[0], 3)
                self.assertEqual(dynamic_colors.shape, (3, 3))
                self.assertEqual(dynamic_colors.dtype, np.uint8)
                self.assertEqual(static_colors.shape, (0, 3))
                np.testing.assert_array_equal(dynamic_colors, colors[0])

    def test_replay_accepts_float_colors_and_converts_to_uint8(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            input_path = tmp_dir_path / "input_float.npz"
            output_dir = tmp_dir_path / "out_float"
            colors_float = np.array(
                [
                    [[0.0, 0.5, 1.0], [0.1, 0.2, 0.3], [0.9, 0.8, 0.7]],
                    [[1.0, 0.0, 0.0], [0.4, 0.6, 0.2], [0.5, 0.5, 0.5]],
                ],
                dtype=np.float32,
            )
            self._make_input_npz(input_path, colors_float)

            result = self._run_replay(input_path, output_dir)
            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")

            expected = np.clip(np.round(colors_float[0] * 255.0), 0, 255).astype(np.uint8)
            frame0_path = output_dir / "frames" / "frame_000000.npz"
            with np.load(frame0_path, allow_pickle=False) as frame0:
                np.testing.assert_array_equal(frame0["dynamic_colors_rgb"], expected)


if __name__ == "__main__":
    unittest.main()
