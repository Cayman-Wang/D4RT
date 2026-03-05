import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestReplayCliDryMode(unittest.TestCase):
    def _make_input_npz(self, path: Path) -> None:
        points = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 0.1, 1.0], [1.1, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        motion = np.zeros_like(points)
        confidence = np.full((2, 2), 0.95, dtype=np.float32)
        visibility = np.full((2, 2), 0.95, dtype=np.float32)
        timestamps = np.array([0.0, 0.1], dtype=np.float64)
        np.savez(
            path,
            points_world=points,
            motion_world=motion,
            confidence=confidence,
            visibility=visibility,
            timestamps=timestamps,
        )

    def test_dry_run_does_not_write_frames(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "run_separation_replay.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            input_path = tmp_dir_path / "input.npz"
            output_dir = tmp_dir_path / "out"
            self._make_input_npz(input_path)

            cmd = [
                sys.executable,
                str(script_path),
                "--input_npz",
                str(input_path),
                "--output_dir",
                str(output_dir),
                "--dry_run",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
            self.assertIn("dry run: True", result.stdout)
            self.assertFalse((output_dir / "frames").exists())
            self.assertFalse((output_dir / "summary.json").exists())

    def test_dry_run_save_json_writes_only_summary(self):
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "run_separation_replay.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            input_path = tmp_dir_path / "input.npz"
            output_dir = tmp_dir_path / "out"
            self._make_input_npz(input_path)

            cmd = [
                sys.executable,
                str(script_path),
                "--input_npz",
                str(input_path),
                "--output_dir",
                str(output_dir),
                "--dry_run",
                "--save_json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertFalse((output_dir / "frames").exists())


if __name__ == "__main__":
    unittest.main()
