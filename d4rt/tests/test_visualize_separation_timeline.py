import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from d4rt.separation.io_contract import SeparationFrame, save_frame_npz


class TestVisualizeSeparationTimeline(unittest.TestCase):
    @staticmethod
    def _load_script_module():
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "visualize_separation_timeline.py"
        spec = importlib.util.spec_from_file_location("visualize_separation_timeline", script_path)
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

    def test_prepare_view_state_supports_window_scrubbing(self):
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
                dynamic_instance_ids=np.array([3], dtype=np.int64),
            )
            self._write_frame(
                frames_dir / "frame_000001.npz",
                timestamp=1.0,
                static_points=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.1, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([3, 8], dtype=np.int64),
            )
            self._write_frame(
                frames_dir / "frame_000002.npz",
                timestamp=2.0,
                static_points=np.array([[0.2, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.2, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([3], dtype=np.int64),
            )

            frame_files = module._resolve_frame_files(frames_dir)
            selected_files = module._select_frame_files(frame_files, 0, -1)
            frames = module._load_frames(selected_files, 0)

            state = module._prepare_view_state(
                frames=frames,
                offset=2,
                static_mode="upto",
                dynamic_mode="window",
                dynamic_window=2,
                color_mode="semantic",
                voxel_size=0.0,
                max_static_points=100,
                max_dynamic_points=100,
                rng=np.random.default_rng(0),
            )

            self.assertEqual(state["frame_index"], 2)
            self.assertEqual(state["timestamp"], 2.0)
            self.assertEqual(state["static_points"].shape[0], 3)
            self.assertEqual(state["dynamic_points"].shape[0], 3)
            self.assertEqual(sorted(np.unique(state["dynamic_ids"]).tolist()), [3, 8])
            self.assertEqual(state["dynamic_colors"].shape, (3, 3))

    def test_prepare_view_state_rgb_fallback_when_colors_missing(self):
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
                dynamic_instance_ids=np.array([4], dtype=np.int64),
                static_colors_rgb=np.array([[10, 11, 12]], dtype=np.uint8),
                dynamic_colors_rgb=np.array([[20, 21, 22]], dtype=np.uint8),
            )
            self._write_frame(
                frames_dir / "frame_000001.npz",
                timestamp=1.0,
                static_points=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
                dynamic_points=np.array([[1.1, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([4], dtype=np.int64),
                static_colors_rgb=None,
                dynamic_colors_rgb=None,
            )

            frame_files = module._resolve_frame_files(frames_dir)
            selected_files = module._select_frame_files(frame_files, 0, -1)
            frames = module._load_frames(selected_files, 0)

            state = module._prepare_view_state(
                frames=frames,
                offset=1,
                static_mode="upto",
                dynamic_mode="frame",
                dynamic_window=2,
                color_mode="rgb",
                voxel_size=0.0,
                max_static_points=100,
                max_dynamic_points=100,
                rng=np.random.default_rng(0),
            )

            self.assertEqual(state["static_points"].shape[0], 2)
            self.assertEqual(state["dynamic_points"].shape[0], 1)
            np.testing.assert_array_equal(state["static_colors"][0], np.array([10, 11, 12], dtype=np.uint8))
            np.testing.assert_array_equal(state["static_colors"][1], np.array([166, 166, 166], dtype=np.uint8))
            np.testing.assert_array_equal(
                state["dynamic_colors"][0], module._instance_color(int(state["dynamic_ids"][0]))
            )


if __name__ == "__main__":
    unittest.main()
