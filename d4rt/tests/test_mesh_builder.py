import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from d4rt.separation.io_contract import SeparationFrame, load_frame_npz, save_frame_npz
from d4rt.separation.mesh_builder import MeshBuilderConfig, SeparationMeshBuilder


class TestMeshBuilder(unittest.TestCase):
    def _make_frame(self, timestamp: float, dynamic_shift: float) -> SeparationFrame:
        static_points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        dynamic_points = np.array(
            [
                [0.2 + dynamic_shift, 0.0, 0.0],
                [0.4 + dynamic_shift, 0.2, 0.1],
                [0.3 + dynamic_shift, 0.5, 0.2],
                [0.1 + dynamic_shift, 0.3, 0.4],
            ],
            dtype=np.float32,
        )
        return SeparationFrame(
            timestamp=timestamp,
            static_points_world=static_points,
            dynamic_points_world=dynamic_points,
            dynamic_instance_ids=np.array([7, 7, 7, 7], dtype=np.int64),
            dynamic_scores=np.full((4,), 0.8, dtype=np.float32),
            confidence=np.full((4,), 0.95, dtype=np.float32),
            visibility=np.full((4,), 0.9, dtype=np.float32),
        )

    def test_build_from_frames_dir_exports_static_and_dynamic_meshes(self):
        builder = SeparationMeshBuilder(
            MeshBuilderConfig(
                static_export_interval_seconds=2.0,
                dynamic_export_interval_seconds=0.5,
                dynamic_window_frames=4,
                voxel_size=0.0,
                bbox_padding=0.01,
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            frames_dir = tmp_path / "frames_in"
            frames_dir.mkdir()
            frames = [
                self._make_frame(timestamp=0.0, dynamic_shift=0.0),
                self._make_frame(timestamp=1.0, dynamic_shift=0.1),
                self._make_frame(timestamp=2.0, dynamic_shift=0.2),
            ]
            for index, frame in enumerate(frames):
                save_frame_npz(frame, frames_dir / f"frame_{index:06d}.npz")

            summary = builder.build_from_frames_dir(frames_dir, tmp_path / "mesh_out")

            self.assertGreaterEqual(summary["exported_static_meshes"], 1)
            self.assertGreaterEqual(summary["exported_dynamic_meshes"], 2)

            mesh_summary_path = tmp_path / "mesh_out" / "mesh_summary.json"
            self.assertTrue(mesh_summary_path.exists())
            mesh_summary = json.loads(mesh_summary_path.read_text())
            self.assertEqual(mesh_summary["input_frame_count"], 3)

            frame0 = load_frame_npz(tmp_path / "mesh_out" / "frames" / "frame_000000.npz")
            frame1 = load_frame_npz(tmp_path / "mesh_out" / "frames" / "frame_000001.npz")
            frame2 = load_frame_npz(tmp_path / "mesh_out" / "frames" / "frame_000002.npz")

            self.assertIsNotNone(frame0.static_mesh_path)
            self.assertTrue((tmp_path / "mesh_out" / frame0.static_mesh_path).exists())
            self.assertEqual(len(frame1.dynamic_meshes), 1)
            self.assertEqual(frame1.dynamic_meshes[0].instance_id, 7)
            self.assertTrue((tmp_path / "mesh_out" / frame1.dynamic_meshes[0].mesh_path).exists())
            self.assertEqual(len(frame2.dynamic_meshes), 1)
            self.assertEqual(frame2.dynamic_meshes[0].instance_id, 7)

