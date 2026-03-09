import tempfile
import unittest
from pathlib import Path

import numpy as np

from d4rt.separation.io_contract import (
    DynamicMeshInfo,
    SeparationFrame,
    load_frame_npz,
    save_frame_npz,
)


class TestIOContract(unittest.TestCase):
    def test_separation_frame_roundtrip_npz(self):
        frame = SeparationFrame(
            timestamp=12.5,
            static_points_world=np.array(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32
            ),
            dynamic_points_world=np.array(
                [[0.1, 0.2, 0.3], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]], dtype=np.float32
            ),
            dynamic_instance_ids=np.array([7, 7, 9], dtype=np.int64),
            dynamic_scores=np.array([0.8, 0.9, 0.75], dtype=np.float32),
            confidence=np.array([0.95, 0.90, 0.85], dtype=np.float32),
            visibility=np.array([0.99, 0.85, 0.88], dtype=np.float32),
            static_colors_rgb=np.array([[11, 12, 13], [21, 22, 23]], dtype=np.uint8),
            dynamic_colors_rgb=np.array([[31, 32, 33], [41, 42, 43], [51, 52, 53]], dtype=np.uint8),
            static_mesh_path="meshes/static_001.ply",
            dynamic_meshes=[
                DynamicMeshInfo(instance_id=7, mesh_path="meshes/dyn_007.ply", pose=np.eye(4)),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "frame_000001.npz"
            save_frame_npz(frame, out_path)
            loaded = load_frame_npz(out_path)

        self.assertAlmostEqual(loaded.timestamp, frame.timestamp)
        np.testing.assert_allclose(loaded.static_points_world, frame.static_points_world)
        np.testing.assert_allclose(loaded.dynamic_points_world, frame.dynamic_points_world)
        np.testing.assert_array_equal(loaded.dynamic_instance_ids, frame.dynamic_instance_ids)
        np.testing.assert_allclose(loaded.dynamic_scores, frame.dynamic_scores)
        np.testing.assert_allclose(loaded.confidence, frame.confidence)
        np.testing.assert_allclose(loaded.visibility, frame.visibility)
        np.testing.assert_array_equal(loaded.static_colors_rgb, frame.static_colors_rgb)
        np.testing.assert_array_equal(loaded.dynamic_colors_rgb, frame.dynamic_colors_rgb)
        self.assertEqual(loaded.static_mesh_path, frame.static_mesh_path)
        self.assertEqual(len(loaded.dynamic_meshes), 1)
        self.assertEqual(loaded.dynamic_meshes[0].instance_id, 7)

    def test_load_legacy_npz_without_colors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "frame_legacy.npz"
            np.savez_compressed(
                out_path,
                timestamp=np.asarray(0.0, dtype=np.float64),
                static_points_world=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                dynamic_points_world=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                dynamic_instance_ids=np.array([3], dtype=np.int64),
                dynamic_scores=np.array([0.7], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                visibility=np.array([0.95], dtype=np.float32),
                static_mesh_path=np.asarray("", dtype=np.str_),
                dynamic_meshes_json=np.asarray("[]", dtype=np.str_),
            )
            loaded = load_frame_npz(out_path)

        self.assertIsNone(loaded.static_colors_rgb)
        self.assertIsNone(loaded.dynamic_colors_rgb)
        self.assertEqual(loaded.static_count, 1)
        self.assertEqual(loaded.dynamic_count, 1)

    def test_to_dict_from_dict_supports_empty_point_and_color_arrays(self):
        frame = SeparationFrame(
            timestamp=0.0,
            static_points_world=np.zeros((0, 3), dtype=np.float32),
            dynamic_points_world=np.zeros((0, 3), dtype=np.float32),
            dynamic_instance_ids=np.zeros((0,), dtype=np.int64),
            dynamic_scores=np.zeros((0,), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            visibility=np.zeros((0,), dtype=np.float32),
            static_colors_rgb=np.zeros((0, 3), dtype=np.uint8),
            dynamic_colors_rgb=np.zeros((0, 3), dtype=np.uint8),
        )

        payload = frame.to_dict()
        loaded = SeparationFrame.from_dict(payload)

        self.assertEqual(loaded.static_points_world.shape, (0, 3))
        self.assertEqual(loaded.dynamic_points_world.shape, (0, 3))
        self.assertEqual(loaded.static_colors_rgb.shape, (0, 3))
        self.assertEqual(loaded.dynamic_colors_rgb.shape, (0, 3))
        self.assertEqual(loaded.static_count, 0)
        self.assertEqual(loaded.dynamic_count, 0)

    def test_separation_frame_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            SeparationFrame(
                timestamp=0.0,
                static_points_world=np.zeros((1, 3), dtype=np.float32),
                dynamic_points_world=np.zeros((2, 3), dtype=np.float32),
                dynamic_instance_ids=np.array([1], dtype=np.int64),
                dynamic_scores=np.array([0.5, 0.6], dtype=np.float32),
                confidence=np.array([0.5, 0.6], dtype=np.float32),
                visibility=np.array([0.5, 0.6], dtype=np.float32),
            )

    def test_separation_frame_rejects_color_length_mismatch(self):
        with self.assertRaises(ValueError):
            SeparationFrame(
                timestamp=0.0,
                static_points_world=np.zeros((2, 3), dtype=np.float32),
                dynamic_points_world=np.zeros((1, 3), dtype=np.float32),
                dynamic_instance_ids=np.array([1], dtype=np.int64),
                dynamic_scores=np.array([0.6], dtype=np.float32),
                confidence=np.array([0.8], dtype=np.float32),
                visibility=np.array([0.9], dtype=np.float32),
                static_colors_rgb=np.array([[1, 2, 3]], dtype=np.uint8),
            )

        with self.assertRaises(ValueError):
            SeparationFrame(
                timestamp=0.0,
                static_points_world=np.zeros((1, 3), dtype=np.float32),
                dynamic_points_world=np.zeros((2, 3), dtype=np.float32),
                dynamic_instance_ids=np.array([1, 2], dtype=np.int64),
                dynamic_scores=np.array([0.6, 0.7], dtype=np.float32),
                confidence=np.array([0.8, 0.9], dtype=np.float32),
                visibility=np.array([0.9, 0.95], dtype=np.float32),
                dynamic_colors_rgb=np.array([[1, 2, 3]], dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
