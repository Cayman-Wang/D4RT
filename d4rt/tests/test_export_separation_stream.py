import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


class TestExportSeparationStream(unittest.TestCase):
    @staticmethod
    def _load_script_module():
        repo_root = Path(__file__).resolve().parents[2]
        script_path = repo_root / "scripts" / "export_separation_stream.py"
        spec = importlib.util.spec_from_file_location("export_separation_stream", script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _build_video() -> np.ndarray:
        video = np.zeros((2, 3, 3, 3), dtype=np.float32)
        for t in range(2):
            for c in range(3):
                for y in range(3):
                    for x in range(3):
                        value = (t * 100 + c * 10 + y * 3 + x) / 255.0
                        video[t, c, y, x] = value
        return video

    def test_sample_query_colors_bilinear_clip_and_nan_fallback(self):
        module = self._load_script_module()
        video = self._build_video()

        t_tgt = np.array([0, 0, 1, 1], dtype=np.int64)
        gt_2d_tgt = np.array(
            [
                [1.0, 2.0],     # exact pixel
                [0.5, 0.5],     # bilinear between 4 neighbors
                [-3.0, 8.0],    # clipped to border (0, 2)
                [np.nan, np.nan],  # fallback to center (1, 1)
            ],
            dtype=np.float32,
        )

        colors = module._sample_query_colors(
            video=video,
            t_tgt=t_tgt,
            gt_2d_tgt=gt_2d_tgt,
            coords_uv=None,
        )

        expected = np.zeros((4, 3), dtype=np.uint8)
        expected[0] = np.round(video[0, :, 2, 1] * 255.0).astype(np.uint8)
        bilinear = 0.25 * (
            video[0, :, 0, 0]
            + video[0, :, 0, 1]
            + video[0, :, 1, 0]
            + video[0, :, 1, 1]
        )
        expected[1] = np.round(bilinear * 255.0).astype(np.uint8)
        expected[2] = np.round(video[1, :, 2, 0] * 255.0).astype(np.uint8)
        expected[3] = np.round(video[1, :, 1, 1] * 255.0).astype(np.uint8)

        np.testing.assert_array_equal(colors, expected)
        self.assertEqual(colors.dtype, np.uint8)

    def test_sample_query_colors_coords_uv_fallback(self):
        module = self._load_script_module()
        video = self._build_video()

        t_tgt = np.array([0, 1], dtype=np.int64)
        coords_uv = np.array(
            [
                [0.0, 0.0],  # top-left
                [1.0, 1.0],  # bottom-right
            ],
            dtype=np.float32,
        )
        colors = module._sample_query_colors(
            video=video,
            t_tgt=t_tgt,
            gt_2d_tgt=None,
            coords_uv=coords_uv,
        )

        expected = np.vstack(
            [
                np.round(video[0, :, 0, 0] * 255.0).astype(np.uint8),
                np.round(video[1, :, 2, 2] * 255.0).astype(np.uint8),
            ]
        )
        np.testing.assert_array_equal(colors, expected)

    def test_dedup_keeps_colors_aligned_with_best_confidence(self):
        module = self._load_script_module()

        point_ids = np.array([1, 2, 1, 3], dtype=np.int64)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        motion = np.zeros_like(points, dtype=np.float32)
        confidence = np.array([0.1, 0.4, 0.9, 0.2], dtype=np.float32)
        visibility = np.array([0.8, 0.8, 0.8, 0.8], dtype=np.float32)
        colors = np.array(
            [
                [10, 0, 0],
                [20, 0, 0],
                [30, 0, 0],  # id=1 highest confidence should keep this
                [40, 0, 0],
            ],
            dtype=np.uint8,
        )

        (
            dedup_ids,
            dedup_points,
            _dedup_motion,
            dedup_conf,
            _dedup_vis,
            dedup_colors,
        ) = module._dedup_frame_entries(
            point_ids=point_ids,
            points_world=points,
            motion_world=motion,
            confidence=confidence,
            visibility=visibility,
            colors_rgb=colors,
        )

        np.testing.assert_array_equal(dedup_ids, np.array([2, 1, 3], dtype=np.int64))
        np.testing.assert_allclose(dedup_points, points[[1, 2, 3]])
        np.testing.assert_allclose(dedup_conf, confidence[[1, 2, 3]])
        np.testing.assert_array_equal(dedup_colors, colors[[1, 2, 3]])

    def test_pack_frames_contains_rgb_uint8_and_alignment(self):
        module = self._load_script_module()

        frame_entries = [
            {
                "timestamp": 0.0,
                "clip_index": 0,
                "clip_frame_index": 0,
                "annotation_path": "a0",
                "point_ids": np.array([1, 2], dtype=np.int64),
                "points_world": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
                "motion_world": np.zeros((2, 3), dtype=np.float32),
                "confidence": np.array([0.9, 0.8], dtype=np.float32),
                "visibility": np.array([0.95, 0.96], dtype=np.float32),
                "colors_rgb": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
            },
            {
                "timestamp": 1.0,
                "clip_index": 0,
                "clip_frame_index": 1,
                "annotation_path": "a1",
                "point_ids": np.array([3], dtype=np.int64),
                "points_world": np.array([[2, 0, 0]], dtype=np.float32),
                "motion_world": np.zeros((1, 3), dtype=np.float32),
                "confidence": np.array([0.7], dtype=np.float32),
                "visibility": np.array([0.9], dtype=np.float32),
                "colors_rgb": np.array([[7, 8, 9]], dtype=np.uint8),
            },
        ]

        packed = module._pack_frames(frame_entries)
        self.assertEqual(packed["colors_rgb"].dtype, np.uint8)
        self.assertEqual(packed["colors_rgb"].shape, (2, 2, 3))
        np.testing.assert_array_equal(packed["colors_rgb"][0, 0], np.array([1, 2, 3], dtype=np.uint8))
        np.testing.assert_array_equal(packed["colors_rgb"][1, 0], np.array([7, 8, 9], dtype=np.uint8))
        self.assertEqual(int(packed["valid_mask"][1, 1]), 0)


if __name__ == "__main__":
    unittest.main()
