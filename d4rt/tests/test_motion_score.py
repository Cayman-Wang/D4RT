import unittest

import numpy as np

from d4rt.separation.motion_score import (
    MotionScoreCalculator,
    MotionScoreConfig,
    classify_scores,
)


class TestMotionScore(unittest.TestCase):
    def test_classify_scores_threshold_behavior(self):
        scores = np.array([0.1, 0.4, 0.8], dtype=np.float32)
        quality = np.array([True, True, True])
        result = classify_scores(scores, quality, static_threshold=0.35, dynamic_threshold=0.55)

        self.assertListEqual(result.static_mask.tolist(), [True, False, False])
        self.assertListEqual(result.dynamic_mask.tolist(), [False, False, True])
        self.assertListEqual(result.uncertain_mask.tolist(), [False, True, False])

    def test_motion_score_calculator_static_dynamic_and_uncertain(self):
        config = MotionScoreConfig(
            dispersion_scale=0.05,
            residual_scale=0.05,
            voxel_size=0.10,
            history_window=4,
            static_threshold=0.35,
            dynamic_threshold=0.55,
        )
        scorer = MotionScoreCalculator(config=config)

        point_ids = np.array([10, 20, 30], dtype=np.int64)

        frames = [
            np.array([[0.00, 0.00, 1.0], [0.00, 0.00, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([[0.00, 0.00, 1.0], [0.25, 0.00, 1.0], [1.3, 1.0, 1.0]], dtype=np.float32),
            np.array([[0.01, 0.00, 1.0], [0.50, 0.00, 1.0], [1.6, 1.0, 1.0]], dtype=np.float32),
        ]

        motions = [
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
        ]

        confidence = np.array([0.95, 0.95, 0.20], dtype=np.float32)
        visibility = np.array([0.95, 0.95, 0.95], dtype=np.float32)

        last = None
        for frame, motion in zip(frames, motions):
            last = scorer.update(
                points_world=frame,
                motion_world=motion,
                confidence=confidence,
                visibility=visibility,
                point_ids=point_ids,
            )

        self.assertIsNotNone(last)
        self.assertTrue(bool(last.static_mask[0]))
        self.assertTrue(bool(last.dynamic_mask[1]))
        self.assertTrue(bool(last.uncertain_mask[2]))
        self.assertGreater(last.scores[1], last.scores[0])


if __name__ == "__main__":
    unittest.main()
