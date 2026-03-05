import unittest

import numpy as np

from d4rt.separation.motion_score import MotionScoreCalculator, MotionScoreConfig


class TestMotionScoreHysteresisStateful(unittest.TestCase):
    def test_middle_band_keeps_previous_state(self):
        scorer = MotionScoreCalculator(
            MotionScoreConfig(
                dispersion_scale=0.1,
                residual_scale=1.0,
                voxel_size=10.0,
                history_window=4,
                static_threshold=0.35,
                dynamic_threshold=0.55,
            )
        )

        point_ids = np.array([42], dtype=np.int64)
        confidence = np.array([0.99], dtype=np.float32)
        visibility = np.array([0.99], dtype=np.float32)

        result1 = scorer.update(
            points_world=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            motion_world=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            confidence=confidence,
            visibility=visibility,
            point_ids=point_ids,
        )
        self.assertTrue(bool(result1.static_mask[0]))

        result2 = scorer.update(
            points_world=np.array([[0.2, 0.0, 1.0]], dtype=np.float32),
            motion_world=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            confidence=confidence,
            visibility=visibility,
            point_ids=point_ids,
        )
        self.assertGreater(result2.scores[0], 0.35)
        self.assertLess(result2.scores[0], 0.55)
        self.assertTrue(bool(result2.static_mask[0]))

        result3 = scorer.update(
            points_world=np.array([[0.5, 0.0, 1.0]], dtype=np.float32),
            motion_world=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            confidence=confidence,
            visibility=visibility,
            point_ids=point_ids,
        )
        self.assertGreaterEqual(result3.scores[0], 0.55)
        self.assertTrue(bool(result3.dynamic_mask[0]))

        result4 = scorer.update(
            points_world=np.array([[0.7, 0.0, 1.0]], dtype=np.float32),
            motion_world=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            confidence=confidence,
            visibility=visibility,
            point_ids=point_ids,
        )
        self.assertGreater(result4.scores[0], 0.35)
        self.assertLess(result4.scores[0], 0.55)
        self.assertTrue(bool(result4.dynamic_mask[0]))


if __name__ == "__main__":
    unittest.main()
