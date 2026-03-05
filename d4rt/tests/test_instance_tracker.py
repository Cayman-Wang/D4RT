import unittest

import numpy as np

from d4rt.separation.instance_tracker import (
    DynamicInstanceTracker,
    TrackerConfig,
    dbscan_cluster,
)


class TestInstanceTracker(unittest.TestCase):
    def test_dbscan_cluster_noise_and_cluster(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [2.0, 2.0, 2.0]],
            dtype=np.float32,
        )
        labels = dbscan_cluster(points, eps=0.1, min_samples=2)

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], -1)

    def test_dynamic_instance_tracker_keeps_ids_across_frames(self):
        tracker = DynamicInstanceTracker(
            TrackerConfig(
                eps=0.25,
                min_samples=2,
                max_center_distance=0.8,
                match_cost_threshold=0.9,
                iou_voxel_size=0.10,
                max_missed_frames=2,
            )
        )

        cluster_a_f1 = np.array(
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]], dtype=np.float32
        )
        cluster_b_f1 = np.array(
            [[2.0, 0.0, 0.0], [2.05, 0.0, 0.0], [2.0, 0.05, 0.0]], dtype=np.float32
        )
        noise_f1 = np.array([[10.0, 10.0, 10.0]], dtype=np.float32)
        frame1 = np.concatenate([cluster_a_f1, cluster_b_f1, noise_f1], axis=0)

        result1 = tracker.update(
            timestamp=0.0,
            points_world=frame1,
            dynamic_scores=np.ones(frame1.shape[0], dtype=np.float32),
        )

        valid_ids_1 = sorted(set(result1.instance_ids[result1.instance_ids >= 0].tolist()))
        self.assertEqual(len(valid_ids_1), 2)
        self.assertIn(-1, result1.instance_ids)

        cluster_a_f2 = cluster_a_f1 + np.array([0.05, 0.0, 0.0], dtype=np.float32)
        cluster_b_f2 = cluster_b_f1 + np.array([0.04, 0.0, 0.0], dtype=np.float32)
        noise_f2 = noise_f1 + np.array([0.5, 0.0, 0.0], dtype=np.float32)
        frame2 = np.concatenate([cluster_a_f2, cluster_b_f2, noise_f2], axis=0)

        result2 = tracker.update(
            timestamp=1.0,
            points_world=frame2,
            dynamic_scores=np.ones(frame2.shape[0], dtype=np.float32),
        )

        valid_ids_2 = sorted(set(result2.instance_ids[result2.instance_ids >= 0].tolist()))
        self.assertListEqual(valid_ids_1, valid_ids_2)
        self.assertIn(-1, result2.instance_ids)


if __name__ == "__main__":
    unittest.main()
