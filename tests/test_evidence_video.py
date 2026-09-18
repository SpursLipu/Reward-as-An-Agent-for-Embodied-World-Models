import unittest
from reward_as_agent.evidence_video import EvidenceVideo, uniform_indices, refinement_indices, verification_indices


class EvidenceVideoTests(unittest.TestCase):
    def test_source_indices_and_times_are_not_sample_positions(self):
        indices = uniform_indices(81, 32)
        self.assertEqual((indices[0], indices[-1], len(indices)), (0, 80, 32))
        video = EvidenceVideo([], 20, 640, 360)
        self.assertEqual(video.manifest([0, 43, 80])[1],
                         {'source_frame_index': 43, 'timestamp_seconds': 2.15, 'view': 'full'})

    def test_short_video_does_not_repeat_frames(self):
        self.assertEqual(uniform_indices(3), [0, 1, 2])

    def test_verification_cannot_miss_an_unsampled_short_clip_transition(self):
        self.assertEqual(verification_indices({}, uniform_indices(81), 81), list(range(81)))

    def test_long_clip_verification_remains_bounded(self):
        sampled = uniform_indices(10000)
        report = {'observations': [{'id': 'E1', 'frames': sampled}],
                  'physics_assessment': {'issues': [{'evidence': ['E1']}]}}
        verified = verification_indices(report, sampled, 10000)
        self.assertTrue(set(sampled) <= set(verified))
        self.assertLessEqual(len(verified), len(sampled) + 16)

    def test_refines_disputed_evidence_without_losing_context(self):
        report = {'observations': [{'id': 'E1', 'frames': [40]}],
                  'physics_assessment': {'issues': [{'evidence': ['E1']}]},
                  'task_assessment': {'confidence': 'high', 'target_match': 'match'}}
        self.assertEqual(refinement_indices(report, [0, 40, 80], 81, 4), [0, 38, 39, 40, 41, 42, 80])


if __name__ == '__main__':
    unittest.main()
