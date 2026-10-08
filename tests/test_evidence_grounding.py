import unittest

from reward_as_agent.evidence_grounding import (
    observation_time_manifest, validate_observation_time_format,
)


class EvidenceTimeTests(unittest.TestCase):
    def test_frame_numbers_cannot_be_presented_as_seconds(self):
        for description in ('到80秒时夹爪离开', '37-38秒夹爪打开', 'at 80 seconds the hand withdraws',
                            'at 2.5 s the hand withdraws', 'after 30 milliseconds'):
            with self.subTest(description=description), self.assertRaises(ValueError):
                validate_observation_time_format({'observations': [{'description': description}]})

    def test_frame_references_and_non_time_measurements_are_allowed(self):
        validate_observation_time_format({'observations': [
            {'description': '原帧37–38夹爪打开，80帧退回。'},
            {'description': 'At source frames 37 and 80, a 30 cm ruler is visible.'},
        ]})

    def test_time_is_derived_from_actual_manifest_not_model_prose(self):
        report = {'observations': [{'id': 'E1', 'frames': [37, 80], 'description': 'The gripper opens and withdraws.'}]}
        manifest = [{'source_frame_index': 37, 'timestamp_seconds': 2.3125},
                    {'source_frame_index': 80, 'timestamp_seconds': 5.0}]
        self.assertEqual(observation_time_manifest(report, manifest), {'E1': manifest})
        with self.assertRaises(KeyError):
            observation_time_manifest(report, manifest[:1])


if __name__ == '__main__':
    unittest.main()
