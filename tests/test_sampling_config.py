"""Offline environment parsing checks for optional sampling temperature."""

from dataclasses import fields
import os
import unittest
from unittest.mock import patch

from reward_as_agent.config import ConfigurationError, Settings, get_settings


class SamplingConfigTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch("reward_as_agent.config.load_dotenv")
        self.load_dotenv = dotenv.start()
        self.addCleanup(dotenv.stop)

    def test_unset_temperature_uses_optional_default(self):
        settings = get_settings()
        self.assertIsNone(settings.temperature)
        self.assertIsNone(next(field.default for field in fields(Settings)
                               if field.name == "temperature"))
        self.load_dotenv.assert_called_once_with()

    def test_empty_or_whitespace_temperature_is_unspecified(self):
        for value in ("", " ", "\t\n \r"):
            with self.subTest(value=value):
                os.environ["REWARD_AS_AGENT_TEMPERATURE"] = value
                self.assertIsNone(get_settings().temperature)

    def test_finite_temperature_inclusive_boundaries_and_decimals(self):
        for value, expected in (("0", 0.0), ("2", 2.0), ("0.7", 0.7),
                                ("1.25", 1.25), (" 0.5\t", 0.5), ("1e-2", 0.01)):
            with self.subTest(value=value):
                os.environ["REWARD_AS_AGENT_TEMPERATURE"] = value
                temperature = get_settings().temperature
                self.assertEqual(temperature, expected)
                self.assertIsInstance(temperature, float)

    def test_invalid_temperature_raises_clear_configuration_error(self):
        invalid = ("NaN", "nan", "Inf", "+infinity", "-inf", "1e309", "-1e309",
                   "-0.01", "2.0001", "3", "garbage", "true", "null", "0.5extra", "1,0")
        for value in invalid:
            with self.subTest(value=value):
                os.environ["REWARD_AS_AGENT_TEMPERATURE"] = value
                with self.assertRaises(ConfigurationError) as caught:
                    get_settings()
                message = str(caught.exception)
                self.assertIn("REWARD_AS_AGENT_TEMPERATURE", message)
                self.assertIn("finite number in [0, 2]", message)
                self.assertIn(repr(value), message)

    def test_other_configuration_defaults_and_overrides_are_preserved(self):
        defaults = get_settings()
        self.assertEqual(defaults.provider, "doubao")
        self.assertEqual(defaults.model, "ep-20260909144216-fgpxr")
        self.assertEqual(defaults.api_base, "https://ark.cn-beijing.volces.com/api/v3")
        self.assertEqual(defaults.port, 7024)
        self.assertEqual(defaults.llm_timeout, 600)
        self.assertTrue(defaults.save_inputs)
        with patch.dict(os.environ, {
            "REWARD_AS_AGENT_TEMPERATURE": "0",
            "REWARD_AS_AGENT_PROVIDER": "doubao",
            "REWARD_AS_AGENT_PORT": "8123",
            "REWARD_AS_AGENT_LLM_TIMEOUT": "12.5",
            "REWARD_AS_AGENT_SAVE_INPUTS": "false",
        }):
            settings = get_settings()
        self.assertEqual(settings.temperature, 0.0)
        self.assertEqual(settings.provider, "doubao")
        self.assertEqual(settings.port, 8123)
        self.assertEqual(settings.llm_timeout, 12.5)
        self.assertFalse(settings.save_inputs)


if __name__ == "__main__":
    unittest.main()
