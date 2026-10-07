import unittest
from unittest import mock
from app.ut_practice import ConfigFileManager

# =====================================================================
# BLOCK 4: External System Simulation (File I/O)
# ---------------------------------------------------------------------
# Testing Guidance:
# - CRITICAL: Do NOT create physical files on the disk during unit testing.
# - Utilize 'unittest.mock.mock_open' to fake file contents directly in memory.
# - Patch the built-in 'open' function and assert behavior for missing or corrupted entries.
# =====================================================================

class TestConfigFileManager(unittest.TestCase):

    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data="dev=dev.example.com\nport=8080") #  positive scenario of file parsing
    def test_load_config_basic(self, mock_file_open):
        manager = ConfigFileManager("app/tests/block4.txt")
        result = manager.load_config()
        assert result == {"dev":"dev.example.com", "port":"8080"}

    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data=" dev=dev.example.com\n port=8080 ") #  verifying strip method
    def test_verifying_strip_for_load_config(self, mock_file_open):
        manager = ConfigFileManager("app/tests/block4.txt")
        result = manager.load_config()
        assert result == {"dev":"dev.example.com", "port":"8080"}

    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data="dev=dev.example.com # comment\nport=8080") #  parsing of file with comments
    def test_load_config_with_comment(self, mock_file_open):
        manager = ConfigFileManager("app/tests/block4.txt")
        result = manager.load_config()
        assert result == {"dev":"dev.example.com # comment", "port":"8080"}

    @mock.patch("builtins.open",
                new_callable=mock.mock_open,
                read_data="dev=dev.example.com # comment \nport=8080 \napi_key='' # intentionally leave blank"
                ) #  parsing of file with empty value for api_key
    def test_load_config_with_empty_value_for_api_key(self, mock_file_open):
        manager = ConfigFileManager("app/tests/block4.txt")
        result = manager.load_config()
        assert result == {"dev":"dev.example.com # comment", "port":"8080", "api_key":"'' # intentionally leave blank"}

    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data="dev=dev.example.com\nport: 8080") # no = for 2nd line
    def test_value_error_raised_for_load_config_parsing(self, mock_file_open):
        manager = ConfigFileManager("app/tests/block4.txt")
        with self.assertRaises(ValueError) as actual_message:
            manager.load_config()

        expected_message = "Corrupted config entry at line 2: missing '=' sign."
        self.assertEqual(str(actual_message.exception), expected_message)

    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data="this is a simple string inside") # string instead of key=value
    def test_value_error_raised_for_file_with_string(self, mock_file_open):
        manager = ConfigFileManager("app/tests/block4.txt")
        with self.assertRaises(ValueError) as actual_message:
            manager.load_config()

        expected_message = "Corrupted config entry at line 1: missing '=' sign."
        self.assertEqual(str(actual_message.exception), expected_message)

    @mock.patch("builtins.open", side_effect=FileNotFoundError)  # for raising FileNotFoundError
    def test_file_not_found(self, mock_file_open):
        manager = ConfigFileManager("app/tests/some_file.txt")
        with self.assertRaises(FileNotFoundError):
            manager.load_config()

