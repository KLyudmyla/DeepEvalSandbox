import pytest
from app.ut_practice import InvalidLogFormatException, LogParser

# =====================================================================
# BLOCK 2: Text Parsing & Log Analysis
# ---------------------------------------------------------------------
# Testing Guidance:
# - Test structural integrity by supplying corrupted log strings.
# - Verify error handling behavior using 'pytest.raises' for empty arrays and ValueError conditions.
# - Ensure regular expressions are tested against multiple edge cases (e.g., adjacent error codes).
# =====================================================================

# I have to create a fixture in order not to repeat test data in each function
@pytest.fixture
def my_logs():
    return [
        "2026-06-10 10:00:00 [INFO] Server started successfully: 200",
        "2026-06-18 10:05:00 [WARNING] User is not authorized: 401",
        "2026-06-18 10:10:00 [ERROR] Database connection failed",
        "2026-06-19 10:05:08 [WARNING] Bad Request: 400",
        "2026-06-19 10:11:22 [WARNING] Too Many Requests: 429",
        "2026-06-19 10:15:08 [ERROR] Internal Server Error: 500",
        "2026-06-19 11:15:08 [ERROR] Gateway Timeout: 504"
    ]

@pytest.fixture
def broken_info():
    return [
        "2026-06-1 10:00:000 [INFO] Server started successfully",
        "2026-06-18 10:05:00 [WARNING] You are using too many tokens",
        "2026-06-18 10:10:00 [ERROR] Database connection failed"
    ]

@pytest.fixture
def broken_warning():
    return [
        "2026-06-16 10:00:00 [INFO] Server started successfully",
        "2026-06-18 10:05:00 WARNING",
        "2026-06-18 10:10:00 [ERROR] Database connection failed"
    ]

class TestLogParser:

    def test_get_error_logs(self, my_logs): # positive scenario, logs are found
        res = LogParser(my_logs)
        actual = res.get_logs_by_level("ERROR")
        print(actual)
        expected = [
            "2026-06-18 10:10:00 [ERROR] Database connection failed",
            "2026-06-19 10:15:08 [ERROR] Internal Server Error: 500",
            "2026-06-19 11:15:08 [ERROR] Gateway Timeout: 504"]
        assert actual == expected

    def test_raising_exception_for_info_log(self, broken_info): # raising exception with incorrect date format
        res = LogParser(broken_info)
        with pytest.raises(InvalidLogFormatException, match=f"Log line structure is corrupted: "):
            res.get_logs_by_level("INFO")

    def test_raising_exception_for_warning(self, broken_warning): # raising exception with regular expression format
        res = LogParser(broken_warning)
        with pytest.raises(InvalidLogFormatException, match=f"Log line structure is corrupted: "):
            res.get_logs_by_level("WARNING")

    def test_empty_list_for_logs(self): # verifying empty list with errors
        res = LogParser([])
        assert res.get_logs_by_level("DEBUG") == []

    def test_appending_function_for_logs(self, my_logs): # verifying that lines are appended
        res = LogParser(my_logs)
        actual = res.get_logs_by_level("WARNING")
        assert len(actual) == 3

    def test_return_all_codes(self, my_logs):
        res = LogParser(my_logs)
        expected = [401, 400, 429, 500, 504]
        assert res.extract_error_codes() == expected

    def test_return_validation_error_for_empty_logs(self):
        res = LogParser([])
        with pytest.raises(ValueError, match="Cannot extract error codes from an empty log history."):
            res.extract_error_codes()

    @pytest.mark.parametrize("complex_logs", [
        ["2026-06-18 10:10:00 [ERROR] Error 404 occurred while recovering from 500"]
    ])

    def test_complex_logs_with_2codes_in_line(self, complex_logs): # verifying logs with 2 codes in 1 line
        res = LogParser(complex_logs)
        expected = [404, 500]
        assert res.extract_error_codes() == expected
