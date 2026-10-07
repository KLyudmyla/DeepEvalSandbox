import re
import requests
from datetime import datetime, timedelta
from typing import Any


# =====================================================================
# Custom Exceptions for Corporate Business Logic
# =====================================================================

class ValidationError(Exception):
    """Raised when user input fails validation rules."""
    pass


class InvalidLogFormatException(Exception):
    """Raised when a log line does not match the expected server format."""
    pass


class SubscriptionExpiredError(Exception):
    """Raised when an operation is attempted on an expired subscription."""
    pass


class FileUploadValidationError(Exception):
    """Raised when a batch of uploaded files violates size, count, or format rules."""
    pass


class ThirdPartyServiceError(Exception):
    """Raised when an external API integration fails or returns an unhealthy status."""
    pass


# =====================================================================
# BLOCK 0: Entry-Level Logic (Basic Arithmetic & Text Insertion)
# ---------------------------------------------------------------------
# Testing Guidance:
# - Warm-up stage! Write the simplest, plain unit tests possible.
# - Do NOT use mocks, parametrization, or complex fixtures here.
# - Just call the function with basic inputs and use standard standard 'assert'.
# =====================================================================

class BasicLLMCalculator:
    """
    A foundational utility class to handle primitive AI token and billing calculations.
    """

    def calculate_prompt_cost(self, tokens_count: int, price_per_1k_tokens: float) -> float:
        """
        Calculates the financial cost for a given number of text tokens.
        Formula: (tokens_count / 1000) * price_per_1k_tokens

        :param tokens_count: Number of tokens used in the prompt (must be positive).
        :param price_per_1k_tokens: Cost in USD per 1,000 tokens.
        :return: Total cost as a float, rounded to 5 decimal places.
        """
        if tokens_count < 0 or price_per_1k_tokens < 0:
            raise ValueError("Tokens count and price cannot be negative.")

        cost = (tokens_count / 1000) * price_per_1k_tokens
        return round(cost, 5)

    def generate_welcome_system_prompt(self, agent_name: str) -> str:
        """
        Generates a standard system prompt greeting for a new AI agent.

        :param agent_name: The commercial name of the AI assistant.
        :return: A formatted system prompt string.
        """
        cleaned_name = agent_name.strip()
        if not cleaned_name:
            return "You are a helpful AI assistant."

        return f"You are {cleaned_name}, a helpful AI assistant."


# =====================================================================
# BLOCK 1: Data Validation & String Processing
# ---------------------------------------------------------------------
# Testing Guidance:
# - Implement both positive and negative test scenarios.
# - Use 'pytest.mark.parametrize' to efficiently cover multiple boundary values.
# - Validate specific error messages caught by 'pytest.raises(ValidationError)'.
# =====================================================================

def validate_user_registration(username: str, email: str, age: int) -> bool:
    """
    Validates user registration data based on corporate business rules.

    Rules:
    - Username must be alphanumeric and between 3 and 15 characters long.
    - Email must contain exactly one '@' and at least one '.' after '@'.
    - Age must be between 18 and 99 (inclusive).

    :param username: The username string to check.
    :param email: The user email address.
    :param age: The age of the user as an integer.
    :return: True if validation passes.
    :raises ValidationError: If any of the business rules are violated.
    """
    if not username or not (3 <= len(username) <= 15) or not username.isalnum():
        raise ValidationError("Username must be alphanumeric and between 3 and 15 characters.")

    if not email or "@" not in email or email.count("@") != 1:
        raise ValidationError("Invalid email format: missing or multiple '@'.")

    domain_part = email.split("@")[1]
    if "." not in domain_part or domain_part.startswith(".") or domain_part.endswith("."):
        raise ValidationError("Invalid email format: domain must contain a valid dot structure.")

    if not isinstance(age, int) or not (18 <= age <= 99):
        raise ValidationError("User must be between 18 and 99 years old.")

    return True


# =====================================================================
# BLOCK 2: Text Parsing & Log Analysis
# ---------------------------------------------------------------------
# Testing Guidance:
# - Test structural integrity by supplying corrupted log strings.
# - Verify error handling behavior using 'pytest.raises' for empty arrays and ValueError conditions.
# - Ensure regular expressions are tested against multiple edge cases (e.g., adjacent error codes).
# =====================================================================

class LogParser:
    """
    A utility class to parse server logs and extract severity levels and HTTP error codes.
    """

    def __init__(self, raw_logs: list[str]):
        """
        Initializes the parser with a list of raw log lines.
        """
        self.logs = raw_logs

    def get_logs_by_level(self, level: str) -> list[str]:
        """
        Filters logs by severity level (e.g., 'INFO', 'WARNING', 'ERROR').
        Expects strict log format: 'YYYY-MM-DD HH:MM:SS [LEVEL] Message text'

        :param level: The severity level string to filter by.
        :return: A list of log lines that match the given severity level.
        :raises InvalidLogFormatException: If any log line doesn't match the required structure.
        """
        upper_level = level.upper()
        pattern = f"\\[{upper_level}\\]"

        filtered_logs = []
        for line in self.logs:
            if not re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[.*?\]", line):
                raise InvalidLogFormatException(f"Log line structure is corrupted: '{line}'")

            if re.search(pattern, line):
                filtered_logs.append(line)

        return filtered_logs

    def extract_error_codes(self) -> list[int]:
        """
        Finds and extracts all HTTP error status codes (4xx and 5xx) inside log messages.

        :return: A list of integers representing the extracted HTTP error codes.
        :raises ValueError: If the logs list is empty.
        """
        if not self.logs:
            raise ValueError("Cannot extract error codes from an empty log history.")

        error_codes = []
        for line in self.logs:
            matches = re.findall(r'\b[45]\d{2}\b', line)
            for match in matches:
                error_codes.append(int(match))
        return error_codes


# =====================================================================
# BLOCK 3: Date, Time & Expiration Logic
# ---------------------------------------------------------------------
# Testing Guidance:
# - CRITICAL: Do NOT use dynamic time like 'datetime.now()' in tests.
# - Inject hardcoded 'current_time' values to make assertions entirely deterministic.
# - Test critical threshold transitions: exactly active, exactly inside grace period, and fully expired.
# =====================================================================

class SubscriptionManager:
    """
    Handles user subscription statuses, expiration alerts, and grace periods.
    """

    def __init__(self, user_id: str, plan_type: str, expires_at: datetime):
        """
        Initializes the manager for a specific user subscription.
        """
        self.user_id = user_id
        self.plan_type = plan_type
        self.expires_at = expires_at

    def verify_access(self, current_time: datetime) -> str:
        """
        Verifies if the user has access to the platform based on current time.

        :param current_time: Naive datetime object representing the point of evaluation.
        :return: A string status ('active' or 'grace_period').
        :raises SubscriptionExpiredError: If the subscription is fully expired (past the 3-day grace period).
        """
        if current_time < self.expires_at:
            return "active"

        grace_end = self.expires_at + timedelta(days=3)
        if self.expires_at <= current_time <= grace_end:
            return "grace_period"

        raise SubscriptionExpiredError(f"Access denied. User subscription expired on {self.expires_at}.")


# =====================================================================
# BLOCK 4: External System Simulation (File I/O)
# ---------------------------------------------------------------------
# Testing Guidance:
# - CRITICAL: Do NOT create physical files on the disk during unit testing.
# - Utilize 'unittest.mock.mock_open' to fake file contents directly in memory.
# - Patch the built-in 'open' function and assert behavior for missing or corrupted entries.
# =====================================================================

class ConfigFileManager:
    """
    Manages system configuration loading and parsing from local text files.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_config(self) -> dict[str, str]:
        """
        Reads a configuration file line by line.

        :return: A dictionary containing the parsed configuration key-value pairs.
        :raises FileNotFoundError: If the configuration file does not exist.
        :raises ValueError: If a non-comment line is corrupted and doesn't contain '='.
        """
        config_data = {}

        with open(self.file_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    raise ValueError(f"Corrupted config entry at line {line_num}: missing '=' sign.")

                key, value = line.split("=", 1)
                config_data[key.strip()] = value.strip()

        return config_data


# =====================================================================
# BLOCK 5: Advanced Business Validation (File Upload Rules)
# ---------------------------------------------------------------------
# Testing Guidance:
# - Practice structural mapping testing by passing arrays of complex dictionaries.
# - Verify cascading limits: trigger file count limit first, then validation extension, then size limits.
# - Apply 'pytest.fixture' to pass pre-configured clean state files into your test cases.
# =====================================================================

class FileUploadValidator:
    """
    Validates a batch of files uploaded by a user based on strict corporate limits.
    """

    def __init__(self, max_files: int = 3, max_total_size_mb: float = 10.0):
        self.max_files = max_files
        self.max_total_size_mb = max_total_size_mb
        self.allowed_extensions = {".pdf", ".txt", ".docx"}

    def validate_batch(self, files: list[dict[str, Any]]) -> bool:
        """
        Validates a list of file dictionaries.
        Each dictionary represents a file: {"name": "document.pdf", "size_mb": 2.5}

        Rules:
        - Total number of files must not exceed max_files.
        - Total cumulative size of all files must not exceed max_total_size_mb.
        - Every file extension must be in the allowed_extensions set.

        :param files: A list of dictionaries containing file metadata.
        :return: True if the batch is valid.
        :raises FileUploadValidationError: If any rule is broken.
        """
        if len(files) > self.max_files:
            raise FileUploadValidationError(f"Too many files. Maximum allowed is {self.max_files}.")

        total_size = 0.0
        for file_info in files:
            name = file_info.get("name", "")
            size = file_info.get("size_mb", 0.0)

            if not any(name.lower().endswith(ext) for ext in self.allowed_extensions):
                raise FileUploadValidationError(f"File '{name}' has an unallowed format.")

            total_size += size

        if total_size > self.max_total_size_mb:
            raise FileUploadValidationError(
                f"Total batch size ({total_size} MB) exceeds limit of {self.max_total_size_mb} MB.")

        return True


# =====================================================================
# BLOCK 6: Network Mocking Practice (Third-Party HTTP Requests)
# ---------------------------------------------------------------------
# Testing Guidance:
# - CRITICAL: Use 'unittest.mock.patch' to isolate the networking layer.
# - Real HTTP calls must be completely blocked during unit test execution.
# - Use 'MagicMock' to simulate HTTP 200, 404, or 500 status codes and raw JSON responses.
# - Use 'side_effect=requests.RequestException' to ensure proper handling of network timeout spikes.
# =====================================================================

class WeatherAlertService:
    """
    Fetches real-time critical weather status from a third-party global monitoring API
    to determine if field automation tools should be paused.
    """

    def __init__(self, endpoint_url: str = "https://api.weather-service.external/v1/alert"):
        self.endpoint_url = endpoint_url

    def should_pause_operations(self) -> bool:
        """
        Sends an HTTP GET request to the external server.
        Expects a JSON response like: {"status": "success", "alert_level": "RED"}

        :return: True if the alert_level is 'RED' or 'ORANGE'.
        :raises ThirdPartyServiceError: If HTTP status is not 200, or connection fails.
        """
        try:
            response = requests.get(self.endpoint_url, timeout=5)

            if response.status_code != 200:
                raise ThirdPartyServiceError(f"External service returned status code {response.status_code}")

            data = response.json()
            alert_level = data.get("alert_level", "").upper()

            return alert_level in ["RED", "ORANGE"]

        except requests.RequestException as e:
            raise ThirdPartyServiceError(f"Failed to connect to weather provider: {str(e)}")


# =====================================================================
# BLOCK 7: Asynchronous Testing Practice (Async/Await Logic)
# ---------------------------------------------------------------------
# Testing Guidance:
# - Use 'pytest-asyncio' library to run asynchronous tests.
# - Mark your test functions with '@pytest.mark.asyncio' decorator.
# - Always use the 'await' keyword when calling asynchronous methods.
# - Test both positive paths and asynchronous exception raising.
# =====================================================================
class AsyncAIAgentEvaluator:
    """
    Simulates an asynchronous evaluation pipeline for LLM outputs and RAG nodes.
    """

    async def async_send_prompt(self, prompt: str) -> dict[str, Any]:
        """
        Simulates an asynchronous HTTP request to an LLM provider.

        :param prompt: The input text prompt for the AI model.
        :return: A dictionary containing the mock response and evaluation status.
        :raises ValueError: If the prompt is empty or just spaces.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # Simulating network latency in async architecture
        # In actual tests, this would be mocked, but here we test the async wrapper flow.
        return {
            "output": f"Refined response for: {prompt.strip()}",
            "status": "completed",
            "tokens_used": len(prompt.split()) * 2
        }

    async def async_validate_token_limit(self, tokens_count: int, max_limit: int) -> bool:
        """
        Asynchronously checks if the generated response stays within context window limits.

        :param tokens_count: Number of tokens in the response.
        :param max_limit: Maximum allowed tokens for the selected model.
        :return: True if valid.
        :raises ValidationError: If the token count exceeds the maximum limit.
        """
        if tokens_count > max_limit:
            raise ValidationError(f"Token limit exceeded. Found {tokens_count}, max allowed is {max_limit}.")

        return True
