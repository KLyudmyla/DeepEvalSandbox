import unittest
from app.ut_practice import BasicLLMCalculator


# =====================================================================
# BLOCK 0: Entry-Level Logic (Basic Arithmetic & Text Insertion)
# ---------------------------------------------------------------------
# Testing Guidance:
# - Warm-up stage! Write the simplest, plain unit tests possible.
# - Do NOT use mocks, parametrization, or complex fixtures here.
# - Just call the function with basic inputs and use standard standard 'assert'.
# =====================================================================

class TestCasesForBlock0(unittest.TestCase):
    calculator = BasicLLMCalculator()

    def test_calculate_prompt_cost_basic(self):
        usual_cost = self.calculator.calculate_prompt_cost(10000, 0.99)
        self.assertEqual(usual_cost, 9.9)

    def test_prompt_cost_result_is_float_format(self):
        result = self.calculator.calculate_prompt_cost(10000, 0.99)
        self.assertIsInstance(result, float)

    def test_prompt_cost_result_is_positive(self):
        positive_cost = self.calculator.calculate_prompt_cost(10000, 0.99)
        self.assertGreater(positive_cost, 0)

    def test_prompt_cost_result_is_rounded(self):
        result = self.calculator.calculate_prompt_cost(10000, 0.1234567)
        self.assertAlmostEqual(result, 1.23457, places=5)

    def test_calculate_prompt_cost_with_negative_tokens_count(self):
        with self.assertRaises(ValueError) as e:
            self.calculator.calculate_prompt_cost(-10000, 0.1234567)

        self.assertEqual(
            str(e.exception),
        "Tokens count and price cannot be negative.")

    def test_calculate_prompt_cost_with_zero_tokens_count(self):
        zero_result = self.calculator.calculate_prompt_cost(0, 0.1234567)
        self.assertEqual(zero_result, 0)

    def test_calculate_prompt_cost_with_zero_price_per_token(self):
        zero_result = self.calculator.calculate_prompt_cost(10000, 0)
        self.assertEqual(zero_result, 0)

    def test_calculate_prompt_cost_with_negative_price_per_token(self):
        with self.assertRaises(ValueError) as e:
            self.calculator.calculate_prompt_cost(10000, -1.23)

        self.assertEqual(
            str(e.exception),
            "Tokens count and price cannot be negative.")

    def test_generate_welcome_system_prompt_basic(self):
        welcome_msg = self.calculator.generate_welcome_system_prompt("LLM Rag")
        self.assertEqual(welcome_msg, "You are LLM Rag, a helpful AI assistant.")

    def test_generate_welcome_system_prompt_with_string_format_check(self):
        msg = self.calculator.generate_welcome_system_prompt("LLM Rag")
        self.assertIsInstance(msg, str)

    def test_generate_welcome_system_prompt_with_empty_agent_name(self):
        welcome_msg = self.calculator.generate_welcome_system_prompt("")
        self.assertEqual(welcome_msg, "You are a helpful AI assistant.")


