import pytest
from app.ut_practice import ValidationError, AsyncAIAgentEvaluator

# =====================================================================
# BLOCK 7: Asynchronous Testing Practice (Async/Await Logic)
# ---------------------------------------------------------------------
# Testing Guidance:
# - Use 'pytest-asyncio' library to run asynchronous tests.
# - Mark your test functions with '@pytest.mark.asyncio' decorator.
# - Always use the 'await' keyword when calling asynchronous methods.
# - Test both positive paths and asynchronous exception raising.
# =====================================================================

class TestAsyncAIAgentEvaluator:

    @pytest.mark.asyncio
    async def test_async_send_prompt_basic(self):
        ai_eval=AsyncAIAgentEvaluator()
        res = await ai_eval.async_send_prompt("Who are you?")
        assert res["output"] == "Refined response for: Who are you?"
        assert res["status"] == "completed"
        assert res["tokens_used"] == 6

    @pytest.mark.asyncio
    async def test_async_send_prompt_with_number(self): # using int instead of string format for prompt
        ai_eval=AsyncAIAgentEvaluator()
        with pytest.raises(AttributeError):
            await ai_eval.async_send_prompt(222)

    @pytest.mark.asyncio
    async def test_async_send_empty_prompt(self): # sending empty prompt
        ai_eval=AsyncAIAgentEvaluator()
        with pytest.raises(ValueError, match="Prompt cannot be empty."):
            await ai_eval.async_send_prompt("")

    @pytest.mark.asyncio
    async def test_async_send_empty_prompt_with_spaces(self): # sending empty prompt with spaces
        ai_eval=AsyncAIAgentEvaluator()
        with pytest.raises(ValueError, match="Prompt cannot be empty."):
            await ai_eval.async_send_prompt(" ")

    @pytest.mark.asyncio
    async def test_async_validate_token_limit(self):
        ai_eval=AsyncAIAgentEvaluator()
        res = await ai_eval.async_validate_token_limit(tokens_count=2, max_limit=10)
        assert res is True

    @pytest.mark.asyncio
    async def test_async_validate_token_count_equal_token_limit(self):
        ai_eval=AsyncAIAgentEvaluator()
        res = await ai_eval.async_validate_token_limit(tokens_count=2, max_limit=2)
        assert res is True

    @pytest.mark.asyncio
    async def test_async_validate_exceeded_token_limit(self):
        ai_eval=AsyncAIAgentEvaluator()
        with pytest.raises(ValidationError, match="Token limit exceeded. Found 1000, max allowed is 10."):
            await ai_eval.async_validate_token_limit(tokens_count=1000, max_limit=10)
