"""Small compatibility boundary for the optional DeepEval runtime dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig, ErrorConfig
    from deepeval.metrics import BaseMetric, GEval
    from deepeval.metrics.g_eval.utils import Rubric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
except ImportError:  # pragma: no cover - allows pure deterministic helpers to run without DeepEval.
    evaluate = None

    class BaseMetric:  # type: ignore[no-redef]
        pass

    class GEval(BaseMetric):  # type: ignore[no-redef]
        pass

    class LLMTestCase:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class LLMTestCaseParams:  # type: ignore[no-redef]
        INPUT = "input"
        RETRIEVAL_CONTEXT = "retrieval_context"
        EXPECTED_OUTPUT = "expected_output"

    class Rubric:  # type: ignore[no-redef]
        def __init__(self, score_range: tuple[int, int], expected_outcome: str):
            self.score_range = score_range
            self.expected_outcome = expected_outcome

    @dataclass
    class AsyncConfig:  # type: ignore[no-redef]
        run_async: bool = False
        throttle_value: int = 0
        max_concurrent: int = 1

    @dataclass
    class DisplayConfig:  # type: ignore[no-redef]
        show_indicator: bool = False
        print_results: bool = False

    @dataclass
    class ErrorConfig:  # type: ignore[no-redef]
        ignore_errors: bool = False
        skip_on_missing_params: bool = False


__all__ = [
    "evaluate",
    "AsyncConfig",
    "DisplayConfig",
    "ErrorConfig",
    "BaseMetric",
    "GEval",
    "Rubric",
    "LLMTestCase",
    "LLMTestCaseParams",
]
