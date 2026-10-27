"""Layer 2 evaluation orchestration using standard DeepEval ``evaluate()``."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from types import SimpleNamespace

from evals.datasets.golden_dataset_l2 import GOLDEN_DATASET

from .config import DEFAULT_JUDGE_MODEL, DEFAULT_METRIC_THRESHOLDS
from .deepeval_bridge import (
    AsyncConfig,
    BaseMetric,
    DisplayConfig,
    ErrorConfig,
    evaluate,
)
from .metrics import (
    HitRateAtKMetric,
    DuplicateFreeRateAtKMetric,
    MRRMetric,
    NDCGAtKMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
    SemanticDocumentRelevanceMetric,
    SourceDocumentProvenanceMetric,
    MetadataFidelityMetric,
)
from .deepeval_bridge import (
    AsyncConfig,
    BaseMetric,
    DisplayConfig,
    ErrorConfig,
    LLMTestCase,
    evaluate,
)
from .reporting import aggregate_layer2_report, assert_layer2_report_passes
from .utils import build_layer2_test_cases


def run_layer2_evaluation(
    agent_or_retriever: Any,
    dataset: Sequence[Mapping[str, Any]] = GOLDEN_DATASET,
    *,
    judge_model: str | Any | None = DEFAULT_JUDGE_MODEL,
    report_path: str | Path | None = None,
    fail_on_threshold: bool = True,
    metric_thresholds: Mapping[str, float] | None = None,
    max_judge_cost_usd: float | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Run Layer 2 retrieval evaluation.

    GEval semantic relevance is executed only for answerable queries.

    All applicable deterministic retrieval metrics are evaluated for
    both answerable and unanswerable queries.

    The final report preserves all Golden Dataset cases.
    """

    if evaluate is None:
        raise RuntimeError(
            "deepeval is required to run run_layer2_evaluation()"
        )

    if max_judge_cost_usd is not None:
        if (
            not math.isfinite(float(max_judge_cost_usd))
            or max_judge_cost_usd < 0
        ):
            raise ValueError(
                "max_judge_cost_usd must be a finite non-negative number"
            )

    # ---------------------------------------------------------
    # 1. Build all test cases once
    # ---------------------------------------------------------

    test_cases = build_layer2_test_cases(
        agent_or_retriever,
        dataset,
    )

    thresholds = {
        **DEFAULT_METRIC_THRESHOLDS,
        **dict(metric_thresholds or {}),
    }

    # ---------------------------------------------------------
    # 2. Split cases by query type
    # ---------------------------------------------------------

    answerable_cases = []
    unanswerable_cases = []

    for case in test_cases:
        metadata = case.additional_metadata or {}

        query_type = metadata.get("query_type")

        if query_type == "answerable":
            answerable_cases.append(case)

        elif query_type == "unanswerable":
            unanswerable_cases.append(case)

        else:
            raise ValueError(
                f"{case.name}: invalid query_type={query_type!r}"
            )

    # ---------------------------------------------------------
    # 3. Define deterministic metrics
    # ---------------------------------------------------------

    def create_deterministic_metrics() -> list[BaseMetric]:
        return [
            RecallAtKMetric(
                thresholds["Recall@k"]
            ),
            PrecisionAtKMetric(
                thresholds["Precision@k"]
            ),
            HitRateAtKMetric(
                thresholds["Hit Rate@k"]
            ),
            MRRMetric(
                thresholds["MRR"]
            ),
            NDCGAtKMetric(
                thresholds["nDCG@k"]
            ),
            DuplicateFreeRateAtKMetric(
                thresholds["Duplicate-free rate@k"]
            ),
            SourceDocumentProvenanceMetric(
                thresholds["Source document ID provenance"]
            ),
            MetadataFidelityMetric(
                thresholds["Metadata key-value fidelity"]
            ),
        ]

    expected_metrics = [
        "Recall@k",
        "Precision@k",
        "Hit Rate@k",
        "MRR",
        "nDCG@k",
        "Duplicate-free rate@k",
        "Source document ID provenance",
        "Metadata key-value fidelity",
    ]

    if judge_model is not None:
        expected_metrics.append(
            "Semantic document relevance (0-3)"
        )

    # ---------------------------------------------------------
    # 4. Common DeepEval execution helper
    # ---------------------------------------------------------

    def execute_group(
        cases: Sequence[LLMTestCase],
        metrics: list[BaseMetric],
        identifier: str,
    ) -> Any:

        return evaluate(
            test_cases=list(cases),
            metrics=metrics,
            identifier=identifier,
            async_config=AsyncConfig(
                run_async=False,
                throttle_value=0,
                max_concurrent=1,
            ),
            display_config=DisplayConfig(
                show_indicator=show_progress,
                print_results=False,
            ),
            error_config=ErrorConfig(
                ignore_errors=False,
                skip_on_missing_params=False,
            ),
        )

    # ---------------------------------------------------------
    # 5. Execute answerable cases with GEval
    # ---------------------------------------------------------

    evaluation_results = []

    if answerable_cases:

        answerable_metrics = create_deterministic_metrics()

        if judge_model is not None:
            answerable_metrics.append(
                SemanticDocumentRelevanceMetric(
                    model=judge_model,
                    threshold=thresholds[
                        "Semantic document relevance (0-3)"
                    ],
                )
            )

        answerable_result = execute_group(
            answerable_cases,
            answerable_metrics,
            "layer2-answerable-retrieval",
        )

        evaluation_results.append(answerable_result)

    # ---------------------------------------------------------
    # 6. Execute unanswerable cases WITHOUT GEval
    # ---------------------------------------------------------

    if unanswerable_cases:

        unanswerable_metrics = create_deterministic_metrics()

        unanswerable_result = execute_group(
            unanswerable_cases,
            unanswerable_metrics,
            "layer2-unanswerable-retrieval",
        )

        evaluation_results.append(unanswerable_result)

    # ---------------------------------------------------------
    # 7. Combine DeepEval results
    # ---------------------------------------------------------

    combined_test_results = []

    for evaluation_result in evaluation_results:
        combined_test_results.extend(
            getattr(
                evaluation_result,
                "test_results",
                [],
            ) or []
        )

    combined_result = SimpleNamespace(
        test_results=combined_test_results,
        confident_link=None,
        test_run_id=None,
    )

    # ---------------------------------------------------------
    # 8. Build unified report
    # ---------------------------------------------------------

    report = aggregate_layer2_report(
        combined_result,
        test_cases,
        thresholds=thresholds,
        expected_metrics=expected_metrics,
        judge_enabled=judge_model is not None,
        max_judge_cost_usd=max_judge_cost_usd,
        index_info=getattr(
            agent_or_retriever,
            "_layer2_index_info",
            None,
        ),
    )

    # ---------------------------------------------------------
    # 9. Save report
    # ---------------------------------------------------------

    if report_path is not None:

        output = Path(report_path)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(
                report,
                indent=2,
                sort_keys=True,
                default=str,
            ) + "\n",
            encoding="utf-8",
        )

    # ---------------------------------------------------------
    # 10. Apply quality gate
    # ---------------------------------------------------------

    if fail_on_threshold:
        assert_layer2_report_passes(report)

    return report


__all__ = ["run_layer2_evaluation"]
