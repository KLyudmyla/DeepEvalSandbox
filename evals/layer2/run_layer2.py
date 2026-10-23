"""Layer 2 evaluation orchestration using standard DeepEval ``evaluate()``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from evals.datasets.golden_dataset_l2 import GOLDEN_DATASET

from .config import DEFAULT_METRIC_THRESHOLDS
from .deepeval_bridge import (
    AsyncConfig,
    BaseMetric,
    DisplayConfig,
    ErrorConfig,
    evaluate,
)
from .metrics import (
    HitRateAtKMetric,
    MRRMetric,
    NDCGAtKMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
    SemanticDocumentRelevanceMetric,
    SourceDocumentProvenanceMetric,
    MetadataFidelityMetric,
)
from .reporting import aggregate_layer2_report, assert_layer2_report_passes
from .utils import build_layer2_test_cases


def run_layer2_evaluation(
    agent_or_retriever: Any,
    dataset: Sequence[Mapping[str, Any]] = GOLDEN_DATASET,
    *,
    judge_model: str | Any | None = None,
    report_path: str | Path | None = None,
    fail_on_threshold: bool = True,
    metric_thresholds: Mapping[str, float] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run the complete Layer 2 suite through standard ``evaluate()``.

    ``show_progress`` controls DeepEval's terminal indicator and per-test
    result output. It is enabled by default so interactive pytest runs can
    show progress when invoked with ``-s``.
    """

    if evaluate is None:
        raise RuntimeError("deepeval is required to run run_layer2_evaluation()")
    test_cases = build_layer2_test_cases(agent_or_retriever, dataset)
    thresholds = {**DEFAULT_METRIC_THRESHOLDS, **dict(metric_thresholds or {})}
    metrics: list[BaseMetric] = [
        RecallAtKMetric(thresholds["Recall@k"]),
        PrecisionAtKMetric(thresholds["Precision@k"]),
        HitRateAtKMetric(thresholds["Hit Rate@k"]),
        MRRMetric(thresholds["MRR"]),
        NDCGAtKMetric(thresholds["nDCG@k"]),
        SourceDocumentProvenanceMetric(thresholds["Source document ID provenance"]),
        MetadataFidelityMetric(thresholds["Metadata key-value fidelity"]),
    ]
    if judge_model is not None:
        metrics.append(
            SemanticDocumentRelevanceMetric(
                model=judge_model,
                threshold=thresholds["Semantic document relevance (0-3)"],
            )
        )

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        identifier="layer2-document-retrieval",
        async_config=AsyncConfig(run_async=False, throttle_value=0, max_concurrent=1),
        display_config=DisplayConfig(
            show_indicator=show_progress,
            print_results=show_progress,
        ),
        error_config=ErrorConfig(ignore_errors=False, skip_on_missing_params=False),
    )
    report = aggregate_layer2_report(result, test_cases, thresholds=thresholds)
    if report_path is not None:
        output = Path(report_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if fail_on_threshold:
        assert_layer2_report_passes(report)
    return report


__all__ = ["run_layer2_evaluation"]
