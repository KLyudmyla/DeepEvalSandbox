"""Deterministic tests for Layer 2 threshold resolution and report gating.It tests whether evaluation results are interpreted correctly"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals.layer2.reporting import (
    aggregate_layer2_report,
    assert_layer2_report_passes,
    resolve_metric_threshold,
)
from evals.layer2.utils import build_layer2_test_cases


def _evaluation(score: float, *, success: bool = False):
    metric = SimpleNamespace(
        name="Recall@k",
        reason="test metric",
        score=score,
        error=None,
        threshold=0.8,
        success=success,
    )
    result = SimpleNamespace(
        test_results=[
            SimpleNamespace(
                input="test query",
                success=success,
                metrics_data=[metric],
            )
        ]
    )
    case = SimpleNamespace(
        name="case-1",
        input="test query",
        additional_metadata={},
    )
    return result, case


def test_case_threshold_takes_precedence_over_global_threshold():
    assert (
        resolve_metric_threshold(
            "Recall@k",
            case_thresholds={"Recall@k": 0.5},
            global_thresholds={"Recall@k": 0.8},
        )
        == 0.5
    )


def test_global_threshold_is_used_when_case_has_no_override():
    assert (
        resolve_metric_threshold(
            "Recall@k",
            case_thresholds={},
            global_thresholds={"Recall@k": 0.8},
        )
        == 0.8
    )


def test_case_thresholds_are_propagated_to_test_case_metadata():
    dataset = [
        {
            "id": "case-1",
            "query": "test query",
            "ground_truth_document_ids": ["doc-1"],
            "expected_metadata_filters": {},
            "graded_relevance_scores": {"doc-1": 3},
            "thresholds": {"Recall@k": 0.25},
        }
    ]

    cases = build_layer2_test_cases(lambda query, k: [{"id": "doc-1"}], dataset)

    assert cases[0].additional_metadata["thresholds"] == {"Recall@k": 0.25}


def test_report_uses_case_threshold_for_success_and_failure_gate():
    evaluation, case = _evaluation(0.75)
    case.additional_metadata = {"thresholds": {"Recall@k": 0.5}}

    report = aggregate_layer2_report(
        evaluation,
        [case],
        thresholds={"Recall@k": 0.8},
    )

    metric = report["cases"][0]["metrics"][0]
    assert metric["threshold"] == 0.5
    assert metric["success"] is True
    assert report["passed"] is True
    assert_layer2_report_passes(report)


def test_report_falls_back_to_global_threshold_and_fails():
    evaluation, case = _evaluation(0.75)

    report = aggregate_layer2_report(
        evaluation,
        [case],
        thresholds={"Recall@k": 0.8},
    )

    metric = report["cases"][0]["metrics"][0]
    assert metric["threshold"] == 0.8
    assert metric["success"] is False
    assert report["passed"] is False
    with pytest.raises(AssertionError, match="case-1 / Recall@k"):
        assert_layer2_report_passes(report)
