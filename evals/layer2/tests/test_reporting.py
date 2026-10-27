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
        additional_metadata={
            "query_type": "answerable",
            "ground_truth_document_ids": ["doc-1"]
        },
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
            "query_type": "answerable",
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
    case.additional_metadata = {"thresholds": {"Recall@k": 0.5}, "query_type": "answerable"}

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


def test_judge_cost_is_summed_once_from_per_case_metric_rows():
    cases = [
        SimpleNamespace(name="case-1", input="one", additional_metadata={"query_type": "answerable","ground_truth_document_ids": ["a"]}),
        SimpleNamespace(name="case-2", input="two", additional_metadata={"query_type": "answerable","ground_truth_document_ids": ["a"]}),
    ]
    results = SimpleNamespace(test_results=[
        SimpleNamespace(
            name="case-1",
            input="one",
            metrics_data=[SimpleNamespace(
                name="Semantic document relevance (0-3) [GEval]",
                score=3,
                reason="ok",
                error=None,
                evaluationCost=0.10,
                evaluationModel="gpt-4o-mini",
            )],
        ),
        SimpleNamespace(
            name="case-2",
            input="two",
            metrics_data=[SimpleNamespace(
                name="Semantic document relevance (0-3) [GEval]",
                score=3,
                reason="ok",
                error=None,
                evaluationCost=0.20,
                evaluationModel="gpt-4o-mini",
            )],
        ),
    ])

    report = aggregate_layer2_report(
        results,
        cases,
        thresholds={"Semantic document relevance (0-3)": 2.0},
        expected_metrics=["Semantic document relevance (0-3)"],
        judge_enabled=True,
    )

    assert report["costs"]["judge"]["total_usd"] == pytest.approx(0.30)
    assert report["costs"]["embeddings"]["total_usd"] is None


def test_budget_fails_safely_when_judge_cost_is_unknown():
    evaluation, case = _evaluation(0.75)
    metric = evaluation.test_results[0].metrics_data[0]
    metric.name = "Semantic document relevance (0-3) [GEval]"
    metric.evaluationModel = "custom-judge"
    metric.evaluationCost = None

    report = aggregate_layer2_report(
        evaluation,
        [case],
        thresholds={"Semantic document relevance (0-3)": 2.0},
        expected_metrics=["Semantic document relevance (0-3)"],
        judge_enabled=True,
        max_judge_cost_usd=1.0,
    )

    assert report["costs"]["judge"]["total_usd"] is None
    assert report["budget"]["status"] == "unknown_cost"
    assert report["passed"] is False


def test_empty_negative_provenance_is_reported_as_not_applicable():
    dataset = [{
        "id": "negative",
        "query": "query",
        "query_type": "unanswerable",
        "ground_truth_document_ids": [],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {},
        "k": 2,
    }]
    case = build_layer2_test_cases(lambda query, k: [], dataset)[0]
    metric = SimpleNamespace(
        name="Source document ID provenance",
        reason="empty retrieval for negative case",
        score=1.0,
        error=None,
        threshold=1.0,
        success=True,
    )
    result = SimpleNamespace(
        name="negative",
        input="query",
        metrics_data=[metric],
    )

    report = aggregate_layer2_report(
        SimpleNamespace(test_results=[result]),
        [case],
        thresholds={"Source document ID provenance": 1.0},
        expected_metrics=["Source document ID provenance"],
    )

    row = report["cases"][0]["metrics"][0]
    assert row["applicable"] is False
    assert row["success"] is None

