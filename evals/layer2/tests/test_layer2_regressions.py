"""Regression tests for Layer 2 metrics, reporting, and dataset validation."""
from types import SimpleNamespace

import pytest

from evals.layer2.metrics import MRRMetric, MetadataFidelityMetric, RecallAtKMetric
from evals.layer2.reporting import aggregate_layer2_report
from evals.layer2.utils import build_layer2_test_cases, validate_golden_dataset


def _case(retrieved, expected=('a.txt',), k=2):
    golden = [{'id': 'one', 'query': 'query', 'query_type': 'answerable' if expected else 'unanswerable',
               'ground_truth_document_ids': list(expected),
               'expected_metadata_filters': {}, 'graded_relevance_scores': {x: 3 for x in expected}, 'k': k}]
    return build_layer2_test_cases(lambda query, k: retrieved, golden)[0]


def _report_case(name, *, query_type='answerable', ground_truth_document_ids=None, **metadata):
    if ground_truth_document_ids is None:
        ground_truth_document_ids = ['a.txt'] if query_type == 'answerable' else []
    return SimpleNamespace(
        name=name,
        input='query',
        additional_metadata={
            'query_type': query_type,
            'ground_truth_document_ids': ground_truth_document_ids,
            **metadata,
        },
    )


def test_duplicate_metric_rows_rejected():
    case = _report_case('one', ground_truth_document_ids=['doc-1'])
    metric = SimpleNamespace(name='Recall@k', score=1.0, error=None, threshold=0.8)
    result = SimpleNamespace(name='one', metrics_data=[metric, metric])
    with pytest.raises(ValueError, match='Duplicate metric'):
        aggregate_layer2_report(SimpleNamespace(test_results=[result]), [case])


def test_invalid_metadata_row_fails():
    case = SimpleNamespace(additional_metadata={
        'retrieved_document_ids': ['a.txt'],
        'retrieved_metadata': [None],
        'ground_truth_document_ids': ['a.txt'],
        'expected_metadata_filters': {},
    })
    assert MetadataFidelityMetric().measure(case) == 0.0


def test_query_type_is_accepted_by_schema():
    dataset = [{
        "id": "valid-answerable",
        "query": "Which framework is preferred?",
        "ground_truth_document_ids": ["engineering_stack.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {
            "engineering_stack.txt": 3
        },
        "query_type": "answerable",
        "k": 2,
    }]

    validate_golden_dataset(dataset)


def test_invalid_query_type_is_rejected():
    dataset = [{
        "id": "invalid-type",
        "query": "Which framework is preferred?",
        "ground_truth_document_ids": ["engineering_stack.txt"],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {
            "engineering_stack.txt": 3
        },
        "query_type": "unknown",
        "k": 2,
    }]

    with pytest.raises(ValueError, match="Golden dataset schema error"):
        validate_golden_dataset(dataset)


def test_inconsistent_query_type_is_rejected():
    dataset = [{
        "id": "invalid-classification",
        "query": "Who is the specialist for Germany?",
        "ground_truth_document_ids": [],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {},
        "query_type": "answerable",
        "k": 2,
    }]

    with pytest.raises(ValueError, match="inconsistent"):
        validate_golden_dataset(dataset)


def test_duplicate_occupies_rank_and_recall_uses_real_top_k():
    case = _case([{'id': 'b.txt'}, {'id': 'b.txt'}, {'id': 'a.txt'}])
    assert RecallAtKMetric().measure(case) == 0
    assert MRRMetric().measure(case) == 0


# The expected document is a.txt. Although a.txt has the higher score, it appears second in the retriever's output, so its reciprocal rank is: 1 / 2 = 0.5
def test_retriever_order_is_preserved_even_with_scores():
    case = _case([{'id': 'b.txt', 'score': 0.1}, {'id': 'a.txt', 'score': 0.9}])
    assert MRRMetric().measure(case) == 0.5


def test_empty_metrics_cannot_pass():
    case = _report_case('one')
    result = SimpleNamespace(name='one', input='query', metrics_data=[])
    report = aggregate_layer2_report(SimpleNamespace(test_results=[result]), [case])
    assert report['passed'] is False
    assert report['cases'][0]['success'] is False
    assert report['cases'][0]['metrics'] == []


# This represents an unsupported query with no ground-truth documents. The metric has a score of 1 and appears successful, but the report should
# mark it as: applicable == False
def test_negative_ranking_metrics_are_not_counted_as_success():
    case = _report_case('negative', query_type='unanswerable')
    metric = SimpleNamespace(name='Recall@k', score=1, threshold=0.8, error=None, success=True)
    result = SimpleNamespace(name='negative', input='query', metrics_data=[metric])
    report = aggregate_layer2_report(SimpleNamespace(test_results=[result]), [case], thresholds={'Recall@k': 0.8})
    assert report['cases'][0]['metrics'][0]['applicable'] is False
    assert 'Recall@k' not in report['metric_totals']
