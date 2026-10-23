"""Offline regression checks for the four retrieval evaluation correctness fixes."""

from evals.datasets.golden_dataset_l2 import GOLDEN_DATASET
from evals.layer2.metrics import (
    MetadataFidelityMetric, SourceDocumentProvenanceMetric,
    recall_at_k, precision_at_k, hit_rate_at_k, reciprocal_rank, ndcg_at_k,
)
from evals.layer2.utils import build_layer2_test_cases, validate_golden_dataset


def test_dataset_is_consistent():
    validate_golden_dataset(GOLDEN_DATASET)
    assert len(GOLDEN_DATASET) == 45


def test_negative_vector_results_are_not_treated_as_retrieval_failure():
    assert recall_at_k(['unrelated.txt'], [], 2) == 1
    assert precision_at_k(['unrelated.txt'], [], 2) == 1
    assert hit_rate_at_k(['unrelated.txt'], [], 2) == 1
    assert reciprocal_rank(['unrelated.txt'], [], 2) == 1
    assert ndcg_at_k(['unrelated.txt'], {}, 2) == 1


def test_provenance_is_order_independent_and_rejects_wrong_source():
    dataset = [{"id": "two", "query": "query", "ground_truth_document_ids": ["a.txt", "b.txt"],
                "expected_metadata_filters": {}, "graded_relevance_scores": {"a.txt": 3, "b.txt": 3}, "k": 2}]
    def retriever(query, k):
        return [{"id": "b.txt", "metadata": {"source": "b.txt"}},
                {"id": "a.txt", "metadata": {"source": "a.txt"}}]
    case = build_layer2_test_cases(retriever, dataset)[0]
    assert SourceDocumentProvenanceMetric().measure(case) == 1
    case.additional_metadata['retrieved_metadata'][0]['source'] = 'wrong.txt'
    assert SourceDocumentProvenanceMetric().measure(case) == 0


def test_metadata_must_match_expected_document_not_just_any_result():
    dataset = [{"id": "one", "query": "query", "ground_truth_document_ids": ["a.txt"],
                "expected_metadata_filters": {"team": "A"},
                "graded_relevance_scores": {"a.txt": 3}, "k": 2}]
    def retriever(query, k):
        return [{"id": "a.txt", "metadata": {"source": "a.txt", "team": "B"}},
                {"id": "b.txt", "metadata": {"source": "b.txt", "team": "A"}}]
    case = build_layer2_test_cases(retriever, dataset)[0]
    assert MetadataFidelityMetric().measure(case) == 0
    case.additional_metadata['retrieved_metadata'][0]['team'] = 'A'
    assert MetadataFidelityMetric().measure(case) == 1

#  only the first two results count, that is why grounded_truth doc which is at place 3  produces 0 score
def test_duplicate_results_do_not_shift_top_k():
    from evals.layer2.metrics import recall_at_k
    assert recall_at_k(['a.txt', 'a.txt', 'b.txt'], ['b.txt'], 2) == 0.0


def test_rank_scores_do_not_reorder_results():
    from evals.layer2.utils import normalize_retrieved_documents
    rows = [{'id': 'a', 'score': 0.1}, {'id': 'b', 'score': 0.9}]
    assert [d.document_id for d in normalize_retrieved_documents(rows)] == ['a', 'b']


def test_dataset_weak_distractor_is_not_binary_positive():
    case = next(c for c in GOLDEN_DATASET if c['id'] == 'distractor-laptop-vs-kitchen-cleaning')
    assert case['ground_truth_document_ids'] == ['office_kitchen.txt']


def test_empty_dataset_rejected():
    import pytest
    with pytest.raises(ValueError):
        validate_golden_dataset([])
