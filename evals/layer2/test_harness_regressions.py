"""Regression tests for retrieval ordering, duplicates, and reporting integrity."""
from types import SimpleNamespace

from evals.layer2.metrics import RecallAtKMetric, MRRMetric
from evals.layer2.reporting import aggregate_layer2_report
from evals.layer2.utils import build_layer2_test_cases


def _case(retrieved, expected=('a.txt',), k=2):
    golden = [{'id': 'one', 'query': 'query', 'ground_truth_document_ids': list(expected),
               'expected_metadata_filters': {}, 'graded_relevance_scores': {x: 3 for x in expected}, 'k': k}]
    return build_layer2_test_cases(lambda query, k: retrieved, golden)[0]


def test_duplicate_occupies_rank_and_recall_uses_real_top_k():
    case = _case([{'id': 'b.txt'}, {'id': 'b.txt'}, {'id': 'a.txt'}])
    assert RecallAtKMetric().measure(case) == 0
    assert MRRMetric().measure(case) == 0


# The expected document is a.txt. Although a.txt has the higher score, it appears second in the retriever’s output, so its reciprocal rank is: 1 / 2 = 0.5
def test_retriever_order_is_preserved_even_with_scores():
    case = _case([{'id': 'b.txt', 'score': 0.1}, {'id': 'a.txt', 'score': 0.9}])
    assert MRRMetric().measure(case) == 0.5


def test_empty_metrics_cannot_pass():
    case = SimpleNamespace(name='one', input='query', additional_metadata={})
    result = SimpleNamespace(name='one', input='query', metrics_data=[])
    report = aggregate_layer2_report(SimpleNamespace(test_results=[result]), [case])
    assert not report['passed']


#  This represents an unsupported query with no ground-truth documents. The metric has a score of 1 and appears successful, but the report should
#  mark it as: applicable == False
def test_negative_ranking_metrics_are_not_counted_as_success():
    case = SimpleNamespace(name='negative', input='query', additional_metadata={'ground_truth_document_ids': []})
    metric = SimpleNamespace(name='Recall@k', score=1, threshold=0.8, error=None, success=True)
    result = SimpleNamespace(name='negative', input='query', metrics_data=[metric])
    report = aggregate_layer2_report(SimpleNamespace(test_results=[result]), [case], thresholds={'Recall@k': 0.8})
    assert report['cases'][0]['metrics'][0]['applicable'] is False
    assert 'Recall@k' not in report['metric_totals']
