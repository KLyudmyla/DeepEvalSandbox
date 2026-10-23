"""Layer 2 retrieval evaluation package."""

from .config import (
    DEFAULT_K,
    DEFAULT_METRIC_THRESHOLDS,
    GOLDEN_DATASET_SCHEMA,
    SEMANTIC_RELEVANCE_MAX,
)
from evals.datasets.golden_dataset_l2 import GOLDEN_DATASET, MY_DATASET
from .metrics import (
    HitRateAtKMetric,
    MetadataFidelityMetric,
    MRRMetric,
    NDCGAtKMetric,
    PrecisionAtKMetric,
    RecallAtKMetric,
    RetrievalMetric,
    SemanticDocumentRelevanceMetric,
    SourceDocumentProvenanceMetric,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .run_layer2 import run_layer2_evaluation
from .reporting import (
    aggregate_layer2_report,
    assert_layer2_report_passes,
    resolve_metric_threshold,
)
from .utils import (
    RetrievedDocument,
    build_layer2_test_cases,
    document_id,
    exact_json_equal,
    normalize_retrieved_documents,
    positive_relevance_map,
    stable_unique_ids,
    validate_golden_dataset,
    validate_k,
)

validate_golden_dataset(GOLDEN_DATASET)

__all__ = [
    "DEFAULT_K",
    "DEFAULT_METRIC_THRESHOLDS",
    "SEMANTIC_RELEVANCE_MAX",
    "GOLDEN_DATASET",
    "GOLDEN_DATASET_SCHEMA",
    "MY_DATASET",
    "RetrievedDocument",
    "document_id",
    "stable_unique_ids",
    "validate_k",
    "positive_relevance_map",
    "normalize_retrieved_documents",
    "exact_json_equal",
    "validate_golden_dataset",
    "build_layer2_test_cases",
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "RetrievalMetric",
    "RecallAtKMetric",
    "PrecisionAtKMetric",
    "HitRateAtKMetric",
    "MRRMetric",
    "NDCGAtKMetric",
    "SourceDocumentProvenanceMetric",
    "MetadataFidelityMetric",
    "SemanticDocumentRelevanceMetric",
    "aggregate_layer2_report",
    "assert_layer2_report_passes",
    "resolve_metric_threshold",
    "run_layer2_evaluation",
]
