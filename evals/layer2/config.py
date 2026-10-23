"""Layer 2 configuration, schema, and metric thresholds.

If a test case does not specify ``k``, evaluate the top 5 retrieved documents by
default. In ``golden_dataset.py``, ``"k": 2`` overrides this default, so those
cases evaluate only the top 2 results.

Metric annotations:
  * Recall@k: fraction of unique ground-truth document IDs found in the first
    k retrieved results. Formula: hits / number of relevant documents.
  * Precision@k: fraction of the first k result slots whose unique IDs are
    ground-truth IDs. Formula: hits / k (the denominator remains k even when
    fewer than k documents are returned).
  * Hit Rate@k: whether at least one ground-truth document ID appears in the
    first k results; score is 1.0 for a hit and 0.0 otherwise.
  * MRR: reciprocal rank of the first ground-truth document in the ranked
    results (1 / rank), or 0.0 when no relevant document is found.
  * nDCG@k: ranking quality using graded relevance scores from 0 to 3. It
    compares discounted cumulative gain in the first k results with the ideal
    ordering; gain is ``(2**relevance - 1) / log2(rank + 1)``.
  * Source document ID provenance: verifies that every retrieved metadata row
    exists and its ``source`` field matches the corresponding retrieved ID.
  * Metadata key-value fidelity: verifies expected full metadata rows or
    expected key-value filters; with no expectation, it checks row count.
  * Semantic document relevance (0-3): an LLM judge evaluates the query,
    retrieved context, and golden relevance map, then assigns an integer score
    for semantic relevance, evidence coverage, and ranking quality.

For unsupported queries with no ground-truth IDs, deterministic retrieval
metrics return 1.0 as not-applicable and are reported separately. The semantic
metric treats an empty retrieval for such a query as fully correct (3.0).
"""

from __future__ import annotations

from typing import Any

DEFAULT_K = 5
SEMANTIC_RELEVANCE_MAX = 3
DEFAULT_METRIC_THRESHOLDS: dict[str, float] = {
    # Fraction of expected unique documents retrieved in the top k.
    "Recall@k": 0.8,
    # Fraction of the k result slots containing an expected document.
    "Precision@k": 0.5,
    # Binary score: at least one expected document appears in the top k.
    "Hit Rate@k": 1.0,
    # Reciprocal rank of the first expected document: 1 / its rank.
    "MRR": 0.5,
    # Graded ranking quality normalized against the ideal top-k ordering.
    "nDCG@k": 0.5,
    # Every returned metadata row has a source ID matching its retrieved ID.
    "Source document ID provenance": 1.0,
    # Expected metadata rows/filters match retrieved metadata exactly.
    "Metadata key-value fidelity": 1.0,
    # LLM-judged semantic relevance and evidence coverage on a 0–3 scale.
    "Semantic document relevance (0-3)": 2.0,
}

GOLDEN_DATASET_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Layer 2 Document Retrieval Golden Dataset",
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "query",
            "ground_truth_document_ids",
            "expected_metadata_filters",
            "graded_relevance_scores",
        ],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "query": {"type": "string", "minLength": 1},
            "ground_truth_document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "expected_metadata_filters": {"type": "object", "additionalProperties": {}},
            "graded_relevance_scores": {
                "type": "object",
                "additionalProperties": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                },
            },
            "k": {"type": "integer", "minimum": 0},
            "expected_metadata": {"type": "array", "items": {"type": "object"}},
            "thresholds": {"type": "object", "additionalProperties": {"type": "number"}},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
    },
}

__all__ = [
    "DEFAULT_K",
    "SEMANTIC_RELEVANCE_MAX",
    "DEFAULT_METRIC_THRESHOLDS",
    "GOLDEN_DATASET_SCHEMA",
]
