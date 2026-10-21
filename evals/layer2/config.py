"""Layer 2 configuration, schema, and metric thresholds."""

from __future__ import annotations

from typing import Any

'''If a test case does not specify "k", evaluate the top 5 retrieved documents by default. In golden_dataset.py, "k": 2 
overrides this default, so those cases evaluate only the top 2 results.
 semantic relevance scores can range from 0 to 3:
   0 = irrelevant
  1 = weakly relevant
  2 = substantially relevant
  3 = strongly/fully relevant
'''

DEFAULT_K = 5
SEMANTIC_RELEVANCE_MAX = 3
DEFAULT_METRIC_THRESHOLDS: dict[str, float] = {
    "Recall@k": 0.8,
    "Precision@k": 0.5,
    "Hit Rate@k": 1.0,
    "MRR": 0.5,
    "nDCG@k": 0.5,
    "Source document ID provenance": 1.0,
    "Metadata key-value fidelity": 1.0,
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
