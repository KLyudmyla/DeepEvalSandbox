"""Layer 2 document-retrieval golden dataset."""

from __future__ import annotations

from typing import Any


# Source IDs are the filenames emitted by app.agent.DocumentSearchAgent.
GOLDEN_DATASET: list[dict[str, Any]] = [
    {
        "id": "engineering-preferred-framework",
        "query": "Which framework is preferred for new Python APIs?",
        "ground_truth_document_ids": ["engineering_stack.txt"],
        "expected_metadata_filters": {"source": "engineering_stack.txt"},
        "graded_relevance_scores": {"engineering_stack.txt": 3},
        "k": 2,
        "tags": ["single_doc", "fact_retrieval", "engineering"],
    },
    {
        "id": "anniversary-bonus-15-years",
        "query": "What is the amount of bonus for a 15 year company anniversary?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["single_doc", "amount_retrieval", "compensation"],
    },
    {
        "id": "legalization-spain",
        "query": "Who can help with legalization issues in Spain?",
        "ground_truth_document_ids": ["valid_document.txt"],
        "expected_metadata_filters": {"source": "valid_document.txt"},
        "graded_relevance_scores": {"valid_document.txt": 3},
        "k": 2,
        "tags": ["single_doc", "contact_retrieval", "legalization"],
    },
    {
        "id": "remote-work-policy",
        "query": "What is the company policy for working remotely?",
        "ground_truth_document_ids": ["hr_remote_work.txt"],
        "expected_metadata_filters": {"source": "hr_remote_work.txt"},
        "graded_relevance_scores": {"hr_remote_work.txt": 3},
        "k": 2,
        "tags": ["single_doc", "policy", "hr"],
    },
    {
        "id": "unknown-germany-legalization",
        "query": "Who is the legalization specialist for Germany?",
        "ground_truth_document_ids": [],
        "expected_metadata_filters": {},
        "graded_relevance_scores": {},
        "k": 2,
        "tags": ["abstention", "no_positive_evidence"],
    },
]

# Preserve the original export name.
MY_DATASET = GOLDEN_DATASET


__all__ = ["GOLDEN_DATASET", "MY_DATASET"]
